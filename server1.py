```python name=server1.py url=https://github.com/yashnaiduu/NeuroScan-Brain-Tumor-Classification/blob/e1b6101b3bbaecfd13a4c2795d211c9387533ec9/server1.py
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import os
import io
import cv2
import base64
import random
import logging
import numpy as np
import tensorflow as tf
from PIL import Image

import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Config
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', '/tmp/Uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['DATASET_PATH'] = os.getenv('DATASET_PATH', './Dataset')
app.config['MODEL_PATH'] = os.getenv('MODEL_PATH', 'mobilenet_brain_tumor_classifier.h5')

MODEL_CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
REPORTING_CLASS_NAMES = MODEL_CLASS_NAMES + ['not_mri']

# Gemini configuration (used only to validate "is this a brain MRI?")
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
gemini_vision_model = None
try:
    genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
    gemini_vision_model = genai.GenerativeModel(GEMINI_MODEL)
    logger.info(f"Gemini API configured successfully. model={GEMINI_MODEL}")
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    gemini_vision_model = None

# Load classification model once
try:
    model = tf.keras.models.load_model(app.config['MODEL_PATH'])
    logger.info("Brain tumor classification model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load classification model: {str(e)}")
    raise SystemExit(1)

# -------- Utils --------
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def cleanup_file(filepath: str) -> None:
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up file: {filepath}")
    except Exception as e:
        logger.error(f"Error cleaning up file {filepath}: {str(e)}")

def is_valid_image(filepath: str) -> bool:
    try:
        Image.open(filepath).verify()
        return True
    except Exception:
        return False

def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def fetch_random_image_path() -> str:
    DATASET_SUBFOLDERS = ['Training', 'Testing']
    dataset_path = app.config['DATASET_PATH']
    dataset_subfolders = [
        os.path.join(dataset_path, sub) for sub in DATASET_SUBFOLDERS if os.path.isdir(os.path.join(dataset_path, sub))
    ]
    if not dataset_subfolders:
        raise FileNotFoundError(f"No '{DATASET_SUBFOLDERS}' subfolders found in: {dataset_path}")

    available_classes_paths = []
    for subfolder in dataset_subfolders:
        for class_name in MODEL_CLASS_NAMES:
            class_path = os.path.join(subfolder, class_name)
            if os.path.isdir(class_path):
                if any(os.path.isfile(os.path.join(class_path, f)) for f in os.listdir(class_path)):
                    available_classes_paths.append(class_path)
    if not available_classes_paths:
        raise FileNotFoundError(f"No image directories with content found within {DATASET_SUBFOLDERS} and classes {MODEL_CLASS_NAMES} in: {dataset_path}")

    random_class_path = random.choice(available_classes_paths)
    image_files = [f for f in os.listdir(random_class_path) if os.path.isfile(os.path.join(random_class_path, f))]
    if not image_files:
        raise FileNotFoundError(f"No image files found in: {random_class_path}")

    random_image_name = random.choice(image_files)
    return os.path.join(random_class_path, random_image_name)

def format_classification_results(predictions: np.ndarray, class_names: list) -> list:
    preds = predictions.tolist()
    if len(preds) != len(class_names):
        logger.error(f"Prediction length ({len(preds)}) does not match reporting class names length ({len(class_names)})")
        n = min(len(preds), len(class_names))
        pairs = zip(preds[:n], class_names[:n])
    else:
        pairs = zip(preds, class_names)
    classes = [
        {'label': name.replace('_', ' ').capitalize(), 'percent': round(float(p) * 100, 2)}
        for p, name in pairs
    ]
    return sorted(classes, key=lambda x: x['percent'], reverse=True)

def encode_image_to_base64(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return None

# -------- Grad-CAM --------
def get_last_conv_layer_name(m):
    for layer in reversed(m.layers):
        if 'conv' in layer.name.lower() and isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(m.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    raise ValueError("No convolutional layer found in model for Grad-CAM")

grad_model = None
def initialize_grad_model():
    global grad_model
    if grad_model is not None:
        return
    try:
        last_conv = get_last_conv_layer_name(model)
        layer_out = model.get_layer(last_conv).output
        grad_model = tf.keras.models.Model([model.inputs], [layer_out, model.output])
        logger.info(f"Grad-CAM model initialized using layer: {last_conv}")
    except Exception as e:
        logger.error(f"Could not initialize Grad-CAM model: {str(e)}")
        grad_model = None

def generate_gradcam(img_array: np.ndarray, class_index: int) -> np.ndarray:
    if grad_model is None:
        initialize_grad_model()
        if grad_model is None:
            raise RuntimeError("Grad-CAM model is not initialized.")

    if not (0 <= class_index < len(MODEL_CLASS_NAMES)):
        raise ValueError(f"Invalid class index {class_index} for Grad-CAM. Must be between 0 and {len(MODEL_CLASS_NAMES)-1}.")

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val == 0:
        logger.warning("Max heatmap value is 0, cannot normalize.")
        return np.zeros((224, 224, 3), dtype=np.uint8)

    heatmap /= max_val
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

# -------- Gemini helper --------
def check_if_mri_with_gemini(image_bytes: bytes) -> dict:
    if not gemini_vision_model:
        logger.warning("Gemini API not available. Skipping MRI validation and proceeding to model.")
        return {'used': False, 'is_mri': True, 'raw': None}
    try:
        image_pil = Image.open(io.BytesIO(image_bytes))
        prompt = (
            "Analyze this image. Is it a medical image, specifically a brain MRI scan of a human? "
            "Respond ONLY with 'YES_MRI' if it is clearly a human brain MRI scan, and ONLY with 'NO_MRI' otherwise."
        )
        response = gemini_vision_model.generate_content([prompt, image_pil])
        text_response = (getattr(response, 'text', '') or '').strip().upper()
        logger.info(f"Gemini raw response text: '{text_response}'")
        return {'used': True, 'is_mri': text_response == 'YES_MRI', 'raw': text_response}
    except Exception as e:
        logger.error(f"Error calling Gemini API for validation: {str(e)}. Proceeding to model.")
        return {'used': True, 'is_mri': True, 'raw': None}

# -------- Routes --------
@app.route('/')
def home():
    return render_template('NeuroScan.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filepath = None
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved uploaded file: {filepath}")

        if not is_valid_image(filepath):
            logger.warning(f"Uploaded file is not a valid image: {filepath}")
            return jsonify({'error': 'Uploaded file is not a valid image'}), 400

        gem_info = {'used': False, 'raw': None}
        try:
            with open(filepath, "rb") as f:
                image_bytes = f.read()
            gem = check_if_mri_with_gemini(image_bytes)
            gem_info = {'used': gem['used'], 'raw': gem['raw']}
            if not gem['is_mri']:
                logger.info("Gemini classified image as NOT a Brain MRI.")
                not_mri_preds = np.zeros(len(REPORTING_CLASS_NAMES))
                not_mri_preds[REPORTING_CLASS_NAMES.index('not_mri')] = 1.0
                classes = format_classification_results(not_mri_preds, REPORTING_CLASS_NAMES)
                return jsonify({
                    'class': 'Likely Not a Brain MRI Scan',
                    'confidence': 1.0,
                    'classes': classes,
                    'gemini': gem_info
                }), 200
        except Exception as e:
            logger.error(f"Error during Gemini check preparation: {str(e)}. Assuming it's an MRI.")

        logger.info("Proceeding with tumor classification using the local model.")
        processed_image = preprocess_image(filepath)
        predictions = model.predict(processed_image, verbose=0)[0]  # shape (4,)
        full_predictions_for_reporting = np.append(predictions, 1e-6)  # add tiny prob for 'not_mri'
        idx = int(np.argmax(predictions))
        predicted_class_name = MODEL_CLASS_NAMES[idx]
        confidence = float(predictions[idx])
        classes = format_classification_results(full_predictions_for_reporting, REPORTING_CLASS_NAMES)

        return jsonify({
            'class': predicted_class_name.replace('_', ' ').capitalize(),
            'confidence': confidence,
            'classes': classes,
            'gemini': gem_info
        }), 200

    except cv2.error:
        logger.error(f"OpenCV error processing image: {filepath}")
        return jsonify({'error': 'Image processing error (OpenCV)'}), 400
    except tf.errors.OpError:
        logger.error("TensorFlow OpError during model prediction.")
        return jsonify({'error': 'Model prediction failed'}), 500
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': f'Unexpected server error: {str(e)}'}), 500
    finally:
        if filepath:
            cleanup_file(filepath)

@app.route('/random', methods=['GET'])
def random_prediction():
    try:
        random_image_path = fetch_random_image_path()
        logger.info(f"Fetching random image: {random_image_path}")
        processed_image = preprocess_image(random_image_path)
        predictions = model.predict(processed_image, verbose=0)[0]
        full_predictions_for_reporting = np.append(predictions, 1e-6)
        idx = int(np.argmax(predictions))
        predicted_class_name = MODEL_CLASS_NAMES[idx]
        confidence = float(predictions[idx])
        classes = format_classification_results(full_predictions_for_reporting, REPORTING_CLASS_NAMES)
        base64_image = encode_image_to_base64(random_image_path)
        if not base64_image:
            return jsonify({'error': 'Failed to encode random image'}), 500
        return jsonify({
            'class': predicted_class_name.replace('_', ' ').capitalize(),
            'confidence': confidence,
            'classes': classes,
            'image': base64_image,
            'gemini': {'used': False, 'raw': None}
        }), 200
    except FileNotFoundError as e:
        logger.error(f"Dataset or random image not found: {str(e)}")
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Unexpected error during random prediction: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing random image: {str(e)}'}), 500

@app.route('/heatmap', methods=['POST'])
def get_heatmap():
    if 'file' not in request.files:
        logger.warning("No file provided in request for heatmap.")
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filepath = None
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved uploaded file for heatmap: {filepath}")

        if not is_valid_image(filepath):
            logger.warning(f"Uploaded file for heatmap is not a valid image: {filepath}")
            return jsonify({'error': 'Uploaded file is not a valid image'}), 400

        try:
            with open(filepath, "rb") as f:
                image_bytes = f.read()
            gem = check_if_mri_with_gemini(image_bytes)
            if not gem['is_mri']:
                logger.warning("Gemini classified image as NOT a Brain MRI. Cannot generate heatmap.")
                return jsonify({'error': 'Cannot generate heatmap for non-MRI images', 'gemini': {'used': gem['used'], 'raw': gem['raw']}}), 400
        except Exception as e:
            logger.error(f"Error during Gemini check preparation for heatmap: {str(e)}. Assuming it's an MRI.")

        logger.info("Proceeding with heatmap generation using the local model.")
        processed_image = preprocess_image(filepath)
        predictions = model.predict(processed_image, verbose=0)[0]
        class_index_for_heatmap = int(np.argmax(predictions))
        heatmap = generate_gradcam(processed_image, class_index_for_heatmap)
        _, buffer = cv2.imencode('.png', heatmap)
        encoded_heatmap = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'heatmap': encoded_heatmap}), 200

    except cv2.error:
        logger.error(f"OpenCV error processing image for heatmap: {filepath}")
        return jsonify({'error': 'Image processing error (OpenCV)'}), 400
    except tf.errors.OpError:
        logger.error("TensorFlow OpError during heatmap generation.")
        return jsonify({'error': 'Model prediction failed during heatmap generation'}), 500
    except RuntimeError as e:
        logger.error(f"Runtime error during heatmap generation: {str(e)}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error during heatmap generation: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error during heatmap generation'}), 500
    finally:
        if filepath:
            cleanup_file(filepath)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    initialize_grad_model()
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=False, host='0.0.0.0', port=port)
```
