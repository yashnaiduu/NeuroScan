import os
import io
import base64
import random
import logging
import time
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'Uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp'}
app.config['DATASET_PATH'] = os.getenv('DATASET_PATH', './Dataset')
app.config['MODEL_PATH'] = os.getenv('MODEL_PATH', 'mobilenet_brain_tumor_classifier.h5')
app.config['CACHE_FOLDER'] = os.getenv('CACHE_FOLDER', './cache')
app.config['CACHE_DURATION'] = int(os.getenv('CACHE_DURATION', 3600))

# Create necessary directories
os.makedirs(app.config['CACHE_FOLDER'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Model class names
MODEL_CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
REPORTING_CLASS_NAMES = MODEL_CLASS_NAMES + ['not_mri']
DATASET_SUBFOLDERS = ['Training', 'Testing']

# Global variables
model = None
gemini_vision_model = None
grad_model = None
app_start_time = time.time()

# Configure Gemini API
def configure_gemini() -> None:
    """Configure Gemini API if available and API key is set."""
    global gemini_vision_model
    
    if not GEMINI_AVAILABLE:
        logger.warning("google-generativeai package not installed. Gemini features disabled.")
        return
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        logger.warning("GOOGLE_API_KEY not set. Gemini validation will be skipped.")
        return
    
    try:
        genai.configure(api_key=api_key)
        gemini_vision_model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("Gemini API configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {str(e)}")
        gemini_vision_model = None

# Load classification model
def load_classification_model() -> None:
    """Load the brain tumor classification model."""
    global model
    
    model_path = app.config['MODEL_PATH']
    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        logger.info("Please ensure the model file is available or set MODEL_URL environment variable.")
        return
    
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Brain tumor classification model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load classification model: {str(e)}")
        raise

# Initialize Grad-CAM model
def initialize_grad_model() -> None:
    """Initialize the Grad-CAM model for heatmap generation."""
    global grad_model
    
    if model is None:
        logger.warning("Cannot initialize Grad-CAM: classification model not loaded")
        return
    
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

def get_last_conv_layer_name(m: tf.keras.Model) -> str:
    """Find the last convolutional layer in the model."""
    for layer in reversed(m.layers):
        if 'conv' in layer.name.lower() and isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    for layer in reversed(m.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    raise ValueError("No convolutional layer found in model for Grad-CAM")

# Utility functions
def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def is_valid_image(filepath: str) -> bool:
    """Validate that the file is a valid image."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False

def cleanup_file(filepath: str) -> None:
    """Remove a file from the filesystem."""
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up file: {filepath}")
    except Exception as e:
        logger.error(f"Error cleaning up file {filepath}: {str(e)}")

def preprocess_image(image_path: str) -> np.ndarray:
    """Load, resize, and normalize the image for the classification model."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encode an image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return None

def format_classification_results(predictions: np.ndarray, class_names: list) -> list:
    """Format prediction results for API response."""
    preds = predictions.tolist()
    if len(preds) != len(class_names):
        logger.error(f"Prediction length ({len(preds)}) does not match class names length ({len(class_names)})")
        n = min(len(preds), len(class_names))
        pairs = zip(preds[:n], class_names[:n])
    else:
        pairs = zip(preds, class_names)
    
    classes = [
        {'label': name.replace('_', ' ').capitalize(), 'percent': round(float(p) * 100, 2)}
        for p, name in pairs
    ]
    return sorted(classes, key=lambda x: x['percent'], reverse=True)

# Gemini validation function
def check_if_mri_with_gemini(image_bytes: bytes) -> Dict[str, Any]:
    """
    Use Gemini Vision API to check if the image is a brain MRI.
    Returns dict with 'used', 'is_mri', and 'raw' keys.
    """
    if not gemini_vision_model:
        logger.debug("Gemini API not available. Skipping MRI validation.")
        return {'used': False, 'is_mri': True, 'raw': None}
    
    try:
        image_pil = Image.open(io.BytesIO(image_bytes))
        prompt = (
            "Analyze this image. Is it a medical image, specifically a brain MRI scan of a human? "
            "Respond ONLY with 'YES_MRI' if it is clearly a human brain MRI scan, and ONLY with 'NO_MRI' otherwise. "
            "Do not include any other text, explanations, or punctuation."
        )
        response = gemini_vision_model.generate_content([prompt, image_pil])
        text_response = (getattr(response, 'text', '') or '').strip().upper()
        logger.info(f"Gemini raw response text: '{text_response}'")
        return {'used': True, 'is_mri': text_response == 'YES_MRI', 'raw': text_response}
    except Exception as e:
        logger.error(f"Error calling Gemini API for validation: {str(e)}. Proceeding to model.")
        return {'used': True, 'is_mri': True, 'raw': None}

# Dataset functions
def fetch_random_image_path() -> str:
    """Fetch a random image path from the dataset."""
    dataset_path = app.config['DATASET_PATH']
    dataset_subfolders = [
        os.path.join(dataset_path, sub) 
        for sub in DATASET_SUBFOLDERS 
        if os.path.isdir(os.path.join(dataset_path, sub))
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
        raise FileNotFoundError(
            f"No image directories with content found within {DATASET_SUBFOLDERS} "
            f"and classes {MODEL_CLASS_NAMES} in: {dataset_path}"
        )
    
    random_class_path = random.choice(available_classes_paths)
    image_files = [
        f for f in os.listdir(random_class_path) 
        if os.path.isfile(os.path.join(random_class_path, f))
    ]
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in: {random_class_path}")
    
    random_image_name = random.choice(image_files)
    return os.path.join(random_class_path, random_image_name)

# Grad-CAM functions
def generate_gradcam(img_array: np.ndarray, class_index: int) -> np.ndarray:
    """Generate Grad-CAM heatmap for the given image and class."""
    if grad_model is None:
        initialize_grad_model()
        if grad_model is None:
            raise RuntimeError("Grad-CAM model is not initialized.")
    
    if not (0 <= class_index < len(MODEL_CLASS_NAMES)):
        raise ValueError(
            f"Invalid class index {class_index} for Grad-CAM. "
            f"Must be between 0 and {len(MODEL_CLASS_NAMES)-1}."
        )
    
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

# Routes
@app.route('/')
def home():
    """Render the main application page."""
    return render_template('NeuroScan.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    uptime = time.time() - app_start_time
    
    health_status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'gemini_available': gemini_vision_model is not None,
        'uptime': round(uptime, 2),
        'version': '1.0.0'
    }
    
    # Check if model file exists
    if not os.path.exists(app.config['MODEL_PATH']):
        health_status['status'] = 'degraded'
        health_status['warning'] = 'Model file not found'
    
    status_code = 200 if health_status['status'] == 'healthy' else 503
    return jsonify(health_status), status_code

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction."""
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check server configuration.'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Supported formats: PNG, JPG, JPEG, BMP'}), 400
    
    filepath = None
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved uploaded file: {filepath}")
        
        if not is_valid_image(filepath):
            logger.warning(f"Uploaded file is not a valid image: {filepath}")
            return jsonify({'error': 'Uploaded file is not a valid image'}), 400
        
        # Gemini validation
        gemini_result = {'used': False, 'is_mri': True, 'raw': None}
        try:
            with open(filepath, "rb") as f:
                image_bytes = f.read()
            gemini_result = check_if_mri_with_gemini(image_bytes)
        except Exception as e:
            logger.error(f"Error during Gemini check: {str(e)}. Proceeding to model.")
        
        if not gemini_result['is_mri']:
            logger.info("Gemini classified image as NOT a Brain MRI.")
            not_mri_preds = np.zeros(len(REPORTING_CLASS_NAMES))
            try:
                not_mri_index = REPORTING_CLASS_NAMES.index('not_mri')
                not_mri_preds[not_mri_index] = 1.0
            except ValueError:
                logger.error("'not_mri' not found in REPORTING_CLASS_NAMES.")
                return jsonify({'error': 'Configuration error'}), 500
            
            classes = format_classification_results(not_mri_preds, REPORTING_CLASS_NAMES)
            return jsonify({
                'class': 'Likely Not a Brain MRI Scan',
                'confidence': 1.0,
                'classes': classes,
                'gemini': gemini_result
            })
        
        # Proceed with classification
        logger.info("Proceeding with tumor classification.")
        processed_image = preprocess_image(filepath)
        predictions = model.predict(processed_image, verbose=0)[0]
        
        # Prepare predictions for reporting
        full_predictions_for_reporting = np.append(predictions, 1e-6)
        
        # Determine predicted class
        model_predicted_index = np.argmax(predictions)
        predicted_class_name = MODEL_CLASS_NAMES[model_predicted_index]
        confidence_in_model_class = float(predictions[model_predicted_index])
        
        # Format results
        classes = format_classification_results(full_predictions_for_reporting, REPORTING_CLASS_NAMES)
        
        result = {
            'class': predicted_class_name.replace('_', ' ').capitalize(),
            'confidence': confidence_in_model_class,
            'classes': classes,
            'gemini': gemini_result
        }
        
        return jsonify(result)
    
    except cv2.error as e:
        logger.error(f"OpenCV error processing image: {str(e)}")
        return jsonify({'error': 'Image processing error'}), 400
    except tf.errors.OpError as e:
        logger.error(f"TensorFlow error during prediction: {str(e)}")
        return jsonify({'error': 'Model prediction failed'}), 500
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        if filepath:
            cleanup_file(filepath)

@app.route('/random', methods=['GET'])
def random_prediction():
    """Get a random prediction from the dataset."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    try:
        random_image_path = fetch_random_image_path()
        logger.info(f"Fetching random image: {random_image_path}")
        
        processed_image = preprocess_image(random_image_path)
        predictions = model.predict(processed_image, verbose=0)[0]
        full_predictions_for_reporting = np.append(predictions, 1e-6)
        
        # Determine predicted class
        model_predicted_index = np.argmax(predictions)
        predicted_class_name = MODEL_CLASS_NAMES[model_predicted_index]
        confidence_in_model_class = float(predictions[model_predicted_index])
        
        # Format results
        classes = format_classification_results(full_predictions_for_reporting, REPORTING_CLASS_NAMES)
        
        # Encode image
        base64_image = encode_image_to_base64(random_image_path)
        if not base64_image:
            return jsonify({'error': 'Failed to encode random image'}), 500
        
        return jsonify({
            'class': predicted_class_name.replace('_', ' ').capitalize(),
            'confidence': confidence_in_model_class,
            'classes': classes,
            'image': base64_image,
            'gemini': {'used': False, 'raw': None}
        }), 200
    
    except FileNotFoundError as e:
        logger.error(f"Dataset or random image not found: {str(e)}")
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"Unexpected error during random prediction: {str(e)}", exc_info=True)
        return jsonify({'error': 'Error processing random image'}), 500

@app.route('/heatmap', methods=['POST'])
def get_heatmap():
    """Generate Grad-CAM heatmap for uploaded image."""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    if 'file' not in request.files:
        logger.warning("No file provided in request for heatmap.")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    filepath = None
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"Saved uploaded file for heatmap: {filepath}")
        
        if not is_valid_image(filepath):
            logger.warning(f"Uploaded file for heatmap is not a valid image: {filepath}")
            return jsonify({'error': 'Uploaded file is not a valid image'}), 400
        
        # Gemini validation
        gemini_result = {'used': False, 'is_mri': True, 'raw': None}
        try:
            with open(filepath, "rb") as f:
                image_bytes = f.read()
            gemini_result = check_if_mri_with_gemini(image_bytes)
        except Exception as e:
            logger.error(f"Error during Gemini check for heatmap: {str(e)}. Proceeding.")
        
        if not gemini_result['is_mri']:
            logger.warning("Gemini classified image as NOT a Brain MRI. Cannot generate heatmap.")
            return jsonify({'error': 'Cannot generate heatmap for non-MRI images'}), 400
        
        # Generate heatmap
        logger.info("Proceeding with heatmap generation.")
        processed_image = preprocess_image(filepath)
        
        # Predict to get class index
        predictions = model.predict(processed_image, verbose=0)[0]
        class_index_for_heatmap = np.argmax(predictions)
        
        # Generate heatmap
        heatmap = generate_gradcam(processed_image, class_index_for_heatmap)
        
        # Encode heatmap
        _, buffer = cv2.imencode('.png', heatmap)
        encoded_heatmap = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'heatmap': encoded_heatmap}), 200
    
    except cv2.error as e:
        logger.error(f"OpenCV error processing image for heatmap: {str(e)}")
        return jsonify({'error': 'Image processing error'}), 400
    except tf.errors.OpError as e:
        logger.error(f"TensorFlow error during heatmap generation: {str(e)}")
        return jsonify({'error': 'Model prediction failed'}), 500
    except RuntimeError as e:
        logger.error(f"Runtime error during heatmap generation: {str(e)}")
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        logger.error(f"Unexpected error during heatmap generation: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500
    finally:
        if filepath:
            cleanup_file(filepath)

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    stats = {
        'model_info': {
            'classes': MODEL_CLASS_NAMES,
            'input_shape': [None, 224, 224, 3] if model else None,
            'loaded': model is not None
        },
        'gemini_available': gemini_vision_model is not None,
        'uptime': round(time.time() - app_start_time, 2)
    }
    return jsonify(stats)

# Initialize on startup
configure_gemini()
load_classification_model()
initialize_grad_model()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug, host='0.0.0.0', port=port)
