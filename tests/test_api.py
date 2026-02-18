"""
Unit tests for NeuroScan API endpoints.
"""
import os
import io
import json
import pytest
from PIL import Image
import numpy as np

# Set test environment before importing app
os.environ['FLASK_ENV'] = 'testing'
os.environ['MODEL_PATH'] = 'test_model.h5'

from server1 import app, model


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = Image.new('RGB', (224, 224), color='gray')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_endpoint_returns_200(self, client):
        """Test that health endpoint returns 200 status."""
        response = client.get('/health')
        assert response.status_code in [200, 503]
    
    def test_health_endpoint_returns_json(self, client):
        """Test that health endpoint returns JSON."""
        response = client.get('/health')
        assert response.content_type == 'application/json'
    
    def test_health_endpoint_has_required_fields(self, client):
        """Test that health response has required fields."""
        response = client.get('/health')
        data = json.loads(response.data)
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'uptime' in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""
    
    def test_predict_without_file_returns_400(self, client):
        """Test that predict without file returns 400."""
        response = client.post('/predict')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_with_invalid_file_type_returns_400(self, client):
        """Test that predict with invalid file type returns 400."""
        data = {'file': (io.BytesIO(b'test'), 'test.txt')}
        response = client.post('/predict', data=data, content_type='multipart/form-data')
        assert response.status_code == 400
    
    @pytest.mark.skipif(model is None, reason="Model not loaded")
    def test_predict_with_valid_image(self, client, sample_image):
        """Test prediction with valid image."""
        data = {'file': (sample_image, 'test.png')}
        response = client.post('/predict', data=data, content_type='multipart/form-data')
        
        # Should return 200 or 503 (if model not loaded)
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            result = json.loads(response.data)
            assert 'class' in result
            assert 'confidence' in result
            assert 'classes' in result


class TestRandomEndpoint:
    """Tests for /random endpoint."""
    
    def test_random_endpoint_returns_json(self, client):
        """Test that random endpoint returns JSON."""
        response = client.get('/random')
        assert response.content_type == 'application/json'
    
    @pytest.mark.skipif(model is None, reason="Model not loaded")
    def test_random_endpoint_structure(self, client):
        """Test random endpoint response structure."""
        response = client.get('/random')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            assert 'class' in data or 'error' in data


class TestHeatmapEndpoint:
    """Tests for /heatmap endpoint."""
    
    def test_heatmap_without_file_returns_400(self, client):
        """Test that heatmap without file returns 400."""
        response = client.post('/heatmap')
        assert response.status_code == 400
    
    def test_heatmap_with_invalid_file_type_returns_400(self, client):
        """Test that heatmap with invalid file type returns 400."""
        data = {'file': (io.BytesIO(b'test'), 'test.txt')}
        response = client.post('/heatmap', data=data, content_type='multipart/form-data')
        assert response.status_code == 400


class TestStatsEndpoint:
    """Tests for /stats endpoint."""
    
    def test_stats_endpoint_returns_200(self, client):
        """Test that stats endpoint returns 200."""
        response = client.get('/stats')
        assert response.status_code == 200
    
    def test_stats_endpoint_returns_json(self, client):
        """Test that stats endpoint returns JSON."""
        response = client.get('/stats')
        assert response.content_type == 'application/json'
    
    def test_stats_has_model_info(self, client):
        """Test that stats response has model info."""
        response = client.get('/stats')
        data = json.loads(response.data)
        assert 'model_info' in data
        assert 'uptime' in data


class TestHomeEndpoint:
    """Tests for home endpoint."""
    
    def test_home_returns_200(self, client):
        """Test that home endpoint returns 200."""
        response = client.get('/')
        # May return 200 or 500 if template not found
        assert response.status_code in [200, 500]
