"""
Unit tests for utility functions.
"""
import os
import io
import pytest
import numpy as np
from PIL import Image
import tempfile

# Set test environment
os.environ['FLASK_ENV'] = 'testing'

from server1 import (
    allowed_file,
    is_valid_image,
    preprocess_image,
    format_classification_results,
    encode_image_to_base64,
    cleanup_file
)


class TestFileValidation:
    """Tests for file validation functions."""
    
    def test_allowed_file_with_valid_extensions(self):
        """Test allowed_file with valid extensions."""
        assert allowed_file('test.png') is True
        assert allowed_file('test.jpg') is True
        assert allowed_file('test.jpeg') is True
        assert allowed_file('test.bmp') is True
    
    def test_allowed_file_with_invalid_extensions(self):
        """Test allowed_file with invalid extensions."""
        assert allowed_file('test.txt') is False
        assert allowed_file('test.pdf') is False
        assert allowed_file('test.exe') is False
        assert allowed_file('test') is False
    
    def test_allowed_file_case_insensitive(self):
        """Test that allowed_file is case insensitive."""
        assert allowed_file('test.PNG') is True
        assert allowed_file('test.JPG') is True
    
    def test_is_valid_image_with_valid_image(self):
        """Test is_valid_image with valid image."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (100, 100), color='red')
            img.save(f.name)
            assert is_valid_image(f.name) is True
            os.unlink(f.name)
    
    def test_is_valid_image_with_invalid_file(self):
        """Test is_valid_image with invalid file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(b'not an image')
            f.flush()
            assert is_valid_image(f.name) is False
            os.unlink(f.name)


class TestImageProcessing:
    """Tests for image processing functions."""
    
    def test_preprocess_image_returns_correct_shape(self):
        """Test that preprocess_image returns correct shape."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (512, 512), color='blue')
            img.save(f.name)
            
            processed = preprocess_image(f.name)
            assert processed.shape == (1, 224, 224, 3)
            assert processed.dtype == np.float64 or processed.dtype == np.float32
            assert np.max(processed) <= 1.0
            assert np.min(processed) >= 0.0
            
            os.unlink(f.name)
    
    def test_preprocess_image_with_invalid_path_raises_error(self):
        """Test that preprocess_image raises error with invalid path."""
        with pytest.raises(ValueError):
            preprocess_image('/nonexistent/path/image.png')


class TestResultFormatting:
    """Tests for result formatting functions."""
    
    def test_format_classification_results(self):
        """Test format_classification_results."""
        predictions = np.array([0.7, 0.2, 0.05, 0.05])
        class_names = ['class1', 'class2', 'class3', 'class4']
        
        results = format_classification_results(predictions, class_names)
        
        assert len(results) == 4
        assert results[0]['label'] == 'Class1'
        assert results[0]['percent'] == 70.0
        assert results[1]['percent'] == 20.0
        
        # Check that results are sorted by percent descending
        assert results[0]['percent'] >= results[1]['percent']
        assert results[1]['percent'] >= results[2]['percent']
    
    def test_format_classification_results_with_mismatched_lengths(self):
        """Test format_classification_results with mismatched lengths."""
        predictions = np.array([0.7, 0.3])
        class_names = ['class1', 'class2', 'class3', 'class4']
        
        # Should handle gracefully
        results = format_classification_results(predictions, class_names)
        assert len(results) == 2


class TestBase64Encoding:
    """Tests for base64 encoding."""
    
    def test_encode_image_to_base64(self):
        """Test encode_image_to_base64."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            img = Image.new('RGB', (50, 50), color='green')
            img.save(f.name)
            
            encoded = encode_image_to_base64(f.name)
            assert encoded is not None
            assert isinstance(encoded, str)
            assert len(encoded) > 0
            
            os.unlink(f.name)
    
    def test_encode_image_to_base64_with_invalid_path(self):
        """Test encode_image_to_base64 with invalid path."""
        encoded = encode_image_to_base64('/nonexistent/path/image.png')
        assert encoded is None


class TestFileCleanup:
    """Tests for file cleanup."""
    
    def test_cleanup_file_removes_existing_file(self):
        """Test that cleanup_file removes existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filepath = f.name
        
        assert os.path.exists(filepath)
        cleanup_file(filepath)
        assert not os.path.exists(filepath)
    
    def test_cleanup_file_with_nonexistent_file(self):
        """Test that cleanup_file handles nonexistent file gracefully."""
        # Should not raise error
        cleanup_file('/nonexistent/path/file.txt')
    
    def test_cleanup_file_with_none(self):
        """Test that cleanup_file handles None gracefully."""
        # Should not raise error
        cleanup_file(None)
