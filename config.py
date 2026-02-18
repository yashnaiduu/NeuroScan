"""
Configuration management for NeuroScan application.
Provides environment-specific settings for development, staging, and production.
"""
import os
from typing import Dict, Any


class Config:
    """Base configuration class."""
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Application
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'Uploads')
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    
    # Model
    MODEL_PATH = os.getenv('MODEL_PATH', 'mobilenet_brain_tumor_classifier.h5')
    MODEL_URL = os.getenv('MODEL_URL', '')
    
    # Dataset
    DATASET_PATH = os.getenv('DATASET_PATH', './Dataset')
    
    # Cache
    CACHE_FOLDER = os.getenv('CACHE_FOLDER', './cache')
    CACHE_DURATION = int(os.getenv('CACHE_DURATION', 3600))
    
    # Rate Limiting
    RATELIMIT_ENABLED = os.getenv('RATELIMIT_ENABLED', 'true').lower() == 'true'
    RATELIMIT_DEFAULT = os.getenv('RATE_LIMIT', '10 per minute')
    RATELIMIT_STORAGE_URL = os.getenv('RATELIMIT_STORAGE_URL', 'memory://')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @staticmethod
    def init_app(app):
        """Initialize application with this config."""
        pass


class DevelopmentConfig(Config):
    """Development environment configuration."""
    DEBUG = True
    TESTING = False
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production environment configuration."""
    DEBUG = False
    TESTING = False
    
    @staticmethod
    def init_app(app):
        """Production-specific initialization."""
        Config.init_app(app)
        
        # Ensure critical environment variables are set
        if not os.getenv('SECRET_KEY'):
            raise ValueError("SECRET_KEY must be set in production")


class TestingConfig(Config):
    """Testing environment configuration."""
    DEBUG = True
    TESTING = True
    RATELIMIT_ENABLED = False


# Configuration dictionary
config: Dict[str, Any] = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env: str = None) -> Config:
    """Get configuration for specified environment."""
    if env is None:
        env = os.getenv('FLASK_ENV', 'development')
    return config.get(env, config['default'])
