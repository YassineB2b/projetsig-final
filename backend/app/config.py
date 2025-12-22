"""
Application Configuration

Environment-based configuration for the LPR system.
"""

import os
from pathlib import Path


class Config:
    """Base configuration."""

    # Flask
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = False
    TESTING = False

    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL',
        'postgresql://lpr:lpr_password@localhost:5432/lpr_db'
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }

    # Redis
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')

    # Storage paths
    BASE_DIR = Path(__file__).parent.parent.parent
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', str(BASE_DIR / 'storage' / 'uploads'))
    DETECTION_FOLDER = os.environ.get('DETECTION_FOLDER', str(BASE_DIR / 'storage' / 'detections'))
    LOG_FOLDER = os.environ.get('LOG_FOLDER', str(BASE_DIR / 'storage' / 'logs'))

    # Upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

    # ML Settings
    USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'
    DETECTION_CONFIDENCE = float(os.environ.get('DETECTION_CONFIDENCE', '0.5'))
    DETECTOR_MODEL_PATH = os.environ.get('DETECTOR_MODEL_PATH', None)
    OCR_LANGUAGES = os.environ.get('OCR_LANGUAGES', 'en,ar').split(',')

    # Storage settings
    STORAGE_RETENTION_DAYS = int(os.environ.get('STORAGE_RETENTION_DAYS', '30'))


class DevelopmentConfig(Config):
    """Development configuration."""

    DEBUG = True
    SQLALCHEMY_ECHO = False  # Set to True to see SQL queries


class ProductionConfig(Config):
    """Production configuration."""

    DEBUG = False

    # Stricter settings for production
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 10,
        'max_overflow': 20,
    }


class TestingConfig(Config):
    """Testing configuration."""

    TESTING = True
    DEBUG = True

    # Use SQLite for tests
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

    # Disable GPU for faster tests
    USE_GPU = False


# Configuration dictionary
config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
