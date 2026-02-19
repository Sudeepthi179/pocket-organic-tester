import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent.absolute()


class Config:
    """Base configuration class with common settings."""
    
    # Secret key for session management and CSRF protection
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Model storage path
    MODEL_PATH = os.path.join(BASE_DIR, 'models')
    
    # Flask settings
    JSON_SORT_KEYS = False
    JSONIFY_PRETTYPRINT_REGULAR = True
    
    # CORS settings
    CORS_HEADERS = 'Content-Type'
    
    # Max upload size (e.g., for image uploads) - 16MB
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024


class DevelopmentConfig(Config):
    """Development environment configuration."""
    
    DEBUG = True
    TESTING = False
    
    # Development-specific settings
    ENV = 'development'


class ProductionConfig(Config):
    """Production environment configuration."""
    
    DEBUG = False
    TESTING = False
    
    # Production-specific settings
    ENV = 'production'
    
    # In production, SECRET_KEY must be set via environment variable
    @property
    def SECRET_KEY(self):
        secret_key = os.environ.get('SECRET_KEY')
        if not secret_key:
            raise ValueError("SECRET_KEY environment variable must be set in production")
        return secret_key


class TestingConfig(Config):
    """Testing environment configuration."""
    
    DEBUG = True
    TESTING = True
    
    # Testing-specific settings
    ENV = 'testing'


# Configuration dictionary for easy access
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
