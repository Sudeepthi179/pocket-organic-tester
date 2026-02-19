"""
Pocket Organic Tester API
Flask application for spectral analysis and organic food detection.
"""

from flask import Flask, jsonify
from flask_cors import CORS
import os
import sys

# Import configuration
from config import config

# Import blueprints
from routes.scan_routes import scan_bp


def create_app(config_name=None):
    """
    Application factory function.
    
    Args:
        config_name: Configuration environment ('development', 'production', 'testing')
                    Defaults to FLASK_ENV environment variable or 'development'
    
    Returns:
        Configured Flask application instance
    """
    # Initialize Flask app
    app = Flask(__name__)
    
    # Load configuration
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app.config.from_object(config[config_name])
    
    # Enable CORS
    CORS(app, resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Register blueprints
    app.register_blueprint(scan_bp)
    
    # Root endpoint - health check
    @app.route('/', methods=['GET'])
    def index():
        """
        Root endpoint that serves as a basic health check.
        
        Returns:
            JSON response with API information
        """
        return jsonify({
            'message': 'Pocket Organic Tester API is running',
            'status': 'online',
            'version': '1.0.0',
            'endpoints': {
                'root': '/',
                'health': '/api/health',
                'info': '/api/info',
                'scan': '/api/scan (POST)'
            }
        }), 200
    
    # Global error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors globally."""
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource does not exist',
            'status': 404
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors globally."""
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred',
            'status': 500
        }), 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        """Handle uncaught exceptions."""
        app.logger.error(f'Unhandled exception: {str(error)}')
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred',
            'status': 500
        }), 500
    
    # Log application startup information
    with app.app_context():
        app.logger.info(f'Application started in {config_name} mode')
        app.logger.info(f'Model path: {app.config.get("MODEL_PATH")}')
    
    return app


# Create application instance
app = create_app()


if __name__ == '__main__':
    # Get configuration from environment
    env = os.environ.get('FLASK_ENV', 'development')
    debug_mode = env == 'development'
    
    # Port configuration
    port = int(os.environ.get('PORT', 5000))
    
    print("="*60)
    print("Pocket Organic Tester API")
    print("="*60)
    print(f"Environment: {env}")
    print(f"Debug Mode: {debug_mode}")
    print(f"Port: {port}")
    print(f"Model Path: {app.config.get('MODEL_PATH')}")
    print("="*60)
    print("\nAPI Endpoints:")
    print(f"  Root:   http://localhost:{port}/")
    print(f"  Health: http://localhost:{port}/api/health")
    print(f"  Info:   http://localhost:{port}/api/info")
    print(f"  Scan:   http://localhost:{port}/api/scan (POST)")
    print("="*60)
    print("\nStarting server...\n")
    
    # Run the application
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        use_reloader=debug_mode
    )
