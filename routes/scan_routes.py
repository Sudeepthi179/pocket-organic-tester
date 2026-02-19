"""
Scan Routes Blueprint
Defines API endpoints for spectral scanning and prediction.
"""

from flask import Blueprint, request, jsonify
import sys
import os

# Add parent directory to path to import predict module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.predict import predict_spectrum


# Create Blueprint
scan_bp = Blueprint('scan', __name__, url_prefix='/api')


@scan_bp.route('/scan', methods=['POST'])
def scan():
    """
    Scan endpoint for spectral analysis.
    
    Accepts spectral data and returns fruit classification and organic status.
    
    Request Body:
        {
            "spectral_values": [float, float, float, float, float, float, float, float]
        }
    
    Returns:
        JSON response with prediction results or error message
        
    Status Codes:
        200: Success
        400: Bad Request (validation error)
        500: Internal Server Error
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate request body exists
        if data is None:
            return jsonify({
                'error': 'Invalid request',
                'message': 'Request body must be valid JSON'
            }), 400
        
        # Validate spectral_values field exists
        if 'spectral_values' not in data:
            return jsonify({
                'error': 'Missing field',
                'message': 'Request must include "spectral_values" field'
            }), 400
        
        spectral_values = data['spectral_values']
        
        # Validate spectral_values is a list or array
        if not isinstance(spectral_values, (list, tuple)):
            return jsonify({
                'error': 'Invalid data type',
                'message': 'spectral_values must be a list or array'
            }), 400
        
        # Validate length
        if len(spectral_values) != 8:
            return jsonify({
                'error': 'Invalid input size',
                'message': f'spectral_values must contain exactly 8 values, got {len(spectral_values)}'
            }), 400
        
        # Validate all values are numeric
        for idx, value in enumerate(spectral_values):
            if not isinstance(value, (int, float)):
                return jsonify({
                    'error': 'Invalid data type',
                    'message': f'All spectral values must be numeric. Value at index {idx} is {type(value).__name__}'
                }), 400
            
            # Check for special float values
            if not (-float('inf') < value < float('inf')):
                return jsonify({
                    'error': 'Invalid value',
                    'message': f'spectral_values contains invalid numeric value at index {idx}'
                }), 400
        
        # Perform prediction
        try:
            result = predict_spectrum(spectral_values)
            
            # Return successful response
            return jsonify({
                'success': True,
                'data': result
            }), 200
            
        except ValueError as e:
            # Validation errors from predict_spectrum
            return jsonify({
                'error': 'Validation error',
                'message': str(e)
            }), 400
            
        except TypeError as e:
            # Type errors from predict_spectrum
            return jsonify({
                'error': 'Type error',
                'message': str(e)
            }), 400
            
        except FileNotFoundError as e:
            # Model files not found
            return jsonify({
                'error': 'Model not found',
                'message': 'Machine learning models are not available. Please train models first.'
            }), 500
            
        except Exception as e:
            # Unexpected errors during prediction
            return jsonify({
                'error': 'Prediction error',
                'message': f'An error occurred during prediction: {str(e)}'
            }), 500
    
    except Exception as e:
        # Catch-all for unexpected errors
        return jsonify({
            'error': 'Internal server error',
            'message': f'An unexpected error occurred: {str(e)}'
        }), 500


@scan_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint to verify the API is running.
    
    Returns:
        JSON response with service status
    """
    try:
        # Try to load models to verify they're available
        from models.predict import load_models
        load_models()
        
        return jsonify({
            'status': 'healthy',
            'service': 'Organic Tester API',
            'models_loaded': True
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'service': 'Organic Tester API',
            'models_loaded': False,
            'error': str(e)
        }), 500


@scan_bp.route('/info', methods=['GET'])
def info():
    """
    Information endpoint that returns API details and expected input format.
    
    Returns:
        JSON response with API information
    """
    return jsonify({
        'api': 'Organic Tester API',
        'version': '1.0.0',
        'endpoints': {
            '/api/scan': {
                'method': 'POST',
                'description': 'Analyze spectral data to classify fruit and organic status',
                'input_format': {
                    'spectral_values': 'Array of 8 numeric values representing spectral channels F1-F8'
                },
                'example_request': {
                    'spectral_values': [0.45, 0.52, 0.58, 0.62, 0.55, 0.48, 0.42, 0.38]
                },
                'example_response': {
                    'success': True,
                    'data': {
                        'fruit': 'Apple',
                        'organic_status': 'Organic',
                        'fruit_confidence': 0.95,
                        'organic_confidence': 0.87
                    }
                }
            },
            '/api/health': {
                'method': 'GET',
                'description': 'Check API health and model availability'
            },
            '/api/info': {
                'method': 'GET',
                'description': 'Get API information and documentation'
            }
        },
        'supported_fruits': ['Apple', 'Banana', 'Tomato'],
        'spectral_channels': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8']
    }), 200


# Error handlers for the blueprint
@scan_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Not found',
        'message': 'The requested endpoint does not exist'
    }), 404


@scan_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'error': 'Method not allowed',
        'message': 'The HTTP method is not allowed for this endpoint'
    }), 405
