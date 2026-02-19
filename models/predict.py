"""
Prediction Module for Spectral Analysis
Loads trained models and performs fruit classification and organic detection.
"""

import numpy as np
import joblib
import os
from pathlib import Path


# Model paths
MODEL_DIR = Path(__file__).parent.absolute()
FRUIT_MODEL_PATH = os.path.join(MODEL_DIR, 'fruit_model.pkl')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
ORGANIC_MODELS_PATH = os.path.join(MODEL_DIR, 'organic_models.pkl')


# Global variables to store loaded models
_fruit_model = None
_label_encoder = None
_organic_models = None


def load_models():
    """
    Load trained models from disk.
    
    Returns:
        Tuple of (fruit_model, label_encoder, organic_models)
        
    Raises:
        FileNotFoundError: If model files are not found
        Exception: If models cannot be loaded
    """
    global _fruit_model, _label_encoder, _organic_models
    
    # Check if models are already loaded
    if _fruit_model is not None and _label_encoder is not None and _organic_models is not None:
        return _fruit_model, _label_encoder, _organic_models
    
    # Verify model files exist
    if not os.path.exists(FRUIT_MODEL_PATH):
        raise FileNotFoundError(f"Fruit model not found at {FRUIT_MODEL_PATH}")
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder not found at {LABEL_ENCODER_PATH}")
    if not os.path.exists(ORGANIC_MODELS_PATH):
        raise FileNotFoundError(f"Organic models not found at {ORGANIC_MODELS_PATH}")
    
    try:
        # Load models
        _fruit_model = joblib.load(FRUIT_MODEL_PATH)
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)
        _organic_models = joblib.load(ORGANIC_MODELS_PATH)
        
        print("Models loaded successfully!")
        print(f"Available fruits: {_label_encoder.classes_}")
        print(f"Organic models for: {list(_organic_models.keys())}")
        
        return _fruit_model, _label_encoder, _organic_models
        
    except Exception as e:
        raise Exception(f"Error loading models: {str(e)}")


def validate_spectral_input(spectral_values):
    """
    Validate spectral input data.
    
    Args:
        spectral_values: Input data to validate
        
    Returns:
        numpy array of validated spectral values
        
    Raises:
        ValueError: If input is invalid
        TypeError: If input contains non-numeric values
    """
    # Check if input is a list or array-like
    if not isinstance(spectral_values, (list, tuple, np.ndarray)):
        raise TypeError("spectral_values must be a list, tuple, or numpy array")
    
    # Check length
    if len(spectral_values) != 8:
        raise ValueError(f"spectral_values must contain exactly 8 values, got {len(spectral_values)}")
    
    # Convert to numpy array and validate numeric types
    try:
        spectral_array = np.array(spectral_values, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"All spectral values must be numeric: {str(e)}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(spectral_array)):
        raise ValueError("spectral_values contains NaN values")
    if np.any(np.isinf(spectral_array)):
        raise ValueError("spectral_values contains infinite values")
    
    # Optional: Check if values are in reasonable range [0, 1] for reflectance data
    if np.any(spectral_array < 0) or np.any(spectral_array > 1):
        print("Warning: Spectral values outside typical range [0, 1]. Results may be unreliable.")
    
    return spectral_array


def predict_spectrum(spectral_values: list) -> dict:
    """
    Predict fruit type and organic status from spectral data.
    
    Args:
        spectral_values: List of exactly 8 numerical spectral values (F1-F8)
        
    Returns:
        Dictionary containing:
            - 'fruit': Predicted fruit name (str)
            - 'organic_status': 'Organic' or 'Non-Organic' (str)
            - 'fruit_confidence': Confidence score for fruit prediction (float, 0-1)
            - 'organic_confidence': Confidence score for organic prediction (float, 0-1)
            
    Raises:
        ValueError: If input validation fails
        TypeError: If input contains non-numeric values
        FileNotFoundError: If models are not found
        Exception: If prediction fails
        
    Example:
        >>> spectral_data = [0.45, 0.52, 0.58, 0.62, 0.55, 0.48, 0.42, 0.38]
        >>> result = predict_spectrum(spectral_data)
        >>> print(result)
        {'fruit': 'Apple', 'organic_status': 'Organic', 
         'fruit_confidence': 0.95, 'organic_confidence': 0.87}
    """
    # Validate input
    spectral_array = validate_spectral_input(spectral_values)
    
    # Load models if not already loaded
    fruit_model, label_encoder, organic_models = load_models()
    
    # Reshape for prediction (models expect 2D array)
    spectral_input = spectral_array.reshape(1, -1)
    
    try:
        # Step 1: Predict fruit type
        fruit_prediction = fruit_model.predict(spectral_input)[0]
        fruit_probabilities = fruit_model.predict_proba(spectral_input)[0]
        
        # Decode fruit label
        fruit_name = label_encoder.inverse_transform([fruit_prediction])[0]
        
        # Get confidence for predicted fruit
        fruit_confidence = float(fruit_probabilities[fruit_prediction])
        
        # Step 2: Predict organic status using fruit-specific model
        if fruit_name not in organic_models:
            raise ValueError(f"No organic model found for fruit: {fruit_name}")
        
        organic_model = organic_models[fruit_name]
        organic_prediction = organic_model.predict(spectral_input)[0]
        organic_probabilities = organic_model.predict_proba(spectral_input)[0]
        
        # Convert binary prediction to readable string
        organic_status = 'Organic' if organic_prediction == 1 else 'Non-Organic'
        
        # Get confidence for organic prediction
        organic_confidence = float(organic_probabilities[organic_prediction])
        
        # Return results
        return {
            'fruit': fruit_name,
            'organic_status': organic_status,
            'fruit_confidence': round(fruit_confidence, 4),
            'organic_confidence': round(organic_confidence, 4)
        }
        
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")


def predict_batch(spectral_batch: list) -> list:
    """
    Predict fruit type and organic status for multiple spectral samples.
    
    Args:
        spectral_batch: List of spectral value lists, each containing 8 values
        
    Returns:
        List of prediction dictionaries
        
    Example:
        >>> batch = [
        ...     [0.45, 0.52, 0.58, 0.62, 0.55, 0.48, 0.42, 0.38],
        ...     [0.72, 0.78, 0.82, 0.85, 0.80, 0.75, 0.68, 0.62]
        ... ]
        >>> results = predict_batch(batch)
    """
    results = []
    
    for idx, spectral_values in enumerate(spectral_batch):
        try:
            prediction = predict_spectrum(spectral_values)
            prediction['sample_index'] = idx
            results.append(prediction)
        except Exception as e:
            results.append({
                'sample_index': idx,
                'error': str(e)
            })
    
    return results


# Example usage and testing
if __name__ == '__main__':
    print("="*60)
    print("Testing Prediction Module")
    print("="*60)
    
    # Test samples (similar to training data)
    test_samples = {
        'Apple (Organic)': [0.47, 0.55, 0.60, 0.64, 0.57, 0.50, 0.45, 0.41],
        'Apple (Non-Organic)': [0.45, 0.52, 0.58, 0.62, 0.55, 0.48, 0.42, 0.38],
        'Banana (Organic)': [0.74, 0.80, 0.85, 0.88, 0.83, 0.77, 0.70, 0.63],
        'Banana (Non-Organic)': [0.72, 0.78, 0.82, 0.85, 0.80, 0.75, 0.68, 0.62],
        'Tomato (Organic)': [0.71, 0.44, 0.37, 0.40, 0.48, 0.55, 0.51, 0.46],
        'Tomato (Non-Organic)': [0.68, 0.42, 0.35, 0.38, 0.45, 0.52, 0.48, 0.44]
    }
    
    print("\nRunning test predictions:\n")
    
    for label, spectral_data in test_samples.items():
        print(f"Testing: {label}")
        print(f"Spectral input: {spectral_data}")
        
        try:
            result = predict_spectrum(spectral_data)
            print(f"Result: {result}")
            print()
        except Exception as e:
            print(f"Error: {str(e)}")
            print()
    
    # Test error handling
    print("="*60)
    print("Testing Error Handling")
    print("="*60)
    
    # Test with wrong number of values
    print("\nTest 1: Wrong number of values (7 instead of 8)")
    try:
        predict_spectrum([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # Test with non-numeric values
    print("\nTest 2: Non-numeric values")
    try:
        predict_spectrum([0.5, 0.5, 'invalid', 0.5, 0.5, 0.5, 0.5, 0.5])
    except TypeError as e:
        print(f"Caught expected error: {e}")
    
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
