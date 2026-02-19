"""
Machine Learning Model Training Script
Loads spectral data from CSV and trains models for fruit classification
and organic/non-organic detection.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os


def load_data(csv_path='../data/synthetic_data.csv'):
    """
    Load spectral dataset from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing the spectral dataset
        
    Returns:
        DataFrame with spectral features (F1-F8), fruit labels, and organic labels
    """
    # Resolve path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, csv_path)
    
    # Check if file exists
    if not os.path.exists(full_path):
        raise FileNotFoundError(
            f"Dataset file not found: {full_path}\n"
            f"Please run 'python data/generate_dataset.py' to create the dataset first."
        )
    
    # Load CSV file
    df = pd.read_csv(full_path)
    
    print(f"Loaded dataset from: {full_path}")
    print(f"Total samples: {len(df)}")
    print(f"\nFruit distribution:")
    print(df['Fruit'].value_counts())
    print(f"\nOrganic distribution:")
    print(df['Organic'].value_counts())
    print(f"\nOrganic distribution by fruit:")
    print(df.groupby(['Fruit', 'Organic']).size())
    
    # Verify expected columns exist
    feature_columns = [f'F{i+1}' for i in range(8)]
    required_columns = feature_columns + ['Fruit', 'Organic']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    
    print(f"\nDataset validation: ✓ All required columns present")
    
    return df


def train_fruit_model(df):
    """
    Train a RandomForestClassifier to classify fruit types.
    
    Args:
        df: DataFrame with spectral features and labels
        
    Returns:
        Tuple of (trained model, label encoder, accuracy)
    """
    print("\n" + "="*60)
    print("Training Fruit Classification Model")
    print("="*60)
    
    # Prepare features and labels
    feature_columns = [f'F{i+1}' for i in range(8)]
    X = df[feature_columns].values
    y = df['Fruit'].values
    
    # Encode fruit labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nFruit Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_
    ))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model, label_encoder, accuracy


def train_organic_models(df):
    """
    Train separate RandomForestClassifier models for organic classification
    for each fruit type.
    
    Args:
        df: DataFrame with spectral features and labels
        
    Returns:
        Dictionary of trained models for each fruit
    """
    print("\n" + "="*60)
    print("Training Organic Classification Models")
    print("="*60)
    
    feature_columns = [f'F{i+1}' for i in range(8)]
    organic_models = {}
    
    for fruit in df['Fruit'].unique():
        print(f"\n--- Training Organic Model for {fruit} ---")
        
        # Filter data for this fruit
        fruit_df = df[df['Fruit'] == fruit].copy()
        
        X = fruit_df[feature_columns].values
        y = (fruit_df['Organic'] == 'Organic').astype(int).values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"{fruit} Organic Model Accuracy: {accuracy:.4f}")
        print(classification_report(
            y_test, y_pred,
            target_names=['Non-Organic', 'Organic']
        ))
        
        organic_models[fruit] = model
    
    return organic_models


def save_models(fruit_model, label_encoder, organic_models, save_dir='models'):
    """
    Save trained models to disk using joblib.
    
    Args:
        fruit_model: Trained fruit classification model
        label_encoder: Label encoder for fruit labels
        organic_models: Dictionary of organic classification models
        save_dir: Directory to save models
    """
    print("\n" + "="*60)
    print("Saving Models")
    print("="*60)
    
    # Create models directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save models
    fruit_model_path = os.path.join(save_dir, 'fruit_model.pkl')
    joblib.dump(fruit_model, fruit_model_path)
    print(f"Saved fruit model to: {fruit_model_path}")
    
    label_encoder_path = os.path.join(save_dir, 'label_encoder.pkl')
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Saved label encoder to: {label_encoder_path}")
    
    organic_models_path = os.path.join(save_dir, 'organic_models.pkl')
    joblib.dump(organic_models, organic_models_path)
    print(f"Saved organic models to: {organic_models_path}")
    
    print("\nAll models saved successfully!")


def main():
    """
    Main function to orchestrate data loading and model training.
    """
    print("="*60)
    print("Spectral Dataset Loading and Model Training")
    print("="*60)
    
    # Load spectral data from CSV
    df = load_data()
    
    # Train fruit classification model
    fruit_model, label_encoder, fruit_accuracy = train_fruit_model(df)
    
    # Train organic classification models
    organic_models = train_organic_models(df)
    
    # Save all models
    save_models(fruit_model, label_encoder, organic_models)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nFruit Classification Accuracy: {fruit_accuracy:.4f}")
    print(f"Number of Organic Models: {len(organic_models)}")
    print(f"Fruits: {list(organic_models.keys())}")


if __name__ == '__main__':
    main()
