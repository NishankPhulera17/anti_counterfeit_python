"""
Training Script for Authenticity Classifier

This script trains a classifier to distinguish between:
- "real": Authentic printed QR codes
- "duplicate": Photocopies, screenshots, or digital reproductions

Uses your 15 extracted metrics as input features.
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from services.authenticity_classifier import AuthenticityClassifier

# Default paths
DEFAULT_MODEL_DIR = "models"
DEFAULT_TRAINING_DATA = "training_data/qr_metrics_labeled.csv"


def load_training_data(filepath: str) -> list:
    """
    Load training data from CSV file.
    
    Expected CSV columns:
    - Sharpness, Contrast, HistogramPeak, EdgeDensity, EdgeStrength
    - NoiseLevel, HighFreqEnergy, ColorDiversity, UniqueColors
    - Saturation, TextureUniformity, CompressionArtifacts
    - HistogramEntropy, DynamicRange, Brightness
    - LightingCondition (bright/normal/dim/low)
    - Label (real/duplicate)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Map labels: 'fake' and 'photocopy' -> 'duplicate' for binary classification
    if 'Label' in df.columns:
        df['Label'] = df['Label'].replace({'fake': 'duplicate', 'photocopy': 'duplicate'})
        print(f"[INFO] Label mapping: 'fake' and 'photocopy' -> 'duplicate'")
        print(f"[INFO] Label distribution: {df['Label'].value_counts().to_dict()}")
    
    # Convert DataFrame to list of dictionaries
    training_data = []
    for _, row in df.iterrows():
        sample = row.to_dict()
        training_data.append(sample)
    
    print(f"[INFO] Loaded {len(training_data)} training samples from {filepath}")
    return training_data


def create_sample_training_data(output_path: str = "training_data/sample_data.csv"):
    """
    Create a sample training data CSV file with the expected format.
    This is just a template - replace with your actual labeled data.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample data structure
    sample_data = {
        'Sharpness': [107.54, 45.23, 98.12, 52.34],
        'Contrast': [84.59, 35.21, 79.45, 41.67],
        'HistogramPeak': [0.096, 0.234, 0.087, 0.198],
        'EdgeDensity': [0.0230, 0.0123, 0.0215, 0.0145],
        'EdgeStrength': [23.57, 12.34, 22.15, 15.67],
        'NoiseLevel': [2.00, 8.45, 2.15, 7.89],
        'HighFreqEnergy': [40162258944.00, 12345678901.23, 38901234567.89, 14567890123.45],
        'ColorDiversity': [0.0003, 0.0001, 0.0003, 0.0001],
        'UniqueColors': [2519, 856, 2345, 912],
        'Saturation': [41.02, 28.45, 39.87, 31.23],
        'TextureUniformity': [0.0593, 0.1234, 0.0621, 0.1156],
        'CompressionArtifacts': [167.03, 345.67, 158.92, 312.45],
        'HistogramEntropy': [6.35, 5.12, 6.28, 5.34],
        'DynamicRange': [254.00, 180.00, 250.00, 195.00],
        'Brightness': [70.93, 45.23, 68.45, 52.34],
        'LightingCondition': ['bright', 'normal', 'bright', 'dim'],
        'Label': ['real', 'duplicate', 'real', 'duplicate']
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Sample training data template created at {output_path}")
    print("[INFO] Replace with your actual labeled data!")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Authenticity Classifier')
    parser.add_argument('--data', type=str, default=DEFAULT_TRAINING_DATA,
                       help='Path to training data CSV file')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'xgboost'],
                       help='Model type to train')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for trained model (default: models/authenticity_classifier_{type}.pkl)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create sample training data template and exit')
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample:
        create_sample_training_data()
        return
    
    # Check if training data exists
    if not os.path.exists(args.data):
        print(f"[ERROR] Training data file not found: {args.data}")
        print(f"[INFO] Use --create-sample to create a template")
        sys.exit(1)
    
    # Load training data
    try:
        training_data = load_training_data(args.data)
    except Exception as e:
        print(f"[ERROR] Failed to load training data: {str(e)}")
        sys.exit(1)
    
    # Validate data
    required_metrics = [
        'Sharpness', 'Contrast', 'HistogramPeak', 'EdgeDensity',
        'EdgeStrength', 'NoiseLevel', 'HighFreqEnergy', 
        'ColorDiversity', 'UniqueColors', 'Saturation',
        'TextureUniformity', 'CompressionArtifacts',
        'HistogramEntropy', 'DynamicRange', 'Brightness',
        'LightingCondition', 'Label'
    ]
    
    # Check if all required columns are present
    sample = training_data[0]
    missing = [m for m in required_metrics if m not in sample]
    if missing:
        print(f"[ERROR] Missing required metrics: {missing}")
        sys.exit(1)
    
    # Initialize classifier
    print(f"\n[INFO] Initializing {args.model_type} classifier...")
    classifier = AuthenticityClassifier(model_type=args.model_type)
    
    # Train
    print(f"[INFO] Training classifier...")
    try:
        results = classifier.train(training_data, test_size=args.test_size, verbose=True)
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save model
    if args.output:
        output_path = args.output
    else:
        os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
        output_path = os.path.join(DEFAULT_MODEL_DIR, f"authenticity_classifier_{args.model_type}.pkl")
    
    classifier.save(output_path)
    
    print(f"\n[INFO] Training complete!")
    print(f"[INFO] Model saved to: {output_path}")
    print(f"[INFO] Test accuracy: {results['accuracy']:.4f}")
    if 'roc_auc' in results:
        print(f"[INFO] ROC AUC: {results['roc_auc']:.4f}")


if __name__ == '__main__':
    main()

