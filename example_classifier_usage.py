"""
Example: Using the Authenticity Classifier

This shows how to use the trained classifier in your verification flow.
"""
from services.authenticity_classifier import AuthenticityClassifier, get_classifier
import os

# Example 1: Load and use a trained model
def example_1_load_and_predict():
    """Load trained model and make predictions"""
    
    # Path to your trained model
    model_path = "models/authenticity_classifier_xgboost.pkl"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}")
        print("[INFO] Train a model first using: python train_authenticity_classifier.py")
        return
    
    # Load classifier
    classifier = AuthenticityClassifier(model_path=model_path)
    
    # Example metrics from a scanned image
    # (Replace with your actual extracted metrics)
    test_metrics = {
        'Sharpness': 107.54,
        'Contrast': 84.59,
        'HistogramPeak': 0.096,
        'EdgeDensity': 0.0230,
        'EdgeStrength': 23.57,
        'NoiseLevel': 2.00,
        'HighFreqEnergy': 40162258944.00,
        'ColorDiversity': 0.0003,
        'UniqueColors': 2519,
        'Saturation': 41.02,
        'TextureUniformity': 0.0593,
        'CompressionArtifacts': 167.03,
        'HistogramEntropy': 6.35,
        'DynamicRange': 254.00,
        'Brightness': 70.93,
        'LightingCondition': 'bright'
    }
    
    # Predict
    result = classifier.predict_single(test_metrics)
    
    print("Prediction Results:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Confidence: {result['confidence']:.2f}%")
    print(f"  Is Authentic: {result['is_authentic']}")
    print(f"  Probabilities: {result['probabilities']}")


# Example 2: Integration into your verification endpoint
def example_2_integration_in_app():
    """
    How to integrate into your app.py verify_cdp endpoint
    
    Add this to your verification flow:
    """
    
    integration_code = '''
# In app.py, in the verify_cdp() function, after extracting metrics:

from services.authenticity_classifier import get_classifier

# ... your existing code ...

# After extracting CDP and getting metrics
# (You'll need to extract your 15 metrics from the scanned image)

# Load classifier (lazy loading, cached after first call)
classifier = get_classifier(
    model_type='xgboost',
    model_path='models/authenticity_classifier_xgboost.pkl'
)

# Prepare metrics dictionary
# (You'll need to extract these from your scanned image)
metrics_dict = {
    'Sharpness': extract_sharpness(scanned_cdp),
    'Contrast': extract_contrast(scanned_cdp),
    # ... extract all 15 metrics ...
    'LightingCondition': lighting_assessment['lighting_info']['status']
}

# Get ML prediction
ml_result = classifier.predict_single(metrics_dict)

# Combine with existing verification
# Option 1: Use ML as additional check
if not ml_result['is_authentic']:
    # ML says it's a duplicate
    print("[WARNING] ML classifier detected duplicate")
    # You can fail verification or lower confidence

# Option 2: Combine scores
# ml_confidence = ml_result['confidence'] / 100.0  # 0-1
# final_score = 0.7 * similarity_score + 0.3 * ml_confidence
'''
    
    print("Integration Example:")
    print("="*60)
    print(integration_code)


# Example 3: Batch prediction
def example_3_batch_prediction():
    """Predict on multiple images"""
    
    model_path = "models/authenticity_classifier_xgboost.pkl"
    classifier = AuthenticityClassifier(model_path=model_path)
    
    # List of metrics from multiple images
    batch_metrics = [
        {
            'Sharpness': 107.54,
            'Contrast': 84.59,
            # ... all metrics ...
            'LightingCondition': 'bright'
        },
        {
            'Sharpness': 45.23,
            'Contrast': 35.21,
            # ... all metrics ...
            'LightingCondition': 'normal'
        },
        # ... more samples ...
    ]
    
    results = []
    for metrics in batch_metrics:
        result = classifier.predict_single(metrics)
        results.append(result)
    
    # Analyze results
    authentic_count = sum(1 for r in results if r['is_authentic'])
    print(f"Authentic: {authentic_count}/{len(results)}")
    print(f"Average confidence: {sum(r['confidence'] for r in results) / len(results):.2f}%")


if __name__ == '__main__':
    print("Authenticity Classifier Usage Examples")
    print("="*60)
    print("\n1. Load and Predict:")
    example_1_load_and_predict()
    
    print("\n\n2. Integration in app.py:")
    example_2_integration_in_app()
    
    print("\n\n3. Batch Prediction:")
    example_3_batch_prediction()

