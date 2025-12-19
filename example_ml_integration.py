"""
Example: Integrating ML Features into Existing Verification Flow

This example shows how to add ML-enhanced features to your existing
anti-counterfeit verification system. It demonstrates:

1. Adding ML features alongside rule-based features
2. Combining scores in an ensemble
3. Updating the verification endpoint

To use this:
1. Install PyTorch: pip install torch torchvision
2. Import and use in your app.py verify_cdp endpoint
"""

from services.feature_extraction import extract_all_features, compare_features
from services.ml_feature_extraction import extract_ml_features, compare_ml_features
import numpy as np


def verify_with_ml_enhancement(reference_cdp, scanned_cdp, 
                               ml_weight=0.3, rule_weight=0.7):
    """
    Enhanced verification combining rule-based and ML features.
    
    Args:
        reference_cdp: Reference CDP image (BGR format)
        scanned_cdp: Scanned CDP image (BGR format)
        ml_weight: Weight for ML similarity score (default: 0.3)
        rule_weight: Weight for rule-based score (default: 0.7)
    
    Returns:
        dict with:
            - final_score: Combined similarity score
            - rule_score: Rule-based score
            - ml_score: ML-based score
            - ml_available: Whether ML features were available
    """
    # 1. Extract rule-based features (existing method)
    ref_rule_features = extract_all_features(reference_cdp)
    scan_rule_features = extract_all_features(scanned_cdp)
    rule_score = compare_features(ref_rule_features, scan_rule_features)
    
    # 2. Extract ML features (new)
    ref_ml_features = extract_ml_features(reference_cdp)
    scan_ml_features = extract_ml_features(scanned_cdp)
    
    ml_score = 0.0
    ml_available = False
    
    if ref_ml_features is not None and scan_ml_features is not None:
        ml_score = compare_ml_features(ref_ml_features, scan_ml_features)
        ml_available = True
    
    # 3. Combine scores (ensemble)
    if ml_available:
        # Use weighted combination
        final_score = (rule_weight * rule_score + ml_weight * ml_score)
    else:
        # Fallback to rule-based only
        final_score = rule_score
        print("[WARNING] ML features not available, using rule-based only")
    
    return {
        'final_score': float(final_score),
        'rule_score': float(rule_score),
        'ml_score': float(ml_score),
        'ml_available': ml_available,
        'threshold': 0.65  # Your existing threshold
    }


# Example usage in app.py verify_cdp endpoint:
"""
# In app.py, modify the verify_cdp function:

from services.ml_feature_extraction import extract_ml_features, compare_ml_features

# After extracting rule-based features (around line 308):
scanned_features = extract_all_features(scanned_cdp)
reference_features = cdp_record.get('features')

# Add ML feature extraction:
ref_ml_features = extract_ml_features(reference_cdp_image)  # You'll need to load reference image
scan_ml_features = extract_ml_features(scanned_cdp)

# Compare ML features:
ml_score = 0.0
if ref_ml_features is not None and scan_ml_features is not None:
    ml_score = compare_ml_features(ref_ml_features, scan_ml_features)

# Combine with existing rule-based score:
rule_score = compare_features(reference_features, scanned_features)

# Ensemble combination (adjust weights based on performance):
ml_weight = 0.3
rule_weight = 0.7
similarity_score = rule_weight * rule_score + ml_weight * ml_score

# Continue with existing threshold check...
"""


if __name__ == '__main__':
    """
    Example test script
    """
    import cv2
    import os
    
    # Example: Load test images
    # You would use your actual CDP images here
    print("ML Integration Example")
    print("=" * 50)
    
    # Check if ML is available
    try:
        from services.ml_feature_extraction import get_ml_extractor
        extractor = get_ml_extractor()
        if extractor.available:
            print("✅ ML features available")
            print(f"   Model: {extractor.model_name}")
            print(f"   Device: {extractor.device}")
        else:
            print("❌ ML features not available (PyTorch not installed?)")
    except Exception as e:
        print(f"❌ Error checking ML availability: {str(e)}")
    
    print("\nTo integrate:")
    print("1. Install PyTorch: pip install torch torchvision")
    print("2. Import verify_with_ml_enhancement in app.py")
    print("3. Replace existing verification logic with ensemble approach")
    print("4. Tune weights (ml_weight, rule_weight) based on performance")

