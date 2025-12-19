# Quick Start: ML Authenticity Classifier

This guide shows you how to quickly set up and use the ML classifier that uses your 15 metrics to detect authentic prints vs duplicates.

## Overview

**What it does**: Classifies scanned QR codes as "real" (authentic print) or "duplicate" (photocopy/screenshot) using your 15 extracted metrics.

**Approach**: Simple, fast, interpretable (Approach A from the document)
- ✅ Uses only your 15 metrics (no CNN needed)
- ✅ Fast inference (~1-5ms)
- ✅ No GPU required
- ✅ Interpretable (see which metrics matter)

---

## Step 1: Install Dependencies

```bash
pip install scikit-learn joblib pandas xgboost
```

Or update your existing requirements:
```bash
pip install -r requirements.txt
```

---

## Step 2: Prepare Training Data

You need a CSV file with your labeled data. Each row should have:
- All 15 metrics (columns)
- `LightingCondition` (bright/normal/dim/low)
- `Label` (real/duplicate)

### Create Sample Template

```bash
python train_authenticity_classifier.py --create-sample
```

This creates `training_data/sample_data.csv` - replace with your actual labeled data.

### CSV Format

```csv
Sharpness,Contrast,HistogramPeak,EdgeDensity,EdgeStrength,NoiseLevel,HighFreqEnergy,ColorDiversity,UniqueColors,Saturation,TextureUniformity,CompressionArtifacts,HistogramEntropy,DynamicRange,Brightness,LightingCondition,Label
107.54,84.59,0.096,0.0230,23.57,2.00,40162258944.00,0.0003,2519,41.02,0.0593,167.03,6.35,254.00,70.93,bright,real
45.23,35.21,0.234,0.0123,12.34,8.45,12345678901.23,0.0001,856,28.45,0.1234,345.67,5.12,180.00,45.23,normal,duplicate
```

**Minimum recommended**: 100+ samples (50+ per class)

---

## Step 3: Train the Model

```bash
python train_authenticity_classifier.py --data training_data/qr_metrics_labeled.csv --model-type random_forest
```

Options:
- `--data`: Path to your training CSV
- `--model-type`: `random_forest` (default) or `xgboost` (often better, but requires xgboost)
- `--output`: Custom output path (default: `models/authenticity_classifier_{type}.pkl`)
- `--test-size`: Fraction for testing (default: 0.2)

### Example Output

```
[INFO] Loaded 200 training samples
[INFO] Training on 160 samples, testing on 40 samples
[INFO] Training random_forest model...

==================================================
Training Results
==================================================

Accuracy: 0.9500

Classification Report:
              precision    recall  f1-score   support

   duplicate       0.96      0.94      0.95        18
        real       0.94      0.96      0.95        22

Top 10 Most Important Features:
  1. NoiseLevel              : 0.1523
  2. CompressionArtifacts    : 0.1345
  3. Sharpness               : 0.1234
  4. EdgeDensity             : 0.1123
  ...

[INFO] Model saved to models/authenticity_classifier_random_forest.pkl
```

---

## Step 4: Use in Your Code

### Basic Usage

```python
from services.authenticity_classifier import AuthenticityClassifier

# Load trained model
classifier = AuthenticityClassifier(
    model_path='models/authenticity_classifier_random_forest.pkl'
)

# Your extracted metrics (from your image analysis)
metrics = {
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
result = classifier.predict_single(metrics)

print(f"Prediction: {result['prediction']}")  # 'real' or 'duplicate'
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Is Authentic: {result['is_authentic']}")
```

### Integration in app.py

Add to your `verify_cdp` endpoint:

```python
from services.authenticity_classifier import get_classifier

# In verify_cdp() function, after extracting CDP:

# Load classifier (cached after first call)
classifier = get_classifier(
    model_type='random_forest',
    model_path='models/authenticity_classifier_random_forest.pkl'
)

# Extract your 15 metrics from scanned_cdp
# (You'll need to implement metric extraction functions)
metrics_dict = {
    'Sharpness': extract_sharpness(scanned_cdp),
    'Contrast': extract_contrast(scanned_cdp),
    # ... extract all 15 metrics ...
    'LightingCondition': lighting_assessment['lighting_info']['status']
}

# Get ML prediction
ml_result = classifier.predict_single(metrics_dict)

# Use in verification decision
if not ml_result['is_authentic']:
    # ML detected duplicate - you can fail or flag
    return jsonify({
        'status': 'failed',
        'message': 'ML classifier detected duplicate/photocopy',
        'ml_confidence': ml_result['confidence']
    })
```

---

## Step 5: Extract Your 15 Metrics

You'll need to implement functions to extract the 15 metrics from images. Here's a template:

```python
def extract_all_metrics(image):
    """
    Extract all 15 metrics from CDP image.
    Replace these with your actual extraction logic.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    metrics = {
        'Sharpness': calculate_sharpness(gray),
        'Contrast': calculate_contrast(gray),
        'HistogramPeak': calculate_histogram_peak(gray),
        'EdgeDensity': calculate_edge_density(gray),
        'EdgeStrength': calculate_edge_strength(gray),
        'NoiseLevel': calculate_noise_level(gray),
        'HighFreqEnergy': calculate_high_freq_energy(gray),
        'ColorDiversity': calculate_color_diversity(image),
        'UniqueColors': count_unique_colors(image),
        'Saturation': calculate_saturation(image),
        'TextureUniformity': calculate_texture_uniformity(gray),
        'CompressionArtifacts': detect_compression_artifacts(gray),
        'HistogramEntropy': calculate_histogram_entropy(gray),
        'DynamicRange': calculate_dynamic_range(gray),
        'Brightness': calculate_brightness(gray),
        'LightingCondition': assess_lighting(image)  # 'bright'/'normal'/'dim'/'low'
    }
    
    return metrics
```

---

## Performance Tips

### 1. Feature Importance
After training, check which metrics matter most:
- High importance = strong signal for classification
- Low importance = can potentially remove to simplify

### 2. Model Selection
- **Random Forest**: Fast, interpretable, good default
- **XGBoost**: Often better accuracy, slightly slower

### 3. Threshold Tuning
The classifier outputs probabilities. You can adjust thresholds:
```python
result = classifier.predict_single(metrics)
if result['probabilities']['real'] > 0.8:  # Stricter threshold
    # Very confident it's real
```

### 4. Handling Imbalanced Data
If you have more "real" than "duplicate" samples:
- The classifier uses `class_weight='balanced'` automatically
- Consider collecting more duplicate examples

---

## Troubleshooting

### "Model not found"
```bash
# Train a model first
python train_authenticity_classifier.py --data your_data.csv
```

### "scikit-learn not available"
```bash
pip install scikit-learn
```

### Low accuracy
- Check if your metrics are being extracted correctly
- Ensure balanced training data
- Try XGBoost instead of Random Forest
- Collect more training samples

### Slow inference
- Random Forest is already fast (~1-5ms)
- If still slow, reduce `n_estimators` in training

---

## Next Steps

1. **Collect Training Data**: Label 100+ images (real vs duplicate)
2. **Train Model**: Run training script
3. **Integrate**: Add to your verification flow
4. **Evaluate**: Test on real-world data
5. **Iterate**: Collect more data, retrain, improve

---

## When to Add CNN Features (Approach B)

Only add CNN features if:
- ✅ Your current accuracy is <90% and you need improvement
- ✅ You have GPU resources
- ✅ You can tolerate 50-200ms inference time
- ✅ You have >1000 labeled training images

For most cases, **Approach A (your 15 metrics) is sufficient and recommended**.

---

## Files Created

- `services/authenticity_classifier.py` - Classifier service
- `train_authenticity_classifier.py` - Training script
- `example_classifier_usage.py` - Usage examples

See `ML_INTEGRATION_OPTIONS.md` for more advanced options.

