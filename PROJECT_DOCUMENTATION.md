# Anti-Counterfeit QR Code System - Project Documentation

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Security Features](#security-features)
4. [API Reference](#api-reference)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Project Structure](#project-structure)
8. [Technical Details](#technical-details)
9. [Machine Learning Integration](#machine-learning-integration)
10. [Development Guidelines](#development-guidelines)
11. [Troubleshooting](#troubleshooting)

---

## Overview

### Purpose

The Anti-Counterfeit QR Code System is a production-ready solution for generating and verifying secure QR codes that cannot be scanned from:
- Photos or screenshots
- Photocopies
- Screen displays
- Low-quality reproductions

Only authentic, physically printed products can be verified, making it ideal for:
- Product authentication
- Anti-counterfeiting
- Supply chain verification
- Quality control
- Brand protection

### Key Capabilities

- **Secure QR Code Generation**: Generate QR codes with embedded CDP (Counterfeit Detection Pattern) patterns
- **Multi-Layer Security**: Multiple security features that break when copied or photographed
- **Production-Ready API**: RESTful API for integration with mobile apps and backend systems
- **Feature-Based Verification**: Uses extracted features (not raw images) for secure comparison
- **Liveness Detection**: Prevents scanning from static images, screenshots, or photos
- **Print Size**: 28×14mm print size (optimized for small codes)

### Technology Stack

- **Backend Framework**: Flask 3.0+
- **Image Processing**: OpenCV 4.8+, PIL/Pillow 10.0+
- **QR Code Generation**: qrcode 7.4+
- **Numerical Computing**: NumPy 1.24+, SciPy 1.10+
- **Security**: PyJWT 2.8+ for token generation
- **Python Version**: 3.8+

---

## System Architecture

### High-Level Architecture

```
┌─────────────┐
│   Client    │  (Mobile App / Web App)
│ Application │
└──────┬──────┘
       │ HTTP/REST API
       ▼
┌─────────────────────────────────────┐
│         Flask Application           │
│  ┌───────────────────────────────┐  │
│  │   API Endpoints               │  │
│  │   - /generate_qr_cdp          │  │
│  │   - /verify_cdp               │  │
│  └───────────────────────────────┘  │
│  ┌───────────────────────────────┐  │
│  │   Services                    │  │
│  │   - CDP Service               │  │
│  │   - Liveness Service          │  │
│  │   - Feature Extraction        │  │
│  │   - Backend Storage           │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   Backend Storage (File-based)      │
│   - CDP Features                    │
│   - Audit Logs                      │
│   - Verification History            │
└─────────────────────────────────────┘
```

### Data Flow

#### Generation Flow

1. **Client Request**: POST `/generate_qr_cdp` with `product_id`
2. **System Generates**:
   - Unique `serial_id` (UUID) - encoded in QR code
   - Unique `cdp_id` (UUID) - cryptographically random CDP pattern
   - QR code with security features
   - CDP pattern with anti-photocopy features
3. **Features Extracted**: Server extracts features from CDP (not storing raw image)
4. **Storage**: Features stored in backend with mapping: `serial_id` → `cdp_id` → `features`
5. **Response**: Returns QR+CDP image (28×14mm) and JWT token

#### Verification Flow

1. **Client Request**: POST `/verify_cdp` with:
   - `cdp_image`: Base64 encoded scanned image
   - `video_frames`: Array of video frame images (for liveness)
   - `product_id`: Product identifier (optional, extracted from QR)
2. **QR Decoding**: Extract `serial_id` from QR code
3. **Backend Lookup**: Retrieve CDP features by `serial_id`
4. **Feature Extraction**: Extract features from scanned CDP
5. **Comparison**: Compare reference and scanned features
6. **Liveness Check**: Verify video frames show real scanning (not static image)
7. **Assessment**: Additional checks for lighting, distance, size
8. **Response**: Authentication result with detailed metrics

### Security Model

#### Production-Ready Architecture

- **Serial ID System**: QR codes encode `serial_id` (UUID), not `product_id` directly
- **CDP Randomness**: Each CDP pattern is cryptographically random (not derivable)
- **Feature-Based Storage**: Only extracted features stored (not raw images)
- **Backend Lookup**: Verification requires backend database access
- **Audit Logging**: All verification attempts logged for abuse detection

#### Security Layers

1. **QR Code Security**: Enhanced QR codes with noise patterns and border security
2. **CDP Patterns**: Multiple anti-photocopy patterns that break when reproduced
3. **Feature Matching**: Robust feature extraction and comparison
4. **Liveness Detection**: Prevents static image/screenshot attacks
5. **Abuse Detection**: Monitors for suspicious verification patterns

---

## Security Features

### CDP (Counterfeit Detection Pattern) Features

#### 1. Frequency Interference Patterns
- Creates aliasing effects with scanner/copier frequencies
- Multiple frequency bands (2.0-4.0 pixel periods)
- Breaks when scanned or photocopied

#### 2. Guilloche Patterns
- Complex curved patterns (used in currency)
- Extremely difficult to reproduce accurately
- Loses fine structure when copied

#### 3. Color-Shift Security
- CMYK color combinations that don't reproduce accurately
- Cyan-Magenta and Yellow-Cyan gradients
- Color shifts when photocopied

#### 4. Embedded Security Marks
- Fine line grids that become distorted
- Corner security marks
- Watermark patterns based on CDP ID hash

#### 5. Microprinting
- Tiny text that becomes unreadable when copied
- CDP ID hash embedded in multiple locations
- Size-optimized for small codes (28×14mm)

#### 6. Holographic Effects
- Gradient overlays that don't reproduce well
- Multiple gradient layers for rich effect
- Enhanced for CMYK printing

#### 7. Screen Frequency Interference
- Patterns designed to interfere with scanner screens
- Creates visible artifacts when scanned
- Multiple frequency layers

### QR Code Security Enhancements

- **Security Noise**: Subtle noise patterns that don't affect readability
- **Border Patterns**: Embedded security patterns in borders
- **Moiré Patterns**: Fine patterns in white areas
- **Frequency Interference**: Applied to background only
- **Corner Security Marks**: Embedded marks in corners

### Liveness Detection

- **Static Image Detection**: Detects if video frames are static (screenshot detection)
- **Frame Variation**: Requires motion/variation between frames
- **Limited Role**: Secondary check (primary security is feature matching)

### Feature Extraction & Matching

#### Extracted Features

1. **Frequency Band Energy** (30% weight)
   - Multi-scale frequency analysis
   - Energy ratios across frequency bands
   - Sensitive to reproduction quality

2. **Edge Density Metrics** (25% weight)
   - Edge preservation statistics
   - Gradient magnitude analysis
   - Canny edge detection metrics

3. **Texture Descriptors** (25% weight)
   - Local Binary Pattern (LBP) histograms
   - Texture uniformity and entropy
   - Dominant pattern analysis

4. **Brightness Normalization** (20% weight)
   - Mean brightness comparison
   - Ensures images are comparable
   - Accounts for lighting variations

#### Similarity Scoring

- Weighted combination of feature differences
- Threshold: 0.65 (tuned via ROC analysis)
- Accounts for mobile capture variations

---

## API Reference

### Base URL

```
http://localhost:8000
```

### Endpoints

#### 1. Generate QR+CDP

**Endpoint**: `POST /generate_qr_cdp`

**Description**: Generate secure QR code with CDP pattern in 28×14mm size.

**Request Body**:
```json
{
  "product_id": "PRODUCT123"
}
```

**Response** (Success - 200):
```json
{
  "status": "success",
  "message": "QR+CDP generated successfully for product PRODUCT123",
  "serial_id": "550e8400-e29b-41d4-a716-446655440000",
  "cdp_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "qrToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "qrCdpImage": "base64_encoded_image_string...",
  "size": {
    "width_mm": 28,
    "height_mm": 14,
    "dpi": 2400,
    "size_name": "28x14"
  }
}
```

**Response** (Error - 400):
```json
{
  "status": "failed",
  "message": "product_id is required"
}
```

#### 2. Verify CDP

**Endpoint**: `POST /verify_cdp`

**Description**: Verify product authenticity by comparing scanned CDP with reference.

**Request Body**:
```json
{
  "cdp_image": "base64_encoded_image_string...",
  "video_frames": [
    "base64_encoded_frame1...",
    "base64_encoded_frame2...",
    "base64_encoded_frame3..."
  ],
  "product_id": "PRODUCT123"
}
```

**Response** (Success - 200):
```json
{
  "similarity_score": 0.782,
  "cdp_score": 0.856,
  "threshold": 0.65,
  "liveness_passed": true,
  "is_authentic": true,
  "serial_id": "550e8400-e29b-41d4-a716-446655440000",
  "cdp_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "product_id": "PRODUCT123",
  "status": "success",
  "message": "Product verified successfully",
  "abuse_detection": {
    "is_abuse": false,
    "abuse_flags": [],
    "total_attempts": 1
  },
  "frame_distance": {
    "coverage_ratio": 32.5,
    "has_warnings": false,
    "warnings": []
  },
  "lighting": {
    "status": "good",
    "quality_score": 85,
    "has_warnings": false,
    "has_critical_warnings": false,
    "warnings": [],
    "metrics": {
      "brightness": 128.5,
      "contrast": 45.2,
      "dynamic_range": 220.0
    }
  },
  "pattern_size": {
    "size_category": "normal",
    "coverage_ratio": 32.5,
    "width_pixels": 650,
    "height_pixels": 325,
    "aspect_ratio": 2.0,
    "has_warnings": false,
    "warnings": []
  }
}
```

**Response** (Failed Verification - 200):
```json
{
  "similarity_score": 0.45,
  "cdp_score": 0.523,
  "threshold": 0.65,
  "liveness_passed": true,
  "is_authentic": false,
  "status": "failed",
  "message": "Product verification failed - similarity score 0.450 below threshold 0.65"
}
```

**Response** (Error - 400):
```json
{
  "status": "failed",
  "message": "video_frames are required (minimum 2 frames) to prevent static screenshot scanning."
}
```

**Response** (Error - 404):
```json
{
  "status": "failed",
  "message": "CDP not found for serial_id: <serial_id>. Item may not be registered."
}
```

**Response Fields**:
- `similarity_score`: Feature-based similarity score (0.0-1.0) comparing extracted features from reference and scanned CDP
- `cdp_score`: Direct image comparison score (0.0-1.0) comparing the scanned CDP image with the reference CDP image using multi-metric analysis (SSIM, correlation, histogram, edge, texture, etc.)
- `threshold`: Similarity threshold used for authentication (default: 0.65)
- `liveness_passed`: Whether liveness detection passed (prevents static screenshot attacks)
- `is_authentic`: Final authentication result (true if similarity_score >= threshold AND liveness_passed)

### Error Codes

- **200**: Request successful (check `status` field in response)
- **400**: Bad request (missing/invalid parameters)
- **404**: Resource not found (CDP not registered)
- **500**: Internal server error

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation Steps

1. **Clone/Navigate to Project Directory**
```bash
cd anti_counterfeit_python
```

2. **Create Virtual Environment** (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

**Optional: Install ML Dependencies**

For ML-enhanced verification (CNN features):
```bash
pip install torch torchvision
```

For authenticity classifier:
```bash
pip install scikit-learn joblib pandas xgboost
```

4. **Create Required Directories** (if not auto-created)
```bash
mkdir -p generated_qr_cdp
mkdir -p generated_cdp
mkdir -p scanned_images
mkdir -p extracted_cdp
mkdir -p backend_storage
```

5. **Start the Server**
```bash
python3 app.py
```

The server will start on `http://localhost:8000`

### Environment Variables

Create a `.env` file in the project root (optional):

```env
# JWT Secret Key (for QR token generation)
JWT_SECRET_KEY=your-secret-key-here

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

---

## Usage Guide

### Basic Usage Examples

#### Generate QR Code (Python)

```python
import requests
import base64

# Generate QR+CDP
response = requests.post(
    "http://localhost:8000/generate_qr_cdp",
    json={"product_id": "PRODUCT123"}
)

result = response.json()
if result["status"] == "success":
    # Save QR+CDP image (28x14mm)
    image_data = base64.b64decode(result["qrCdpImage"])
    with open("product_qr_cdp.png", "wb") as f:
        f.write(image_data)
    
    print(f"Serial ID: {result['serial_id']}")
    print(f"CDP ID: {result['cdp_id']}")
    print(f"Size: {result['size']['size_name']} ({result['size']['width_mm']}x{result['size']['height_mm']}mm)")
```

#### Verify Product (Python)

```python
import requests
import base64
from PIL import Image
import io

# Read scanned image
with open("scanned_product.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Simulate video frames (in real app, capture from camera)
frames = [image_data, image_data, image_data]  # Should be different frames

# Verify
response = requests.post(
    "http://localhost:8000/verify_cdp",
    json={
        "cdp_image": image_data,
        "video_frames": frames,
        "product_id": "PRODUCT123"
    }
)

result = response.json()
if result["is_authentic"]:
    print("✅ Product is authentic!")
    print(f"Similarity Score: {result['similarity_score']:.3f}")
else:
    print("❌ Product verification failed")
    print(f"Score: {result['similarity_score']:.3f} (threshold: {result['threshold']})")
```

#### Generate QR Code (cURL)

```bash
curl -X POST http://localhost:8000/generate_qr_cdp \
  -H "Content-Type: application/json" \
  -d '{"product_id": "PRODUCT123"}' \
  > response.json
```

#### Verify Product (cURL)

```bash
# Prepare base64 encoded image
IMAGE_B64=$(base64 -i scanned_product.png)

curl -X POST http://localhost:8000/verify_cdp \
  -H "Content-Type: application/json" \
  -d "{
    \"cdp_image\": \"$IMAGE_B64\",
    \"video_frames\": [\"$IMAGE_B64\", \"$IMAGE_B64\"],
    \"product_id\": \"PRODUCT123\"
  }"
```

### Mobile App Integration

#### iOS (Swift) Example

```swift
// Generate QR Code
func generateQRCode(productId: String) {
    let url = URL(string: "http://your-server:8000/generate_qr_cdp")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    
    let body: [String: Any] = ["product_id": productId]
    request.httpBody = try? JSONSerialization.data(withJSONObject: body)
    
    URLSession.shared.dataTask(with: request) { data, response, error in
        if let data = data {
            let result = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
            // Handle response
        }
    }.resume()
}

// Verify Product
func verifyProduct(imageData: Data, frames: [Data]) {
    let url = URL(string: "http://your-server:8000/verify_cdp")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    
    let imageBase64 = imageData.base64EncodedString()
    let framesBase64 = frames.map { $0.base64EncodedString() }
    
    let body: [String: Any] = [
        "cdp_image": imageBase64,
        "video_frames": framesBase64,
        "product_id": "PRODUCT123"
    ]
    request.httpBody = try? JSONSerialization.data(withJSONObject: body)
    
    URLSession.shared.dataTask(with: request) { data, response, error in
        // Handle response
    }.resume()
}
```

#### Android (Kotlin) Example

```kotlin
// Generate QR Code
fun generateQRCode(productId: String) {
    val client = OkHttpClient()
    val json = JSONObject().apply {
        put("product_id", productId)
    }
    
    val requestBody = json.toString().toRequestBody("application/json".toMediaType())
    val request = Request.Builder()
        .url("http://your-server:8000/generate_qr_cdp")
        .post(requestBody)
        .build()
    
    client.newCall(request).enqueue(object : Callback {
        override fun onResponse(call: Call, response: Response) {
            // Handle response
        }
        override fun onFailure(call: Call, e: IOException) {
            // Handle error
        }
    })
}
```

### Printing Guidelines

#### Recommended Print Specifications

- **Printer Resolution**: 2400 DPI (minimum: 1200 DPI)
- **Paper Type**: 200-300 GSM coated paper
- **Print Size**: 28×14mm
- **Color Mode**: CMYK (TIFF format preferred)
- **File Format**: TIFF (CMYK) or PNG (RGB fallback)

#### Print Quality Testing

```bash
python3 test_print_quality.py
```

---

## Project Structure

```
anti_counterfeit_python/
├── app.py                          # Main Flask application
├── generate_qr_cdp.py              # QR+CDP generation logic
├── requirements.txt                # Python dependencies
├── README.md                       # Quick start guide
├── PROJECT_DOCUMENTATION.md        # This file
│
├── services/                       # Core services
│   ├── __init__.py
│   ├── cdp_service.py             # CDP comparison and extraction
│   ├── liveness_service.py        # Liveness detection
│   ├── feature_extraction.py      # Feature extraction and comparison
│   ├── backend_storage.py         # Backend storage (file-based)
│   ├── generate_qr_token.py       # JWT token generation
│   ├── ml_feature_extraction.py   # CNN feature extraction (ML Approach B)
│   └── authenticity_classifier.py  # Authenticity classifier (ML Approach A)
│
├── utils/                          # Utility functions
│   ├── __init__.py
│   └── image_utils.py             # Image processing utilities
│
├── generated_qr_cdp/               # Generated QR+CDP images
├── generated_cdp/                  # Reference CDP patterns
├── scanned_images/                 # Scanned images (verification)
├── extracted_cdp/                  # Extracted CDP regions
├── cropped_qr_cdp/                 # Cropped yellow border images
├── backend_storage/                # Backend storage (features, audit logs)
│   ├── features/                   # CDP feature files
│   └── audit/                      # Audit logs
│
├── training_data/                  # ML training data
│   └── sample_data.csv            # Sample training data template
│
├── models/                         # Trained ML models (created after training)
│   └── authenticity_classifier_*.pkl # Trained classifier models
│
├── train_authenticity_classifier.py # ML classifier training script
├── example_ml_integration.py      # ML integration examples
├── example_classifier_usage.py     # Classifier usage examples
├── ML_FEATURES_EXPLAINED.md       # CNN features documentation
├── ML_INTEGRATION_OPTIONS.md      # ML integration options
├── QUICK_START_ML.md              # Quick start guide for ML
│
└── test_images/                    # Test images
```

### Key Files

- **app.py**: Flask application with API endpoints
- **generate_qr_cdp.py**: Core generation logic with security patterns
- **services/cdp_service.py**: CDP extraction and comparison
- **services/liveness_service.py**: Liveness detection and quality assessment
- **services/feature_extraction.py**: Feature extraction and similarity scoring
- **services/backend_storage.py**: File-based backend storage (production would use database)
- **services/ml_feature_extraction.py**: CNN feature extraction for ML-enhanced verification
- **services/authenticity_classifier.py**: Authenticity classifier using 15 metrics
- **train_authenticity_classifier.py**: Training script for authenticity classifier

---

## Technical Details

### CDP Pattern Generation

#### Pattern Parameters (Optimized for Small Codes)

For 28×14mm codes at 2400 DPI:
- **Moiré Spacing**: ~0.15mm physical spacing
- **Micro-dot Size**: ~0.35mm physical size
- **Line Grid Spacing**: ~0.25mm
- **Frequency Periods**: 2.0-4.0 pixels (interferes with 300-600 DPI scanners)
- **Guilloche Scale**: 1.2x (increased for visibility)

#### Pattern Application Order

1. Base random pattern (cryptographically secure)
2. Anti-photocopy patterns (frequency interference, color-shift, etc.)
3. Holographic effects
4. Guilloche patterns
5. Screen frequency interference
6. Microprinting

### Feature Extraction Details

#### Frequency Analysis

- **Downsampling**: To 256×256 for FFT efficiency
- **Frequency Bands**:
  - Low: 0.0-1.5 pixel periods
  - Medium-Low: 1.5-2.5
  - Medium: 2.5-3.5
  - Medium-High: 3.5-4.5
  - High: 4.5-10.0
- **Energy Ratios**: Normalized ratios for stability

#### Texture Analysis (LBP)

- **Resolution**: Downsampled to 128×128
- **LBP Calculation**: Vectorized 8-neighborhood
- **Histogram Analysis**: Uniformity, entropy, dominant patterns
- **Comparison**: Histogram correlation

#### Edge Analysis

- **Normalization**: Min-max normalization
- **Gradient Calculation**: Sobel operators (3×3 kernel)
- **Edge Detection**: Canny (50, 150 thresholds)
- **Metrics**: Density, strength (mean, std, max)

### Similarity Scoring Algorithm

```python
# Weighted combination
similarity = (
    freq_score * 0.30 +      # Frequency bands
    edge_score * 0.25 +      # Edge metrics
    texture_score * 0.25 +   # Texture (LBP)
    brightness_score * 0.20  # Brightness normalization
)

# Threshold: 0.65 (tuned via ROC analysis)
is_authentic = similarity >= 0.65 and liveness_passed
```

### Backend Storage Schema

#### Feature Record
```json
{
  "cdp_id": "uuid",
  "product_id": "string",
  "serial_id": "uuid",
  "features": {
    "freq_low": 0.123,
    "freq_medium": 0.456,
    "edge_density": 0.789,
    "texture_lbp_uniformity": 0.234,
    ...
  },
  "created_at": "ISO8601",
  "updated_at": "ISO8601"
}
```

#### Audit Log Entry
```json
{
  "cdp_id": "uuid",
  "serial_id": "uuid",
  "success": true,
  "score": 0.782,
  "timestamp": "ISO8601",
  "metadata": {
    "ip": "127.0.0.1",
    "user_agent": "Mozilla/5.0..."
  }
}
```

---

## Machine Learning Integration

### Overview

The system supports two complementary ML approaches for enhanced verification:

1. **CNN Feature Extraction** (Approach B): Deep learning features from pre-trained CNNs
2. **Authenticity Classifier** (Approach A): Supervised classifier using 15 extracted metrics

Both approaches can be used alongside the existing rule-based feature matching for improved accuracy.

### ML Approach A: Authenticity Classifier

#### Description

A fast, interpretable classifier that uses 15 image quality metrics to distinguish between:
- **"real"**: Authentic printed QR codes
- **"duplicate"**: Photocopies, screenshots, or digital reproductions

#### Features

- ✅ Fast inference (~1-5ms)
- ✅ No GPU required
- ✅ Interpretable (feature importance analysis)
- ✅ Uses only your 15 metrics (no CNN needed)
- ✅ Works with CPU-only systems

#### Input Features (15 Metrics)

The classifier uses the following metrics extracted from CDP images:

1. **Sharpness**: Image sharpness score
2. **Contrast**: Contrast level
3. **HistogramPeak**: Histogram peak value
4. **EdgeDensity**: Edge density metric
5. **EdgeStrength**: Edge strength measurement
6. **NoiseLevel**: Noise level detection
7. **HighFreqEnergy**: High frequency energy
8. **ColorDiversity**: Color diversity metric
9. **UniqueColors**: Count of unique colors
10. **Saturation**: Color saturation
11. **TextureUniformity**: Texture uniformity measure
12. **CompressionArtifacts**: Compression artifact detection
13. **HistogramEntropy**: Histogram entropy
14. **DynamicRange**: Dynamic range
15. **Brightness**: Average brightness
16. **LightingCondition**: Lighting condition (bright/normal/dim/low)

#### Training the Classifier

**Step 1: Prepare Training Data**

Create a CSV file with labeled data. Each row should contain all 15 metrics plus `LightingCondition` and `Label`:

```csv
Sharpness,Contrast,HistogramPeak,EdgeDensity,EdgeStrength,NoiseLevel,HighFreqEnergy,ColorDiversity,UniqueColors,Saturation,TextureUniformity,CompressionArtifacts,HistogramEntropy,DynamicRange,Brightness,LightingCondition,Label
107.54,84.59,0.096,0.0230,23.57,2.00,40162258944.00,0.0003,2519,41.02,0.0593,167.03,6.35,254.00,70.93,bright,real
45.23,35.21,0.234,0.0123,12.34,8.45,12345678901.23,0.0001,856,28.45,0.1234,345.67,5.12,180.00,45.23,normal,duplicate
```

**Minimum recommended**: 100+ samples (50+ per class)

**Step 2: Create Sample Template**

```bash
python train_authenticity_classifier.py --create-sample
```

This creates `training_data/sample_data.csv` - replace with your actual labeled data.

**Step 3: Train the Model**

```bash
python train_authenticity_classifier.py --data training_data/qr_metrics_labeled.csv --model-type random_forest
```

Options:
- `--data`: Path to your training CSV
- `--model-type`: `random_forest` (default) or `xgboost`
- `--output`: Custom output path (default: `models/authenticity_classifier_{type}.pkl`)
- `--test-size`: Fraction for testing (default: 0.2)

**Model Types**:
- **Random Forest**: Fast, interpretable, good default
- **XGBoost**: Often better accuracy, slightly slower

#### Using the Classifier

**Basic Usage**:

```python
from services.authenticity_classifier import AuthenticityClassifier

# Load trained model
classifier = AuthenticityClassifier(
    model_path='models/authenticity_classifier_random_forest.pkl'
)

# Your extracted metrics
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

**Integration in API**:

```python
from services.authenticity_classifier import get_classifier

# In verify_cdp() function, after extracting CDP:
classifier = get_classifier(
    model_type='random_forest',
    model_path='models/authenticity_classifier_random_forest.pkl'
)

# Extract metrics from scanned_cdp
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
    return jsonify({
        'status': 'failed',
        'message': 'ML classifier detected duplicate/photocopy',
        'ml_confidence': ml_result['confidence']
    })
```

#### Performance

- **Inference Time**: ~1-5ms per prediction
- **Accuracy**: Typically 90-95% with sufficient training data
- **Feature Importance**: Model provides interpretable feature importance rankings

### ML Approach B: CNN Feature Extraction

#### Description

Uses pre-trained Convolutional Neural Networks (CNNs) to extract deep learning feature vectors from CDP images. These features capture complex visual patterns that complement rule-based features.

#### Features

- ✅ Captures complex, non-obvious patterns
- ✅ Learned from vast amounts of data (ImageNet)
- ✅ Can detect subtle differences
- ⚠️ Requires PyTorch (optional dependency)
- ⚠️ Slower inference (50-200ms, faster with GPU)
- ⚠️ Less interpretable (black box)

#### Supported Models

- **ResNet50**: 2048-dimensional feature vector (recommended)
- **ResNet18**: 512-dimensional feature vector (faster, smaller)
- **VGG16**: 25,088-dimensional feature vector (larger, more detailed)

#### Installation

```bash
pip install torch torchvision
```

#### Using CNN Features

**Basic Usage**:

```python
from services.ml_feature_extraction import extract_ml_features, compare_ml_features

# Extract features from reference and scanned CDPs
ref_ml_features = extract_ml_features(reference_cdp)    # (2048,)
scan_ml_features = extract_ml_features(scanned_cdp)      # (2048,)

# Compare using cosine similarity
ml_score = compare_ml_features(ref_ml_features, scan_ml_features)
# Returns: 0.0 to 1.0 (1.0 = identical)
```

**Advanced Usage**:

```python
from services.ml_feature_extraction import MLFeatureExtractor

# Initialize extractor
extractor = MLFeatureExtractor(
    model_name='resnet50',  # or 'resnet18', 'vgg16'
    use_gpu=True  # Use GPU if available
)

# Extract features
features = extractor.extract_features(cdp_image)

# Compare features
similarity = extractor.compare_features(ref_features, scan_features)
```

#### Ensemble Approach

Combine CNN features with rule-based features for enhanced verification:

```python
from services.feature_extraction import extract_all_features, compare_features
from services.ml_feature_extraction import extract_ml_features, compare_ml_features

# Rule-based features (existing method)
ref_rule_features = extract_all_features(reference_cdp)
scan_rule_features = extract_all_features(scanned_cdp)
rule_score = compare_features(ref_rule_features, scan_rule_features)

# ML features (new)
ref_ml_features = extract_ml_features(reference_cdp)
scan_ml_features = extract_ml_features(scanned_cdp)
ml_score = compare_ml_features(ref_ml_features, scan_ml_features)

# Ensemble combination (adjust weights based on performance)
ml_weight = 0.3
rule_weight = 0.7
final_score = rule_weight * rule_score + ml_weight * ml_score
```

#### What CNN Features Capture

The features are **learned representations** from ImageNet training (1.2 million images):

- **Texture patterns**: Complex textures, patterns, and visual structures
- **Spatial relationships**: How patterns relate to each other spatially
- **Hierarchical features**:
  - Low-level: edges, corners, gradients
  - Mid-level: shapes, patterns, textures
  - High-level: complex visual structures
- **Invariant representations**: Robust to lighting, rotation, scale, minor distortions

#### Performance

- **Inference Time**: 
  - CPU: 100-200ms per image
  - GPU: 50-100ms per image
- **Feature Dimensions**: 
  - ResNet50: 2048
  - ResNet18: 512
  - VGG16: 25,088
- **Memory**: ~200-500MB for model loading

### Choosing the Right Approach

#### Use Approach A (Authenticity Classifier) When:

- ✅ You need fast inference (<10ms)
- ✅ You don't have GPU resources
- ✅ You want interpretable results
- ✅ You have 100+ labeled training samples
- ✅ You want to understand which metrics matter most

#### Use Approach B (CNN Features) When:

- ✅ You need higher accuracy and current accuracy is <90%
- ✅ You have GPU resources available
- ✅ You can tolerate 50-200ms inference time
- ✅ You have >1000 labeled training images
- ✅ You want to capture complex patterns not captured by metrics

#### Use Both (Ensemble) When:

- ✅ You want maximum accuracy
- ✅ You have both labeled data and GPU resources
- ✅ You can combine rule-based, classifier, and CNN features

### ML Integration Files

- **`services/authenticity_classifier.py`**: Authenticity classifier service (Approach A)
- **`services/ml_feature_extraction.py`**: CNN feature extraction (Approach B)
- **`train_authenticity_classifier.py`**: Training script for classifier
- **`example_ml_integration.py`**: Integration examples
- **`example_classifier_usage.py`**: Classifier usage examples

### Training Data Requirements

#### Authenticity Classifier

- **Minimum**: 100 samples (50 per class)
- **Recommended**: 500+ samples (250+ per class)
- **Format**: CSV with 15 metrics + LightingCondition + Label
- **Balance**: Balanced classes recommended (use `class_weight='balanced'`)

#### CNN Features

- **Pre-trained**: Uses ImageNet pre-trained models (no training needed)
- **Fine-tuning** (optional): Requires 1000+ labeled images if fine-tuning
- **Transfer learning**: Works well out-of-the-box for feature extraction

### Performance Tips

1. **Feature Importance**: After training classifier, check which metrics matter most
2. **Model Selection**: 
   - Random Forest: Fast, interpretable, good default
   - XGBoost: Often better accuracy, slightly slower
3. **Threshold Tuning**: Adjust prediction thresholds based on your use case
4. **Ensemble Weights**: Tune `ml_weight` and `rule_weight` based on validation performance
5. **GPU Acceleration**: Use GPU for CNN features if available (10-20x speedup)

### Troubleshooting

#### "PyTorch not available"

```bash
pip install torch torchvision
```

#### "scikit-learn not available"

```bash
pip install scikit-learn
```

#### "Model not found"

```bash
# Train a model first
python train_authenticity_classifier.py --data your_data.csv
```

#### Low accuracy

- Check if metrics are being extracted correctly
- Ensure balanced training data
- Try XGBoost instead of Random Forest
- Collect more training samples
- For CNN features, try different models (ResNet18 vs ResNet50)

#### Slow inference

- **Classifier**: Already fast (~1-5ms), reduce `n_estimators` if needed
- **CNN**: Use GPU if available, or try ResNet18 (smaller, faster)

### Additional Resources

- **`ML_FEATURES_EXPLAINED.md`**: Detailed explanation of CNN features
- **`ML_INTEGRATION_OPTIONS.md`**: Advanced integration options
- **`QUICK_START_ML.md`**: Quick start guide for ML integration

---

## Development Guidelines

### Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Document functions with docstrings
- Use meaningful variable names

### Adding New Security Patterns

1. Create pattern function in `generate_qr_cdp.py`
2. Add to `add_anti_photocopy_pattern()` chain
3. Test with 28×14mm codes
4. Verify pattern breaks when photocopied
5. Update documentation

### Adding New Features

1. Create feature extraction function in `services/feature_extraction.py`
2. Add to `extract_all_features()`
3. Add comparison logic to `compare_features()`
4. Tune weights via ROC analysis
5. Update documentation

### Testing

#### Unit Tests
```bash
# Run unit tests (if available)
pytest tests/
```

#### Integration Tests
```bash
# Test generation
python3 generate_qr_cdp.py

# Test verification
python3 test_print_quality.py
```

#### Manual Testing
```bash
# Generate test QR codes
python3 generate_dummy_qr_codes.py

# Test batch generation
python3 generate_batch_qr_cdp.py
```

### Performance Considerations

- **Feature Extraction**: Optimized with downsampling
- **FFT Operations**: Limited to 256×256 for speed
- **LBP Calculation**: Vectorized for efficiency
- **Storage**: Features only (not raw images)
- **Caching**: Consider caching for frequently accessed features

### Production Deployment

#### Recommended Changes

1. **Database**: Replace file-based storage with PostgreSQL/MongoDB
2. **Authentication**: Add API key authentication
3. **Rate Limiting**: Implement rate limiting per IP/client
4. **Logging**: Use structured logging (JSON format)
5. **Monitoring**: Add health check endpoint
6. **Caching**: Redis for frequently accessed data
7. **CDN**: Serve generated images via CDN
8. **SSL/TLS**: Use HTTPS for all API calls

#### Environment Configuration

```python
# Production settings
PRODUCTION = os.getenv('ENVIRONMENT') == 'production'
DEBUG = not PRODUCTION
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', 8000))
DATABASE_URL = os.getenv('DATABASE_URL')
REDIS_URL = os.getenv('REDIS_URL')
```

---

## Troubleshooting

### Common Issues

#### 1. QR Code Not Scannable

**Symptoms**: QR code cannot be scanned by mobile apps

**Solutions**:
- Increase print resolution to 2400 DPI
- Ensure high contrast (black/white)
- Check print quality (no smudging, blur)
- Ensure proper lighting and camera focus when scanning

#### 2. Low CDP Score

**Symptoms**: Similarity score below threshold (0.65)

**Solutions**:
- Use high-quality printer (2400 DPI)
- Improve lighting conditions
- Ensure proper camera focus
- Move camera to optimal distance (20-45% frame coverage)
- Check for print defects

#### 3. Liveness Check Failing

**Symptoms**: `liveness_passed: false`

**Solutions**:
- Use live camera feed (not static image)
- Ensure video frames show variation
- Capture multiple frames (minimum 2)
- Use real printed material (not screen/photocopy)

#### 4. Backend Storage Errors

**Symptoms**: Storage operations failing

**Solutions**:
- Check directory permissions
- Ensure `backend_storage/` directory exists
- Check disk space
- Verify file permissions

#### 5. Import Errors

**Symptoms**: Module import failures

**Solutions**:
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python version (requires 3.8+)
python3 --version

# Verify virtual environment is activated
which python  # Should point to venv
```

### Debug Mode

Enable debug logging:
```python
# In app.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Issues

#### Slow Generation

- Check system resources (CPU, memory)
- Reduce number of simultaneous requests
- Consider async processing for batch operations

#### Slow Verification

- Optimize feature extraction (already downsampled)
- Consider caching extracted features
- Use database indexes for serial_id lookups

---

## Additional Resources

### Related Documentation

- [README.md](README.md) - Quick start guide
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) - Detailed API reference (if exists)
- [PRINTING_REQUIREMENTS.md](PRINTING_REQUIREMENTS.md) - Printing specifications (if exists)

### External References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [QR Code Standards](https://www.qrcode.com/en/)
- [CMYK Printing Guide](https://www.printingcenterusa.com/printing-resources/cmyk-color-model/)

---

## Version History

- **v1.0** (November 2025)
  - Initial release
  - Multi-layer security features
  - Production-ready API
  - Feature-based verification
  - Multiple print sizes support

---

## License

[Specify your license here]

---

## Support

For issues, questions, or contributions, please:
1. Check this documentation first
2. Review error messages and logs
3. Test with provided scripts
4. [Contact information or issue tracker]

---

**Last Updated**: December 2025

