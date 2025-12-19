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
9. [Development Guidelines](#development-guidelines)
10. [Troubleshooting](#troubleshooting)

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
- **Multiple Print Sizes**: Support for 28×14mm, 40×20mm, and 50×25mm print sizes

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
5. **Response**: Returns QR+CDP images (3 sizes) and JWT token

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

**Description**: Generate secure QR code with CDP pattern in multiple sizes.

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
  "message": "QR+CDP generated successfully in all 3 sizes for product PRODUCT123",
  "serial_id": "550e8400-e29b-41d4-a716-446655440000",
  "cdp_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "qrToken": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "qrCdpImages": {
    "28x14": "base64_encoded_image_string...",
    "40x20": "base64_encoded_image_string...",
    "50x25": "base64_encoded_image_string..."
  },
  "sizes": {
    "28x14": {"width_mm": 28, "height_mm": 14, "dpi": 2400},
    "40x20": {"width_mm": 40, "height_mm": 20, "dpi": 2400},
    "50x25": {"width_mm": 50, "height_mm": 25, "dpi": 2400}
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
    # Save 28x14mm image
    image_28x14 = base64.b64decode(result["qrCdpImages"]["28x14"])
    with open("product_28x14.png", "wb") as f:
        f.write(image_28x14)
    
    print(f"Serial ID: {result['serial_id']}")
    print(f"CDP ID: {result['cdp_id']}")
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
- **Print Sizes**: 
  - Small: 28×14mm
  - Medium: 40×20mm
  - Large: 50×25mm
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
│   └── generate_qr_token.py       # JWT token generation
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
└── test_images/                    # Test images
```

### Key Files

- **app.py**: Flask application with API endpoints
- **generate_qr_cdp.py**: Core generation logic with security patterns
- **services/cdp_service.py**: CDP extraction and comparison
- **services/liveness_service.py**: Liveness detection and quality assessment
- **services/feature_extraction.py**: Feature extraction and similarity scoring
- **services/backend_storage.py**: File-based backend storage (production would use database)

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

## Development Guidelines

### Code Style

- Follow PEP 8 style guide
- Use type hints where appropriate
- Document functions with docstrings
- Use meaningful variable names

### Adding New Security Patterns

1. Create pattern function in `generate_qr_cdp.py`
2. Add to `add_anti_photocopy_pattern()` chain
3. Test with small codes (28×14mm) and large codes (50×25mm)
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
- Increase physical size to 40×20mm or larger
- Ensure high contrast (black/white)
- Check print quality (no smudging, blur)

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

