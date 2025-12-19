# 🛡️ Anti-Counterfeit QR Code System

<div align="center">

**Advanced QR Code Authentication System with Multi-Layer Security**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange.svg)](https://opencv.org/)

*Prevent scanning from images, photocopies, and screenshots*

</div>

---

## 🎯 What Does This System Do?

This system generates **secure QR codes** that cannot be scanned from:
- ❌ Photos or screenshots
- ❌ Photocopies
- ❌ Screen displays
- ❌ Low-quality reproductions

Only **authentic, physically printed** products can be verified! ✅

---

## ✨ Key Features

### 🔒 Multi-Layer Security

1. **Anti-Photocopy Patterns** - Moiré patterns, micro-dots, fine gradients
2. **Microprinting** - Tiny text that breaks when copied
3. **Holographic Effects** - Color-shifting patterns
4. **Enhanced QR Codes** - Security noise and border patterns
5. **Liveness Detection** - Blur, screen, photocopy, and static image detection
6. **CDP Pattern Matching** - Pixel-perfect comparison

### 📡 RESTful API

- **Generate** secure QR codes with embedded CDP patterns
- **Verify** product authenticity with multi-layer checks
- **Comprehensive** error handling and responses

### 🛠️ Quality Tools

- Print quality testing
- Batch generation (1000+ QR codes)
- Automated quality checks

---

## 🚀 Quick Start

### Installation

```bash
# Clone or navigate to project directory
cd anti_counterfeit_python

# Install dependencies
pip install -r requirements.txt

# Start the server
python3 app.py
```

Server runs on: `http://localhost:8000`

### Generate Your First QR Code

```bash
curl -X POST http://localhost:8000/generate_qr_cdp \
  -H "Content-Type: application/json" \
  -d '{"product_id": "PRODUCT123"}'
```

### Verify a Product

```bash
curl -X POST http://localhost:8000/verify_cdp \
  -H "Content-Type: application/json" \
  -d '{
    "cdp_image": "<base64_image>",
    "video_frames": ["<frame1>", "<frame2>"],
    "product_id": "PRODUCT123"
  }'
```

---

## 📊 System Overview

```
┌─────────────┐
│  Generate   │ → QR Code + CDP Pattern + Security Features
│  QR Code    │
└─────────────┘
       │
       ▼
┌─────────────┐
│   Print     │ → High-quality printer (2400 DPI recommended)
│   Product   │
└─────────────┘
       │
       ▼
┌─────────────┐
│   Scan      │ → Mobile app with live camera feed
│   Product   │
└─────────────┘
       │
       ▼
┌─────────────┐
│  Verify     │ → Multi-layer security checks
│  Authentic  │
└─────────────┘
```

---

## 📁 Project Structure

```
anti_counterfeit_python/
├── app.py                    # Main Flask application
├── generate_qr_cdp.py        # QR+CDP generation
├── services/                 # Core services
│   ├── cdp_service.py       # CDP comparison
│   ├── liveness_service.py  # Liveness detection
│   └── generate_qr_token.py  # JWT tokens
├── utils/                    # Utilities
│   └── image_utils.py       # Image processing
├── generated_qr_cdp/         # Generated QR codes
├── generated_cdp/            # Reference CDP patterns
└── Documentation/            # Complete documentation
```

---

## 🔐 Security Layers

| Layer | Feature | Purpose |
|-------|---------|---------|
| 1 | Anti-Photocopy Patterns | Break when copied |
| 2 | Microprinting | Unreadable when copied |
| 3 | Holographic Effects | Don't reproduce well |
| 4 | Enhanced QR Code | Security noise patterns |
| 5 | Liveness Detection | Verify physical presence |
| 6 | CDP Matching | Pixel-perfect comparison |

---

## 📚 Documentation

### Complete Documentation
- **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** - Complete project overview
- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - Full API reference
- **[API_QUICK_REFERENCE.md](API_QUICK_REFERENCE.md)** - Quick API guide

### Specialized Guides
- **[PRINTING_REQUIREMENTS.md](PRINTING_REQUIREMENTS.md)** - Printing specifications
- **[CDP_VERIFICATION_FLOW.md](CDP_VERIFICATION_FLOW.md)** - Verification process with diagrams
- **[REAL_WORLD_SCANNING_GUIDE.md](REAL_WORLD_SCANNING_GUIDE.md)** - Natural lighting & limitations
- **[LIGHTING_WARNINGS_GUIDE.md](LIGHTING_WARNINGS_GUIDE.md)** - Lighting warnings feature ⭐ NEW
- **[MOBILE_APP_IMPLEMENTATION_SPEC.md](MOBILE_APP_IMPLEMENTATION_SPEC.md)** - Mobile app implementation guide

---

## 🧪 Testing

### Generate Test QR Codes

```bash
# Generate 1000 test QR codes
python3 generate_dummy_qr_codes.py
```

### Test Print Quality

```bash
# Test print quality of scanned QR code
python3 test_print_quality.py
```

---

## 🖨️ Printing Requirements

### Minimum
- **Printer**: 1200 DPI
- **Paper**: 200+ GSM, coated
- **Size**: 2cm x 2cm

### Recommended
- **Printer**: 2400 DPI inkjet
- **Paper**: 250-300 GSM photo paper
- **Size**: 3cm x 3cm or larger

See [PRINTING_REQUIREMENTS.md](PRINTING_REQUIREMENTS.md) for details.

---

## 📈 Performance

- **Generation**: ~139 products/second
- **Verification**: < 500ms per product
- **Batch (1000)**: ~7 seconds

---

## 🎯 Use Cases

✅ **Product Authentication**  
✅ **Anti-Counterfeiting**  
✅ **Supply Chain Verification**  
✅ **Quality Control**  
✅ **Brand Protection**  

---

## 🛠️ Technologies

- **Flask** - Web framework
- **OpenCV** - Image processing
- **qrcode** - QR code generation
- **NumPy** - Numerical operations
- **Pillow** - Image manipulation
- **PyJWT** - Token generation

---

## 📝 API Endpoints

### Generate QR Code
```
POST /generate_qr_cdp
Body: { "product_id": "PRODUCT123" }
```

### Verify Product
```
POST /verify_cdp
Body: {
  "cdp_image": "<base64>",
  "video_frames": ["<frame1>", "<frame2>"],
  "product_id": "PRODUCT123"
}
```

---

## 🔧 Troubleshooting

### QR Code Not Scannable?
- Increase print resolution to 2400 DPI
- Increase size to 3cm x 3cm
- Ensure high contrast

### Low CDP Score?
- Use high-quality printer
- Improve lighting
- Ensure proper focus

### Liveness Check Failing?
- Use live camera feed (not static image)
- Ensure real printed material
- Use high-quality prints

---

## 🚀 Quick Examples

### Python Example

```python
import requests

# Generate QR code
response = requests.post(
    "http://localhost:8000/generate_qr_cdp",
    json={"product_id": "PRODUCT123"}
)
result = response.json()
print(result["qrCdpImage"])  # Base64 encoded image

# Verify product
response = requests.post(
    "http://localhost:8000/verify_cdp",
    json={
        "cdp_image": cdp_image,
        "video_frames": [frame1, frame2, frame3],
        "product_id": "PRODUCT123"
    }
)
result = response.json()
if result["is_authentic"]:
    print("✅ Authentic!")
```

---

## 📊 What We've Built

### ✅ Implemented Features

- [x] Enhanced QR code generation with security features
- [x] Anti-photocopy patterns (moiré, micro-dots, gradients)
- [x] Microprinting with product hash
- [x] Holographic-like effects
- [x] Multi-layer liveness detection
- [x] CDP pattern matching
- [x] RESTful API endpoints
- [x] File-based CDP storage
- [x] Print quality testing
- [x] Batch generation tools
- [x] Comprehensive documentation

### 📦 Generated Assets

- 1000+ test QR codes ready for scanning
- Complete API documentation
- Printing requirements guide
- Quality testing tools
- Visual flow diagrams

---

## 🎓 Learn More

For complete details, see:
- **[PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)** - Full project documentation

---

## 📞 Support

- Check documentation files
- Review error messages
- Check server logs
- Test with quality script

---

<div align="center">

**Built with ❤️ for Anti-Counterfeiting**

*Version 1.0 | November 2025*

</div>

