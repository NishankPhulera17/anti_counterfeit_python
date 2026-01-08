# Device Information API Update

## Summary

Updated the `/verify_cdp` API endpoint to accept and store device information (manufacturer, model, OS, camera megapixels) in the training data CSV for ML model training and analysis.

## Changes Made

### 1. API Endpoint Updated (`app.py`)

#### New Optional Parameters

The `/verify_cdp` endpoint now accepts the following optional parameters in the request payload:

```typescript
{
  cdp_image: string;                  // Required (base64)
  product_id?: string;                // Optional
  label_condition?: string;           // Optional: "real" or "duplicate"
  lighting_condition?: string;        // Optional: "bright", "normal", "dim", "low"
  device_manufacturer?: string;       // NEW: e.g., "Apple", "Samsung"
  device_model?: string;              // NEW: e.g., "iPhone 13", "Galaxy S21"
  device_os?: string;                 // NEW: e.g., "iOS 15.0", "Android 12"
  camera_megapixels?: number;         // NEW: e.g., 12.0
}
```

#### Response Includes Device Info

The API response now includes the device information if provided:

```json
{
  "status": "success",
  "authentic": true,
  "score": 0.85,
  "product_id": "PRODUCT1",
  "device_manufacturer": "Apple",
  "device_model": "iPhone 13 Pro",
  "device_os": "iOS 16.0",
  "camera_megapixels": 12.0,
  ...
}
```

### 2. Training Data Collector Updated (`services/training_data_collector.py`)

#### Updated Function Signature

```python
def append_to_training_csv(
    metrics: Dict[str, float], 
    lighting_condition: str,
    label: str,
    csv_path: str = "training_data/sample_data.csv",
    device_manufacturer: Optional[str] = None,      # NEW
    device_model: Optional[str] = None,            # NEW
    device_os: Optional[str] = None,               # NEW
    camera_megapixels: Optional[float] = None      # NEW
) -> bool:
```

### 3. CSV Structure Updated (`training_data/sample_data.csv`)

#### New Columns Added

The CSV now includes 4 additional columns (total 21 columns):

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `DeviceManufacturer` | string | Device manufacturer | "Apple", "Samsung", "Google" |
| `DeviceModel` | string | Device model | "iPhone 13 Pro", "Galaxy S21" |
| `DeviceOS` | string | Operating system | "iOS 16.0", "Android 12" |
| `CameraMegapixels` | float | Camera resolution | 12.0, 48.0, 64.0 |

#### Full CSV Structure (21 columns)

```csv
Sharpness,Contrast,HistogramPeak,EdgeDensity,EdgeStrength,NoiseLevel,
HighFreqEnergy,ColorDiversity,UniqueColors,Saturation,TextureUniformity,
CompressionArtifacts,HistogramEntropy,DynamicRange,Brightness,
LightingCondition,Label,
DeviceManufacturer,DeviceModel,DeviceOS,CameraMegapixels
```

### 4. Existing Data Migrated

- All 61 existing rows have been updated with empty values for the new device columns
- New verification requests will populate these fields when device info is provided
- Empty values are acceptable (backwards compatible)

## Usage Examples

### Frontend Integration (TypeScript/React)

```typescript
// Example: Sending verification request with device info
const payload = {
  cdp_image: base64Image,
  device_manufacturer: deviceInfo.manufacturer,  // "Apple"
  device_model: deviceInfo.model,                // "iPhone 13 Pro"
  device_os: deviceInfo.os,                      // "iOS 16.0"
  camera_megapixels: 12.0,                       // Camera resolution
  label_condition: "real",                       // Optional: for training
  lighting_condition: "bright"                   // Optional: for training
};

const response = await fetch('http://localhost:8000/verify_cdp', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
});

const result = await response.json();
console.log('Device info:', {
  manufacturer: result.device_manufacturer,
  model: result.device_model,
  os: result.device_os,
  camera: result.camera_megapixels
});
```

### Python/cURL Examples

```bash
# cURL example
curl -X POST http://localhost:8000/verify_cdp \
  -H "Content-Type: application/json" \
  -d '{
    "cdp_image": "<base64_encoded_image>",
    "device_manufacturer": "Apple",
    "device_model": "iPhone 13 Pro",
    "device_os": "iOS 16.0",
    "camera_megapixels": 12.0,
    "label_condition": "real",
    "lighting_condition": "bright"
  }'
```

```python
# Python example
import requests
import base64

with open('test_image.png', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

payload = {
    'cdp_image': image_base64,
    'device_manufacturer': 'Samsung',
    'device_model': 'Galaxy S21',
    'device_os': 'Android 12',
    'camera_megapixels': 64.0
}

response = requests.post(
    'http://localhost:8000/verify_cdp',
    json=payload
)

print(response.json())
```

## Benefits

### 1. Enhanced ML Training Data
- **Device-specific patterns**: Identify if certain devices produce better/worse scans
- **Camera quality correlation**: Analyze relationship between camera resolution and scan quality
- **OS-specific behavior**: Detect differences in image processing across platforms

### 2. Analytics & Insights
- **Popular devices**: Track which devices are most commonly used for verification
- **Quality by device**: Compare verification success rates across devices
- **Camera performance**: Correlate camera megapixels with scan quality metrics

### 3. Debugging & Support
- **Issue reproduction**: Replicate issues specific to device/OS combinations
- **Targeted optimization**: Optimize for commonly used devices
- **User support**: Provide device-specific guidance

### 4. Future ML Models
- **Device-aware models**: Train models that account for device characteristics
- **Quality prediction**: Predict scan quality based on device specs
- **Recommendations**: Suggest optimal scanning settings per device

## Data Privacy Considerations

- Device information is **optional** and stored for ML training purposes
- No personally identifiable information (PII) is collected
- Device data helps improve the anti-counterfeit system for all users
- Aggregated analytics only (no individual user tracking)

## Backwards Compatibility

✅ **Fully backwards compatible**:
- Old clients (without device info) continue to work
- New fields are optional
- Empty values are acceptable in CSV
- Existing training data remains valid

## Testing

### CSV Structure Test
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('training_data/sample_data.csv')
print(f'Total columns: {len(df.columns)}')
print(f'Total rows: {len(df)}')
print('Device columns:', [c for c in df.columns if 'Device' in c or 'Camera' in c])
"
```

### API Test (requires Flask app running)
```bash
# Test with device info
curl -X POST http://localhost:8000/verify_cdp \
  -H "Content-Type: application/json" \
  -d '{
    "cdp_image": "...",
    "device_manufacturer": "Apple",
    "device_model": "iPhone 13",
    "device_os": "iOS 16.0",
    "camera_megapixels": 12.0
  }'
```

## Files Modified

| File | Changes |
|------|---------|
| `app.py` | Added device info extraction and passing to training collector |
| `services/training_data_collector.py` | Added device info parameters and CSV columns |
| `training_data/sample_data.csv` | Added 4 new columns (DeviceManufacturer, DeviceModel, DeviceOS, CameraMegapixels) |

## Migration Notes

- ✅ Existing CSV data automatically migrated with empty device fields
- ✅ No breaking changes to API
- ✅ All existing functionality preserved
- ✅ New fields are optional and backwards compatible

## Next Steps

1. **Update frontend** to collect and send device information
2. **Monitor data** to see device distribution
3. **Analyze patterns** to identify device-specific issues
4. **Train models** using device information as features
5. **Optimize** for commonly used devices

## Example Data After Update

```csv
Sharpness,Contrast,...,Brightness,LightingCondition,Label,DeviceManufacturer,DeviceModel,DeviceOS,CameraMegapixels
59.86,29.34,...,213.43,avg,fake,,,,
86.41,29.93,...,207.98,avg,fake,,,,
73.36,24.03,...,192.95,bright,real,Apple,iPhone 13 Pro,iOS 16.0,12.0
54.13,21.61,...,188.42,bright,real,Samsung,Galaxy S21,Android 12,64.0
```

## Summary of Benefits

✅ **Enhanced ML training** with device context  
✅ **Better analytics** on device usage patterns  
✅ **Improved debugging** for device-specific issues  
✅ **Backwards compatible** with existing systems  
✅ **Privacy-friendly** (no PII collected)  
✅ **Future-ready** for device-aware ML models  

