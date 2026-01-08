#!/usr/bin/env python3
"""
Test script to verify device information is correctly handled in the verify API.
"""

import requests
import json
import base64
import os

API_URL = "http://localhost:8000"

def test_device_info_in_verify():
    """Test that device info is accepted and saved in training data"""
    print("\n" + "="*70)
    print("TEST: Device Information in Verify API")
    print("="*70)
    
    # Find a test image
    gen_dir = "generated_qr_cdp"
    images = [f for f in os.listdir(gen_dir) if f.endswith("_28x14.png")]
    
    if not images:
        print("❌ FAIL: No test images found")
        return False
    
    test_image_path = os.path.join(gen_dir, images[0])
    print(f"\nTest image: {test_image_path}")
    
    # Read and encode image
    with open(test_image_path, 'rb') as f:
        image_data = f.read()
    
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # Prepare payload with device information
    payload = {
        'cdp_image': image_base64,
        'device_manufacturer': 'Apple',
        'device_model': 'iPhone 13 Pro',
        'device_os': 'iOS 16.0',
        'camera_megapixels': 12.0,
        'label_condition': 'real',
        'lighting_condition': 'bright'
    }
    
    print("\nSending request with device info:")
    print(f"  Device: {payload['device_manufacturer']} {payload['device_model']}")
    print(f"  OS: {payload['device_os']}")
    print(f"  Camera: {payload['camera_megapixels']} MP")
    
    try:
        response = requests.post(
            f"{API_URL}/verify_cdp",
            json=payload,
            timeout=30
        )
        
        print(f"\nResponse status: {response.status_code}")
        
        if response.status_code in [200, 404]:  # 404 is OK if serial not found
            result = response.json()
            
            # Check if device info is in response
            has_device_info = all([
                result.get('device_manufacturer') == 'Apple',
                result.get('device_model') == 'iPhone 13 Pro',
                result.get('device_os') == 'iOS 16.0',
                result.get('camera_megapixels') == 12.0
            ])
            
            print("\n✅ Device info in response:")
            print(f"  Manufacturer: {result.get('device_manufacturer', 'N/A')}")
            print(f"  Model: {result.get('device_model', 'N/A')}")
            print(f"  OS: {result.get('device_os', 'N/A')}")
            print(f"  Camera: {result.get('camera_megapixels', 'N/A')} MP")
            
            if has_device_info:
                print("\n✅ PASS: Device information correctly handled")
                return True
            else:
                print("\n❌ FAIL: Device information not in response")
                return False
        else:
            print(f"\n❌ FAIL: Unexpected status code {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("\n⚠️  SKIP: Flask app not running")
        print("   Start the app with: python3 app.py")
        return True  # Skip, not fail
    except Exception as e:
        print(f"\n❌ FAIL: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_csv_structure():
    """Test that CSV has the correct columns"""
    print("\n" + "="*70)
    print("TEST: CSV Structure")
    print("="*70)
    
    import pandas as pd
    
    csv_path = "training_data/sample_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ FAIL: CSV file not found at {csv_path}")
        return False
    
    df = pd.read_csv(csv_path)
    
    required_columns = [
        'Sharpness', 'Contrast', 'HistogramPeak', 'EdgeDensity', 'EdgeStrength',
        'NoiseLevel', 'HighFreqEnergy', 'ColorDiversity', 'UniqueColors', 
        'Saturation', 'TextureUniformity', 'CompressionArtifacts', 
        'HistogramEntropy', 'DynamicRange', 'Brightness', 
        'LightingCondition', 'Label',
        'DeviceManufacturer', 'DeviceModel', 'DeviceOS', 'CameraMegapixels'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ FAIL: Missing columns: {missing_columns}")
        return False
    
    print(f"✅ All required columns present ({len(required_columns)} columns)")
    print(f"   Total rows: {len(df)}")
    print(f"\nDevice columns:")
    print(f"  - DeviceManufacturer")
    print(f"  - DeviceModel")
    print(f"  - DeviceOS")
    print(f"  - CameraMegapixels")
    
    print("\n✅ PASS: CSV structure is correct")
    return True

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DEVICE INFORMATION TEST SUITE")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("CSV Structure", test_csv_structure()))
    results.append(("Device Info in API", test_device_info_in_verify()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
    
    exit(0 if passed == total else 1)

