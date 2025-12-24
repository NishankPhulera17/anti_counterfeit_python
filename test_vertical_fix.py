#!/usr/bin/env python3
"""
Test script to verify the vertical image handling fix in cdp_service.py.
This script loads the problematic cropped image and runs extract_cdp_region on it.
"""
import cv2
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.cdp_service import extract_cdp_region

def test_vertical_image():
    """Test that vertical images are handled correctly."""
    # Load the problematic cropped image
    cropped_path = "cropped_qr_cdp/cropped_qr_cdp_20251223_164504_903488.png"
    
    if not os.path.exists(cropped_path):
        print(f"[ERROR] Test image not found: {cropped_path}")
        return False
    
    img = cv2.imread(cropped_path)
    if img is None:
        print(f"[ERROR] Failed to load image: {cropped_path}")
        return False
    
    h, w = img.shape[:2]
    print(f"[TEST] Original image dimensions: {w}x{h}")
    print(f"[TEST] Image is {'vertical' if h > w else 'horizontal'}")
    
    # Run the extraction
    print("\n[TEST] Running extract_cdp_region...")
    cdp = extract_cdp_region(img, save_file=True, custom_filename="test_vertical_fix.png")
    
    if cdp is None or cdp.size == 0:
        print("[ERROR] CDP extraction returned empty result")
        return False
    
    cdp_h, cdp_w = cdp.shape[:2]
    print(f"\n[TEST] Extracted CDP dimensions: {cdp_w}x{cdp_h}")
    
    # Check that the extracted CDP has reasonable texture (not QR code)
    gray_cdp = cv2.cvtColor(cdp, cv2.COLOR_BGR2GRAY)
    cdp_variance = gray_cdp.var()
    print(f"[TEST] Extracted CDP variance: {cdp_variance:.2f}")
    
    # QR codes typically have very high variance (>3000), CDP has moderate variance (<3000)
    if cdp_variance > 3000:
        print("[WARNING] Extracted region has very high variance - might be QR code, not CDP!")
        return False
    else:
        print("[SUCCESS] Variance suggests this is CDP, not QR code.")
        return True

if __name__ == "__main__":
    success = test_vertical_image()
    print(f"\n[RESULT] Test {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)
