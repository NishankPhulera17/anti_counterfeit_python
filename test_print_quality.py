#!/usr/bin/env python3
"""
Test script to verify print quality of generated QR codes.
This script helps you check if your printed QR codes meet the requirements.
"""

import cv2
import numpy as np
import os
from generate_qr_cdp import CDP_DIR, OUTPUT_DIR
from utils.image_utils import decode_qr_code, check_image_quality_for_qr
from services.liveness_service import detect_photocopy, check_blur, check_fft_screen

def analyze_print_quality(image_path):
    """
    Analyze a printed QR code image to check print quality.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {image_path}")
    print(f"{'='*60}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ ERROR: Could not load image from {image_path}")
        return False
    
    print(f"Image dimensions: {img.shape[1]}x{img.shape[0]} pixels")
    
    # 1. Check QR code readability
    print("\n1. QR Code Readability:")
    product_id = decode_qr_code(img, require_quality=False, save_folder="scanned_images")
    if product_id:
        print(f"   ✅ QR code readable: {product_id}")
    else:
        print(f"   ❌ QR code NOT readable")
    
    # 2. Check image quality
    print("\n2. Image Quality Checks:")
    quality_ok = check_image_quality_for_qr(img)
    if quality_ok:
        print("   ✅ Image quality: PASS")
    else:
        print("   ❌ Image quality: FAIL")
    
    # 3. Check sharpness (blur)
    print("\n3. Sharpness Check:")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"   Blur score: {blur_score:.2f}")
    if blur_score > 100:
        print("   ✅ Sharpness: GOOD")
    elif blur_score > 50:
        print("   ⚠️  Sharpness: ACCEPTABLE")
    else:
        print("   ❌ Sharpness: POOR (too blurry)")
    
    # 4. Check contrast
    print("\n4. Contrast Check:")
    contrast = np.std(gray)
    print(f"   Contrast (std dev): {contrast:.2f}")
    if contrast > 30:
        print("   ✅ Contrast: GOOD")
    else:
        print("   ❌ Contrast: POOR")
    
    # 5. Check for photocopy characteristics
    print("\n5. Photocopy Detection:")
    is_authentic = detect_photocopy(img)
    if is_authentic:
        print("   ✅ Appears to be authentic print (not photocopy)")
    else:
        print("   ⚠️  May be photocopy or low-quality reproduction")
    
    # 6. Check screen detection
    print("\n6. Screen Detection:")
    is_printed = check_fft_screen(img)
    if is_printed:
        print("   ✅ Appears to be printed material")
    else:
        print("   ⚠️  May be displayed on screen")
    
    # 7. Check red border
    print("\n7. Red Border Detection:")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Red wraps around 0/180 in HSV, so we need two ranges
    lower_red1 = np.array([0, 100, 100])    # Red near 0
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])  # Red near 180
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    red_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    red_percentage = (red_pixels / total_pixels) * 100
    print(f"   Red border coverage: {red_percentage:.2f}%")
    if red_percentage > 5:
        print("   ✅ Red border detected")
    else:
        print("   ❌ Red border NOT detected")
    
    # 8. Overall assessment
    print("\n" + "="*60)
    print("OVERALL ASSESSMENT:")
    print("="*60)
    
    checks_passed = sum([
        product_id is not None,
        quality_ok,
        blur_score > 50,
        contrast > 30,
        is_authentic,
        is_printed,
        red_percentage > 5
    ])
    
    total_checks = 7
    score = (checks_passed / total_checks) * 100
    
    print(f"Quality Score: {score:.1f}% ({checks_passed}/{total_checks} checks passed)")
    
    if score >= 85:
        print("✅ EXCELLENT: Print quality meets all requirements")
        return True
    elif score >= 70:
        print("⚠️  GOOD: Print quality is acceptable but could be improved")
        return True
    elif score >= 50:
        print("⚠️  FAIR: Print quality needs improvement")
        return False
    else:
        print("❌ POOR: Print quality does not meet requirements")
        return False


def main():
    """
    Main function to test print quality.
    """
    print("="*60)
    print("QR Code Print Quality Tester")
    print("="*60)
    print("\nThis script analyzes printed QR codes to verify print quality.")
    print("Place a photo/scan of your printed QR code in the test_images/ folder")
    print("or provide the path to an image file.\n")
    
    # Check for test images directory
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f"Created {test_dir}/ directory")
        print("Please place test images there and run this script again.\n")
        return
    
    # Look for images in test directory
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        import glob
        image_files.extend(glob.glob(os.path.join(test_dir, ext)))
        image_files.extend(glob.glob(os.path.join(test_dir, ext.upper())))
    
    if not image_files:
        print(f"No images found in {test_dir}/ directory")
        print("Please add test images and run again.\n")
        return
    
    # Analyze each image
    results = []
    for img_path in image_files:
        result = analyze_print_quality(img_path)
        results.append((img_path, result))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All prints meet quality requirements!")
    else:
        print("⚠️  Some prints need improvement. Review recommendations above.")


if __name__ == "__main__":
    main()

