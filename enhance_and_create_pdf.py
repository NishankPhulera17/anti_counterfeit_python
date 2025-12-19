#!/usr/bin/env python3
"""
Script to enhance subtle patterns in QR codes (especially dots in white space)
and then create a high-quality PDF that preserves these enhanced details.
This makes the security features more visible in the PDF output.
"""

import os
import glob
import re
import numpy as np
from PIL import Image
import cv2

def natural_sort_key(text):
    """Extract number from filename for natural sorting"""
    match = re.search(r'product(\d+)', text)
    return int(match.group(1)) if match else 0

def enhance_qr_patterns(image):
    """
    Enhance subtle patterns in QR code white areas to make them more visible.
    Specifically enhances:
    - Moiré patterns
    - Frequency interference patterns
    - Small dots/security marks
    """
    img_array = np.array(image)
    
    # Convert to BGR if needed (PIL uses RGB, OpenCV uses BGR)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        # Assume RGB from PIL
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    h, w = img_bgr.shape[:2]
    enhanced = img_bgr.copy()
    
    # Convert to grayscale to identify white areas
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    white_mask = gray > 200  # White areas (more lenient threshold)
    
    # 1. Enhance moiré patterns (make them more visible)
    # Original: -5 intensity, Enhanced: -15 intensity
    for i in range(0, h, 5):
        if i % 10 < 3:
            mask = white_mask[i, :]
            enhanced[i, mask] = np.clip(enhanced[i, mask].astype(np.int16) - 15, 0, 255).astype(np.uint8)
    
    # 2. Enhance frequency interference patterns
    # Original: *3 amplitude, Enhanced: *8 amplitude
    y, x = np.ogrid[:h, :w]
    freq_pattern = np.sin(x * 0.15) * np.cos(y * 0.15) * 8  # Increased from 3 to 8
    
    # Apply to white areas with more intensity
    for c in range(3):
        channel_white = white_mask
        enhanced[:, :, c][channel_white] = np.clip(
            enhanced[:, :, c][channel_white].astype(np.int16) + 
            freq_pattern[channel_white], 0, 255
        ).astype(np.uint8)
    
    # 3. Add more visible micro-dots in white areas
    # Add small random dots to white space
    np.random.seed(42)  # Deterministic but visible
    dot_mask = (np.random.random((h, w)) > 0.98) & white_mask  # 2% of white pixels
    enhanced[dot_mask] = np.clip(enhanced[dot_mask].astype(np.int16) - 20, 0, 255).astype(np.uint8)
    
    # Add bright dots too
    bright_dot_mask = (np.random.random((h, w)) > 0.995) & white_mask  # 0.5% of white pixels
    enhanced[bright_dot_mask] = np.clip(enhanced[bright_dot_mask].astype(np.int16) + 15, 0, 255).astype(np.uint8)
    
    # Convert back to RGB for PIL
    if len(enhanced.shape) == 3:
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    else:
        enhanced_rgb = enhanced
    
    return Image.fromarray(enhanced_rgb)

def create_enhanced_pdf():
    """
    Create a PDF with enhanced QR patterns for better visibility.
    """
    # Directory containing TIFF files
    tiff_dir = os.path.join(os.path.dirname(__file__), "generated_qr_cdp")
    output_pdf = os.path.join(os.path.dirname(__file__), "generated_qr_cdp_catalog_enhanced.pdf")
    
    # Find all .tiff files
    tiff_pattern = os.path.join(tiff_dir, "*.tiff")
    tiff_files = glob.glob(tiff_pattern)
    
    if not tiff_files:
        print(f"[ERROR] No .tiff files found in {tiff_dir}")
        return False
    
    # Sort files naturally
    tiff_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    print(f"[INFO] Found {len(tiff_files)} .tiff files")
    print(f"[INFO] Enhancing QR patterns for better visibility...")
    print(f"[INFO] Creating enhanced PDF: {output_pdf}")
    
    images = []
    for tiff_file in tiff_files:
        try:
            print(f"[INFO] Processing: {os.path.basename(tiff_file)}")
            img = Image.open(tiff_file)
            
            # Convert CMYK to RGB if needed
            if img.mode == 'CMYK':
                # Convert CMYK to RGB carefully
                cmyk_array = np.array(img)
                c, m, y, k = cmyk_array[:, :, 0], cmyk_array[:, :, 1], cmyk_array[:, :, 2], cmyk_array[:, :, 3]
                
                # CMYK to RGB conversion
                r = 255 * (1 - c/255) * (1 - k/255)
                g = 255 * (1 - m/255) * (1 - k/255)
                b = 255 * (1 - y/255) * (1 - k/255)
                
                rgb_array = np.stack([r, g, b], axis=2).astype(np.uint8)
                img = Image.fromarray(rgb_array, mode='RGB')
            elif img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Enhance QR patterns (especially white space patterns)
            enhanced_img = enhance_qr_patterns(img)
            images.append(enhanced_img)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {tiff_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if not images:
        print("[ERROR] No images could be processed")
        return False
    
    # Save as PDF with maximum quality
    print(f"[INFO] Saving enhanced PDF with {len(images)} pages...")
    
    try:
        images[0].save(
            output_pdf,
            "PDF",
            resolution=2400.0,
            quality=100,
            optimize=False,
            save_all=True,
            append_images=images[1:] if len(images) > 1 else []
        )
        
        file_size = os.path.getsize(output_pdf)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"[SUCCESS] Enhanced PDF created: {output_pdf}")
        print(f"[INFO] Total pages: {len(images)}")
        print(f"[INFO] Resolution: 2400 DPI")
        print(f"[INFO] Quality: Maximum")
        print(f"[INFO] File size: {file_size_mb:.2f} MB")
        print(f"[INFO] Patterns enhanced: Moiré (-15), Frequency (*8), Micro-dots added")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_enhanced_pdf()

