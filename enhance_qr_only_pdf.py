#!/usr/bin/env python3
"""
Script to enhance subtle patterns ONLY in the QR code white space (left half),
while preserving the CDP portion (right half) unchanged.
This makes the security features visible in PDFs without affecting the CDP.
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

def extract_qr_region(image_array, border_thickness=10):
    """
    Extract QR code region (left half) from the combined image.
    Assumes structure: [Yellow Border][QR Code][Padding][CDP][Yellow Border]
    """
    h, w = image_array.shape[:2]
    
    # Remove yellow border (first and last border_thickness pixels)
    if len(image_array.shape) == 3:
        cropped = image_array[border_thickness:-border_thickness, border_thickness:-border_thickness]
    else:
        cropped = image_array[border_thickness:-border_thickness, border_thickness:-border_thickness]
    
    # QR code is the left half (before midpoint)
    cropped_h, cropped_w = cropped.shape[:2]
    midpoint = cropped_w // 2
    
    # Extract QR region (left half)
    if len(cropped.shape) == 3:
        qr_region = cropped[:, :midpoint]
    else:
        qr_region = cropped[:, :midpoint]
    
    return qr_region, midpoint

def enhance_qr_white_space_only(qr_region):
    """
    Enhance subtle patterns ONLY in QR code white areas.
    Makes security features more visible without affecting QR readability.
    """
    if len(qr_region.shape) == 2:
        # Grayscale - convert to BGR
        qr_bgr = cv2.cvtColor(qr_region, cv2.COLOR_GRAY2BGR)
    else:
        # Already color - assume RGB, convert to BGR
        qr_bgr = cv2.cvtColor(qr_region, cv2.COLOR_RGB2BGR)
    
    h, w = qr_bgr.shape[:2]
    enhanced = qr_bgr.copy()
    
    # Identify white areas in QR code
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    white_mask = gray > 200  # White areas (QR code white modules)
    
    # 1. Enhance moiré patterns in white space (make horizontal lines more visible)
    # Original: -5 intensity, Enhanced: -20 intensity for better visibility
    for i in range(0, h, 5):
        if i % 10 < 3:
            mask = white_mask[i, :]
            enhanced[i, mask] = np.clip(enhanced[i, mask].astype(np.int16) - 20, 0, 255).astype(np.uint8)
    
    # 2. Enhance frequency interference patterns (make wave patterns more visible)
    # Original: *3 amplitude, Enhanced: *10 amplitude
    y, x = np.ogrid[:h, :w]
    freq_pattern = np.sin(x * 0.15) * np.cos(y * 0.15) * 10  # Increased from 3 to 10
    
    # Apply to white areas with more intensity
    for c in range(3):
        channel_white = white_mask
        enhanced[:, :, c][channel_white] = np.clip(
            enhanced[:, :, c][channel_white].astype(np.int16) + 
            freq_pattern[channel_white], 0, 255
        ).astype(np.uint8)
    
    # 3. Add visible micro-dots in white space (small dots that should be visible)
    # Use deterministic seed for consistency
    np.random.seed(42)
    # Add dark dots (2% of white pixels)
    dot_mask = (np.random.random((h, w)) > 0.98) & white_mask
    enhanced[dot_mask] = np.clip(enhanced[dot_mask].astype(np.int16) - 25, 0, 255).astype(np.uint8)
    
    # Add bright dots (0.5% of white pixels)
    bright_dot_mask = (np.random.random((h, w)) > 0.995) & white_mask
    enhanced[bright_dot_mask] = np.clip(enhanced[bright_dot_mask].astype(np.int16) + 20, 0, 255).astype(np.uint8)
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    return enhanced_rgb

def enhance_image_qr_only(image):
    """
    Enhance only the QR code portion (left half) while keeping CDP unchanged.
    """
    img_array = np.array(image)
    
    # Convert to RGB if needed
    if img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Extract QR region (left half)
    qr_region, midpoint = extract_qr_region(img_array, border_thickness=10)
    
    # Enhance only QR white space
    enhanced_qr = enhance_qr_white_space_only(qr_region)
    
    # Reconstruct full image: enhanced QR + original CDP
    h, w = img_array.shape[:2]
    border_thickness = 10
    
    # Create new image array
    result = img_array.copy()
    
    # Replace QR region with enhanced version
    # Account for border removal
    cropped_h = h - 2 * border_thickness
    cropped_w = w - 2 * border_thickness
    cropped_midpoint = cropped_w // 2
    
    # Place enhanced QR in left half
    result[border_thickness:border_thickness+cropped_h, 
           border_thickness:border_thickness+cropped_midpoint] = enhanced_qr
    
    return Image.fromarray(result)

def create_qr_enhanced_pdf():
    """
    Create a PDF with enhanced QR white space patterns for better visibility.
    """
    tiff_dir = os.path.join(os.path.dirname(__file__), "generated_qr_cdp")
    output_pdf = os.path.join(os.path.dirname(__file__), "generated_qr_cdp_catalog_qr_enhanced.pdf")
    
    # Find all .tiff files
    tiff_pattern = os.path.join(tiff_dir, "*.tiff")
    tiff_files = glob.glob(tiff_pattern)
    
    if not tiff_files:
        print(f"[ERROR] No .tiff files found in {tiff_dir}")
        return False
    
    # Sort files naturally
    tiff_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    print(f"[INFO] Found {len(tiff_files)} .tiff files")
    print(f"[INFO] Enhancing QR white space patterns only (CDP unchanged)...")
    print(f"[INFO] Creating PDF: {output_pdf}")
    
    images = []
    for tiff_file in tiff_files:
        try:
            print(f"[INFO] Processing: {os.path.basename(tiff_file)}")
            img = Image.open(tiff_file)
            
            # Convert CMYK to RGB if needed
            if img.mode == 'CMYK':
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
            
            # Enhance only QR white space (left half)
            enhanced_img = enhance_image_qr_only(img)
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
    print(f"[INFO] Saving PDF with {len(images)} pages...")
    
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
        
        print(f"[SUCCESS] QR-enhanced PDF created: {output_pdf}")
        print(f"[INFO] Total pages: {len(images)}")
        print(f"[INFO] Resolution: 2400 DPI")
        print(f"[INFO] Quality: Maximum")
        print(f"[INFO] File size: {file_size_mb:.2f} MB")
        print(f"[INFO] QR patterns enhanced: Moiré (-20), Frequency (*10), Micro-dots added")
        print(f"[INFO] CDP portion: Unchanged (original quality)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create PDF: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    create_qr_enhanced_pdf()

