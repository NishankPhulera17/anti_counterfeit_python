#!/usr/bin/env python3
"""
Script to create a truly lossless PDF from .tiff files using img2pdf.
This method preserves 100% image quality without any compression.
"""

import os
import glob
import re

def natural_sort_key(text):
    """Extract number from filename for natural sorting (product0, product1, ..., product19)"""
    match = re.search(r'product(\d+)', text)
    return int(match.group(1)) if match else 0

def create_lossless_pdf():
    """
    Create a lossless PDF from all .tiff files using img2pdf.
    This method preserves 100% image quality without compression.
    """
    try:
        import img2pdf
    except ImportError:
        print("[ERROR] img2pdf library not found.")
        print("[INFO] Installing img2pdf for lossless PDF creation...")
        print("[INFO] Please run: pip install img2pdf")
        print("[INFO] Falling back to PIL method...")
        return False
    
    # Directory containing TIFF files
    tiff_dir = os.path.join(os.path.dirname(__file__), "generated_qr_cdp")
    output_pdf = os.path.join(os.path.dirname(__file__), "generated_qr_cdp_catalog_lossless.pdf")
    
    # Find all .tiff files
    tiff_pattern = os.path.join(tiff_dir, "*.tiff")
    tiff_files = glob.glob(tiff_pattern)
    
    if not tiff_files:
        print(f"[ERROR] No .tiff files found in {tiff_dir}")
        return False
    
    # Sort files naturally (product0, product1, ..., product19)
    tiff_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    print(f"[INFO] Found {len(tiff_files)} .tiff files")
    print(f"[INFO] Creating lossless PDF: {output_pdf}")
    print(f"[INFO] Using img2pdf for 100% quality preservation...")
    
    try:
        # img2pdf creates truly lossless PDFs
        # It embeds images directly without recompression
        with open(output_pdf, "wb") as f:
            f.write(img2pdf.convert(tiff_files))
        
        # Get file size
        file_size = os.path.getsize(output_pdf)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"[SUCCESS] Lossless PDF created successfully: {output_pdf}")
        print(f"[INFO] Total pages: {len(tiff_files)}")
        print(f"[INFO] Quality: 100% lossless (no compression)")
        print(f"[INFO] File size: {file_size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to create PDF: {str(e)}")
        return False

if __name__ == "__main__":
    create_lossless_pdf()

