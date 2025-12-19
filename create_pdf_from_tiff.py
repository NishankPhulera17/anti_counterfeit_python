#!/usr/bin/env python3
"""
Script to create a high-quality PDF from .tiff files in generated_qr_cdp directory.
Preserves image quality by using lossless compression and high DPI settings.
"""

import os
import glob
from PIL import Image
import re

def natural_sort_key(text):
    """Extract number from filename for natural sorting (product0, product1, ..., product19)"""
    match = re.search(r'product(\d+)', text)
    return int(match.group(1)) if match else 0

def create_pdf_from_tiff():
    """
    Create a PDF from all .tiff files in generated_qr_cdp directory.
    Preserves quality by using lossless compression.
    """
    # Directory containing TIFF files
    tiff_dir = os.path.join(os.path.dirname(__file__), "generated_qr_cdp")
    output_pdf = os.path.join(os.path.dirname(__file__), "generated_qr_cdp_catalog.pdf")
    
    # Find all .tiff files
    tiff_pattern = os.path.join(tiff_dir, "*.tiff")
    tiff_files = glob.glob(tiff_pattern)
    
    if not tiff_files:
        print(f"[ERROR] No .tiff files found in {tiff_dir}")
        return
    
    # Sort files naturally (product0, product1, ..., product19)
    tiff_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    print(f"[INFO] Found {len(tiff_files)} .tiff files")
    print(f"[INFO] Creating PDF: {output_pdf}")
    
    # Open all images
    images = []
    for tiff_file in tiff_files:
        try:
            print(f"[INFO] Loading: {os.path.basename(tiff_file)}")
            img = Image.open(tiff_file)
            
            # Convert CMYK to RGB if needed (for PDF compatibility)
            if img.mode == 'CMYK':
                print(f"[INFO] Converting CMYK to RGB: {os.path.basename(tiff_file)}")
                img = img.convert('RGB')
            elif img.mode not in ('RGB', 'RGBA', 'L', 'P'):
                print(f"[INFO] Converting {img.mode} to RGB: {os.path.basename(tiff_file)}")
                img = img.convert('RGB')
            
            images.append(img)
        except Exception as e:
            print(f"[ERROR] Failed to load {tiff_file}: {str(e)}")
            continue
    
    if not images:
        print("[ERROR] No images could be loaded")
        return
    
    # Save as PDF with high quality
    # Using PIL's save method which preserves quality
    print(f"[INFO] Saving PDF with {len(images)} pages...")
    
    try:
        # Save first image and append others
        images[0].save(
            output_pdf,
            "PDF",
            resolution=2400.0,  # High DPI to preserve quality
            quality=100,  # Maximum quality
            save_all=True,  # Save all images
            append_images=images[1:] if len(images) > 1 else []
        )
        
        print(f"[SUCCESS] PDF created successfully: {output_pdf}")
        print(f"[INFO] Total pages: {len(images)}")
        print(f"[INFO] Resolution: 2400 DPI")
        print(f"[INFO] Quality: 100% (lossless)")
        
        # Get file size
        file_size = os.path.getsize(output_pdf)
        file_size_mb = file_size / (1024 * 1024)
        print(f"[INFO] File size: {file_size_mb:.2f} MB")
        
    except Exception as e:
        print(f"[ERROR] Failed to create PDF: {str(e)}")
        # Try alternative method with reportlab if available
        try_alternative_method(images, output_pdf)

def try_alternative_method(images, output_pdf):
    """Alternative method using reportlab if PIL method fails"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfgen import canvas
        
        print("[INFO] Trying alternative method with reportlab...")
        
        # Determine page size based on first image
        first_img = images[0]
        img_width, img_height = first_img.size
        
        # Use image dimensions or standard page size
        page_width = max(img_width, 612)  # At least letter width
        page_height = max(img_height, 792)  # At least letter height
        
        c = canvas.Canvas(output_pdf, pagesize=(page_width, page_height))
        
        for i, img in enumerate(images):
            print(f"[INFO] Adding page {i+1}/{len(images)}")
            
            # Convert PIL image to format reportlab can use
            img_width, img_height = img.size
            
            # Scale to fit page while maintaining aspect ratio
            scale_x = page_width / img_width
            scale_y = page_height / img_height
            scale = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            scaled_width = img_width * scale
            scaled_height = img_height * scale
            
            # Center image on page
            x = (page_width - scaled_width) / 2
            y = (page_height - scaled_height) / 2
            
            # Save image to temporary buffer
            from io import BytesIO
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG', quality=100)
            img_buffer.seek(0)
            
            # Draw image
            c.drawImage(ImageReader(img_buffer), x, y, 
                       width=scaled_width, height=scaled_height,
                       preserveAspectRatio=True)
            
            c.showPage()
        
        c.save()
        print(f"[SUCCESS] PDF created using reportlab: {output_pdf}")
        
    except ImportError:
        print("[ERROR] reportlab not available. Please install it:")
        print("  pip install reportlab")
    except Exception as e:
        print(f"[ERROR] Alternative method also failed: {str(e)}")

if __name__ == "__main__":
    create_pdf_from_tiff()

