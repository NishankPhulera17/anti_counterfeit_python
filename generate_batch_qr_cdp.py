#!/usr/bin/env python3
"""
Script to generate multiple QR+CDP images in batch.
Generates QR+CDP for product IDs from product0 to product4 (5 products).
Generates 3 sizes: 28×14mm, 40×20mm, and 50×25mm, all at 2400 DPI.
"""

from generate_qr_cdp import generate_qr_cdp
import os

# Size configurations: (width_mm, height_mm, dpi, size_name)
SIZE_CONFIGS = [
    (28, 14, 2400, "28x14"),
    (40, 20, 2400, "40x20"),
    (50, 25, 2400, "50x25"),
]

def main():
    total_generated = 0
    
    # Generate for each size configuration
    for width_mm, height_mm, dpi, size_name in SIZE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"[INFO] Generating QR+CDP codes for size: {width_mm}×{height_mm}mm at {dpi} DPI")
        print(f"{'='*60}\n")
        
        # Generate 5 QR+CDP images for each size: product0 through product4
        for i in range(5):
            product_id = f"product{i}"
            print(f"[INFO] Generating {size_name} QR+CDP for {product_id}...")
            try:
                generate_qr_cdp(
                    qr_data=product_id,
                    qr_size=None,  # Auto-calculate based on physical size and DPI
                    cdp_size=None,  # Auto-calculate based on physical size and DPI
                    padding=20,
                    border_thickness=25,
                    physical_size_mm=(width_mm, height_mm),
                    dpi=dpi,
                    size_suffix=size_name  # Add size suffix to filename
                )
                print(f"[INFO] Successfully generated {size_name} QR+CDP for {product_id}")
                total_generated += 1
            except Exception as e:
                print(f"[ERROR] Failed to generate {size_name} QR+CDP for {product_id}: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"[INFO] Batch generation complete! Generated {total_generated} QR+CDP images.")
    print(f"[INFO] Breakdown: 5 products × 3 sizes = 15 total images")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

