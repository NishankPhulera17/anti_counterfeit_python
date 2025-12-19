import os
import qrcode
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO
import hashlib
import uuid
import secrets
from typing import Tuple

# Directory to store generated images
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_qr_cdp")
CDP_DIR = os.path.join(os.path.dirname(__file__), "generated_cdp")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CDP_DIR, exist_ok=True)


def bgr_to_cmyk(bgr_image: np.ndarray) -> np.ndarray:
    """
    Convert BGR (OpenCV) image to CMYK format.
    CMYK values are in range 0-255.
    """
    # Convert BGR to RGB first
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    # Normalize to 0-1 range
    rgb_normalized = rgb_image.astype(np.float32) / 255.0
    
    # Convert RGB to CMYK
    # K (black) is the minimum of the inverted RGB values
    k = 1.0 - np.max(rgb_normalized, axis=2)
    
    # Avoid division by zero
    k_mask = k < 1.0
    c = np.zeros_like(k)
    m = np.zeros_like(k)
    y = np.zeros_like(k)
    
    # Calculate CMY when K < 1
    c[k_mask] = (1.0 - rgb_normalized[:, :, 0][k_mask] - k[k_mask]) / (1.0 - k[k_mask])
    m[k_mask] = (1.0 - rgb_normalized[:, :, 1][k_mask] - k[k_mask]) / (1.0 - k[k_mask])
    y[k_mask] = (1.0 - rgb_normalized[:, :, 2][k_mask] - k[k_mask]) / (1.0 - k[k_mask])
    
    # Stack into CMYK array and convert back to 0-255 range
    cmyk = np.stack([c, m, y, k], axis=2)
    cmyk = (cmyk * 255.0).astype(np.uint8)
    
    return cmyk


def save_cmyk_image(image: np.ndarray, file_path: str, save_rgb_fallback=True, dpi: int = 1200):
    """
    Save image in CMYK format as TIFF (primary format for printing).
    PNG doesn't support CMYK, so TIFF is used for CMYK output.
    Input image should be in BGR format (OpenCV standard).
    
    Args:
        image: BGR image as numpy array
        file_path: Base file path (will be converted to .tiff for CMYK)
        save_rgb_fallback: If True, also save RGB PNG for compatibility
        dpi: DPI resolution to embed in image metadata (default: 1200 DPI)
    """
    # Convert BGR to CMYK
    cmyk_image = bgr_to_cmyk(image)
    
    # Convert numpy array to PIL Image in CMYK mode
    # PIL expects CMYK in shape (height, width, 4)
    pil_image = Image.fromarray(cmyk_image, mode='CMYK')
    
    # Save as TIFF in CMYK format (primary format for printing)
    # Use .tiff extension for CMYK files
    if file_path.endswith('.png'):
        tiff_path = file_path.replace('.png', '.tiff')
    else:
        tiff_path = file_path + '.tiff'
    
    # Save TIFF with DPI metadata - PIL will use the dpi parameter
    pil_image.save(tiff_path, format='TIFF', compression='tiff_lzw', dpi=(dpi, dpi))
    print(f"[INFO] Saved CMYK image (primary) at {dpi} DPI: {tiff_path}")
    
    # Also save as PNG in RGB for compatibility (if requested)
    if save_rgb_fallback:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_rgb = Image.fromarray(rgb_image, mode='RGB')
        if not file_path.endswith('.png'):
            file_path = file_path + '.png'
        # Save PNG with DPI metadata
        pil_rgb.save(file_path, format='PNG', dpi=(dpi, dpi))
        print(f"[INFO] Saved RGB image (compatibility) at {dpi} DPI: {file_path}")
    
    return tiff_path


def calculate_optimal_pattern_params(image_size_pixels: tuple, physical_size_mm: tuple = (28, 14), dpi: int = 1200) -> dict:
    """
    Calculate optimal pattern parameters for small QR codes (28×14mm).
    Adjusts pattern frequencies and densities based on physical dimensions and DPI.
    
    Args:
        image_size_pixels: (width, height) in pixels
        physical_size_mm: (width, height) in millimeters (default: 28×14mm)
        dpi: Printing resolution (default: 1200 DPI)
    
    Returns:
        Dictionary with optimized pattern parameters
    """
    # Calculate pixels per millimeter
    pixels_per_mm = dpi / 25.4  # 25.4 mm per inch
    
    # Calculate physical dimensions in pixels
    physical_w_px = physical_size_mm[0] * pixels_per_mm
    physical_h_px = physical_size_mm[1] * pixels_per_mm
    
    # For small codes (28×14mm), we need patterns that:
    # 1. Are visible at print size
    # 2. Break when photocopied
    # 3. Work at high DPI (1200+)
    
    # Pattern frequencies (in pixels) - optimized for small codes
    # These frequencies create interference with common scanner/copier frequencies
    # Scanner frequencies: typically 300-600 DPI (11.8-23.6 pixels/mm)
    # At 1200 DPI: 47.2 pixels/mm
    
    # Moiré pattern spacing: 0.1-0.2mm physical spacing works well
    moire_spacing_px = 0.15 * pixels_per_mm  # ~0.15mm spacing
    
    # Micro-dot size: increased for better visibility (0.35mm physical size)
    # Increased from 0.25mm to make dots even more visible
    microdot_size_px = max(1, int(0.35 * pixels_per_mm))  # ~0.35mm (increased for better visibility)
    
    # Line grid spacing: 0.2-0.3mm for small codes
    line_spacing_px = max(2, int(0.25 * pixels_per_mm))  # ~0.25mm
    
    # Frequency interference periods (in pixels)
    # Target frequencies that interfere with 300-600 DPI scanners
    freq_periods = [
        2.0,  # High frequency (interferes with 600 DPI scanners)
        2.5,  # Medium-high
        3.0,  # Medium (interferes with 300 DPI scanners)
        3.5,  # Medium-low
        4.0   # Lower frequency
    ]
    
    # Gradient frequencies (adjusted for small size)
    # Lower frequencies for small codes to ensure visibility
    gradient_freq_x = 0.12  # Slightly higher than default for small codes
    gradient_freq_y = 0.12
    
    # Guilloche pattern scale (adjusted for small codes)
    guilloche_scale = 1.2  # Slightly increased for better visibility
    
    return {
        'moire_spacing': moire_spacing_px,
        'microdot_size': microdot_size_px,
        'line_spacing': line_spacing_px,
        'freq_periods': freq_periods,
        'gradient_freq_x': gradient_freq_x,
        'gradient_freq_y': gradient_freq_y,
        'guilloche_scale': guilloche_scale,
        'is_small_code': physical_size_mm[0] < 20 or physical_size_mm[1] < 20
    }


def add_frequency_interference_pattern(image: np.ndarray, pattern_params: dict = None) -> np.ndarray:
    """
    Add frequency-based interference patterns that exploit scanner/copier limitations.
    These patterns create aliasing and moiré effects when scanned or photocopied.
    Optimized for small codes (28×14mm) at high DPI.
    """
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    # Use optimized parameters if provided, otherwise use defaults
    if pattern_params is None:
        pattern_params = calculate_optimal_pattern_params((w, h))
    
    freq_periods = pattern_params.get('freq_periods', [2.0, 2.5, 3.0, 3.5, 4.0])
    
    y, x = np.ogrid[:h, :w]
    
    # Pattern 1: High-frequency lines (exploits scanner sampling limitations)
    freq1 = 2 * np.pi * x / freq_periods[0]
    pattern1 = np.sin(freq1) * 12
    
    # Pattern 2: Diagonal interference (creates moiré with scanner grid)
    freq2 = 2 * np.pi * (x + y) / freq_periods[2]
    pattern2 = np.sin(freq2) * 10
    
    # Pattern 3: Circular frequency interference
    center_x, center_y = w / 2, h / 2
    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    freq3 = 2 * np.pi * radius / freq_periods[3]
    pattern3 = np.sin(freq3) * 8
    
    # Combine patterns
    combined_pattern = pattern1 + pattern2 * 0.7 + pattern3 * 0.5
    
    # Apply to each color channel with slight phase shifts
    for c in range(3):
        phase_shift = c * np.pi / 6
        channel_pattern = combined_pattern * np.cos(phase_shift)
        result[:, :, c] = np.clip(result[:, :, c] + channel_pattern, 0, 255)
    
    return result.astype(np.uint8)


def add_color_shift_security(image: np.ndarray, cdp_id: str) -> np.ndarray:
    """
    Add color-shift security patterns using CMYK-specific color combinations.
    These colors don't reproduce accurately when photocopied or scanned.
    """
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    # Use cdp_id for pattern generation (deterministic within this CDP)
    seed = int(hashlib.md5(cdp_id.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    
    y, x = np.ogrid[:h, :w]
    
    # Create regions with specific color combinations that fail in photocopying
    # Pattern 1: Cyan-Magenta gradient (CMYK colors that shift when copied)
    cm_pattern = np.sin(x * 0.08) * np.cos(y * 0.08)
    
    # Pattern 2: Yellow-Cyan combination (hard to reproduce accurately)
    yc_pattern = np.cos(x * 0.1) * np.sin(y * 0.1)
    
    # Apply color shifts to specific channels
    # Cyan shift (affects B channel in BGR)
    result[:, :, 0] = np.clip(result[:, :, 0] + cm_pattern * 15, 0, 255)
    # Magenta shift (affects G channel in BGR)
    result[:, :, 1] = np.clip(result[:, :, 1] + yc_pattern * 12, 0, 255)
    # Yellow shift (affects R channel in BGR)
    result[:, :, 2] = np.clip(result[:, :, 2] + (cm_pattern + yc_pattern) * 8, 0, 255)
    
    return result.astype(np.uint8)


def add_embedded_security_marks(image: np.ndarray, cdp_id: str, pattern_params: dict = None) -> np.ndarray:
    """
    Add embedded security marks that become visible or distorted when photocopied.
    These are subtle patterns that break during the copying process.
    Optimized for small codes (28×14mm).
    """
    h, w = image.shape[:2]
    result = image.copy()
    
    # Use optimized parameters if provided
    if pattern_params is None:
        pattern_params = calculate_optimal_pattern_params((w, h))
    
    line_spacing = pattern_params.get('line_spacing', 8)
    
    # Use cdp_id for deterministic pattern generation
    seed = int(hashlib.md5(cdp_id.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    
    # Create embedded watermark pattern (barely visible but breaks when copied)
    y, x = np.ogrid[:h, :w]
    
    # Pattern 1: Fine line grid (becomes distorted when copied)
    # Use optimized spacing for small codes
    line_mask = (x % line_spacing < 1) | (y % line_spacing < 1)
    result[line_mask] = np.clip(result[line_mask].astype(np.int16) - 8, 0, 255).astype(np.uint8)
    
    # Pattern 2: Embedded text pattern (cdp_id hash as invisible watermark)
    hash_val = hashlib.md5(cdp_id.encode()).hexdigest()
    # Create a subtle pattern based on hash
    for i, char in enumerate(hash_val[:16]):
        val = int(char, 16)
        x_pos = (i * w // 16) % w
        y_pos = (val * h // 16) % h
        # Add subtle mark
        if 0 <= y_pos < h and 0 <= x_pos < w:
            result[y_pos, x_pos] = np.clip(result[y_pos, x_pos].astype(np.int16) - 5, 0, 255).astype(np.uint8)
    
    # Pattern 3: Corner security marks (become obvious when copied)
    # For small codes, use smaller but more visible marks
    mark_size = max(3, min(h, w) // 12)  # Smaller marks for small codes
    # Top-left corner
    result[:mark_size, :mark_size] = np.clip(
        result[:mark_size, :mark_size].astype(np.int16) + 
        np.random.randint(-10, 10, (mark_size, mark_size, 3), dtype=np.int16), 0, 255
    ).astype(np.uint8)
    
    return result


def add_guilloche_pattern(image: np.ndarray, cdp_id: str, pattern_params: dict = None) -> np.ndarray:
    """
    Add Guilloche patterns (complex curved patterns) that are extremely difficult to reproduce.
    These are used in currency and official documents for anti-counterfeiting.
    Optimized for small codes (28×14mm).
    """
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    # Use optimized parameters if provided
    if pattern_params is None:
        pattern_params = calculate_optimal_pattern_params((w, h))
    
    guilloche_scale = pattern_params.get('guilloche_scale', 1.0)
    
    # Use cdp_id for deterministic pattern generation
    seed = int(hashlib.md5(cdp_id.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    
    y, x = np.ogrid[:h, :w]
    center_x, center_y = w / 2, h / 2
    
    # Normalize coordinates
    nx = (x - center_x) / max(w, h) * 2
    ny = (y - center_y) / max(w, h) * 2
    
    # Guilloche pattern 1: Concentric circles with phase modulation
    radius = np.sqrt(nx**2 + ny**2)
    angle = np.arctan2(ny, nx)
    
    # Complex sinusoidal modulation (scaled for small codes)
    pattern1 = np.sin(radius * 8 * guilloche_scale + angle * 3) * np.cos(radius * 5 * guilloche_scale - angle * 2)
    
    # Guilloche pattern 2: Lissajous curves (scaled)
    pattern2 = np.sin(nx * 6 * guilloche_scale + ny * 4 * guilloche_scale) * np.cos(nx * 4 * guilloche_scale - ny * 6 * guilloche_scale)
    
    # Guilloche pattern 3: Spiral interference (scaled)
    pattern3 = np.sin(radius * 10 * guilloche_scale + angle * 5) * np.cos(angle * 8 * guilloche_scale)
    
    # Combine patterns
    combined = (pattern1 * 0.4 + pattern2 * 0.35 + pattern3 * 0.25) * 18
    
    # Apply with color channel variations
    for c in range(3):
        phase = c * np.pi / 4
        channel_pattern = combined * np.cos(phase)
        result[:, :, c] = np.clip(result[:, :, c] + channel_pattern, 0, 255)
    
    return result.astype(np.uint8)


def add_screen_frequency_interference(image: np.ndarray, pattern_params: dict = None) -> np.ndarray:
    """
    Add patterns designed to interfere with scanner screen frequencies.
    Creates visible artifacts when scanned or photocopied.
    Optimized for small codes (28×14mm).
    """
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    # Use optimized parameters if provided
    if pattern_params is None:
        pattern_params = calculate_optimal_pattern_params((w, h))
    
    screen_freqs = pattern_params.get('freq_periods', [2.0, 2.5, 3.0, 3.5, 4.0])
    
    y, x = np.ogrid[:h, :w]
    
    interference = np.zeros((h, w), dtype=np.float32)
    for freq in screen_freqs:
        # Horizontal interference
        interference += np.sin(2 * np.pi * x / freq) * 3
        # Vertical interference
        interference += np.sin(2 * np.pi * y / freq) * 3
        # Diagonal interference
        interference += np.sin(2 * np.pi * (x + y) / freq) * 2
    
    # Apply interference to all channels
    for c in range(3):
        result[:, :, c] = np.clip(result[:, :, c] + interference, 0, 255)
    
    return result.astype(np.uint8)


def add_anti_photocopy_pattern(image: np.ndarray, cdp_id: str = "", 
                                physical_size_mm: tuple = (28, 14), dpi: int = 1200) -> np.ndarray:
    """
    Add comprehensive anti-photocopy patterns that break when photocopied:
    - Enhanced Moiré patterns (fine parallel lines)
    - Micro-dots
    - Fine gradient patterns
    - Frequency interference patterns
    - Color-shift security
    - Embedded security marks
    - Guilloche patterns
    - Screen frequency interference
    Optimized for small codes (28×14mm) at high DPI (1200+).
    
    Args:
        image: Input image array
        product_id: Product ID for deterministic patterns
        physical_size_mm: Physical dimensions in mm (width, height)
        dpi: Printing resolution (default: 1200 DPI)
    """
    h, w = image.shape[:2]
    result = image.copy()
    
    # Calculate optimal pattern parameters for small codes
    pattern_params = calculate_optimal_pattern_params((w, h), physical_size_mm, dpi)
    moire_spacing = int(pattern_params['moire_spacing'])
    microdot_size = pattern_params['microdot_size']
    
    # 1. Enhanced moiré pattern (fine parallel lines that interfere with scanner)
    # Use optimized spacing for small codes
    spacing = max(2, moire_spacing // 2)  # Convert to integer spacing
    for i in range(0, h, spacing):
        if i % (spacing * 2) < spacing // 2:  # Create alternating pattern
            result[i, :] = np.clip(result[i, :].astype(np.int16) - 25, 0, 255).astype(np.uint8)
    
    # Add diagonal moiré lines for more complexity
    for i in range(0, w, spacing):
        if i % (spacing * 2) < spacing // 2:
            result[:, i] = np.clip(result[:, i].astype(np.int16) - 20, 0, 255).astype(np.uint8)
    
    # 2. Enhanced micro-dots pattern (circular dots that don't reproduce well)
    # For small codes, slightly increase density for visibility
    dot_density = 0.96 if pattern_params['is_small_code'] else 0.97
    microdot_size = pattern_params.get('microdot_size', 3)
    dot_radius = max(1, microdot_size // 2)  # Radius for circular dots (half of size)
    
    # Create a temporary image for drawing circular dots
    dot_overlay_light = np.zeros_like(result, dtype=np.int16)
    dot_overlay_dark = np.zeros_like(result, dtype=np.int16)
    
    # Generate random positions for bright dots
    dot_positions = np.random.random((h, w)) > dot_density
    y_coords, x_coords = np.where(dot_positions)
    
    # Draw circular bright dots (lighter than background)
    for y, x in zip(y_coords, x_coords):
        # Only draw if not too close to edge
        if dot_radius < x < w - dot_radius and dot_radius < y < h - dot_radius:
            cv2.circle(dot_overlay_light, (x, y), dot_radius, (40, 40, 40), -1)  # Light dots
    
    # Add darker circular micro-dots for contrast
    dark_dot_density = 0.985
    dark_dot_positions = np.random.random((h, w)) > dark_dot_density
    dark_y_coords, dark_x_coords = np.where(dark_dot_positions)
    
    # Draw circular dark dots
    for y, x in zip(dark_y_coords, dark_x_coords):
        # Only draw if not too close to edge
        if dot_radius < x < w - dot_radius and dot_radius < y < h - dot_radius:
            cv2.circle(dot_overlay_dark, (x, y), dot_radius, (30, 30, 30), -1)  # Dark dots (will subtract)
    
    # Blend dots with the result image: add light dots, subtract dark dots
    result = np.clip(result.astype(np.int16) + dot_overlay_light - dot_overlay_dark, 0, 255).astype(np.uint8)
    
    # 3. Fine gradient overlay (hard to reproduce accurately)
    y, x = np.ogrid[:h, :w]
    grad_freq_x = pattern_params.get('gradient_freq_x', 0.1)
    grad_freq_y = pattern_params.get('gradient_freq_y', 0.1)
    gradient = np.sin(x * grad_freq_x) * np.cos(y * grad_freq_y) * 15
    result = np.clip(result.astype(np.int16) + gradient[:, :, np.newaxis], 0, 255).astype(np.uint8)
    
    # Add additional wave pattern
    wave_pattern = np.sin(x * grad_freq_x * 0.5) * np.cos(y * grad_freq_y * 0.5) * 12
    result = np.clip(result.astype(np.int16) + wave_pattern[:, :, np.newaxis], 0, 255).astype(np.uint8)
    
    # 4. Add frequency interference patterns
    result = add_frequency_interference_pattern(result, pattern_params)
    
    # 5. Add color-shift security (if cdp_id provided)
    if cdp_id:
        result = add_color_shift_security(result, cdp_id)
    
    # 6. Add embedded security marks (if cdp_id provided)
    if cdp_id:
        result = add_embedded_security_marks(result, cdp_id, pattern_params)
    
    # 7. Add Guilloche patterns (if cdp_id provided)
    if cdp_id:
        result = add_guilloche_pattern(result, cdp_id, pattern_params)
    
    # 8. Add screen frequency interference
    result = add_screen_frequency_interference(result, pattern_params)
    
    return result


def add_microprinting(image: np.ndarray, text: str, position: tuple) -> np.ndarray:
    """
    Add microprinting (tiny text that becomes unreadable when photocopied).
    Uses OpenCV to draw text instead of PIL to avoid font rendering issues.
    """
    result = image.copy()
    
    # Use OpenCV's putText for more reliable text rendering
    # This avoids PIL font division by zero issues
    try:
        # Try different font scales to find one that works
        font_scale = 0.3
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size to ensure it fits
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Ensure position is within image bounds
        x, y = position
        if x + text_width > result.shape[1]:
            x = result.shape[1] - text_width - 2
        if y - text_height < 0:
            y = text_height + 2
        
        # Draw text in black
        cv2.putText(result, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    except Exception as e:
        # If text drawing fails, skip microprinting (not critical)
        print(f"[WARNING] Failed to draw microprinting: {str(e)}, skipping...", flush=True)
    
    return result


def add_holographic_effect(image: np.ndarray) -> np.ndarray:
    """
    Add holographic-like effect using gradients that don't reproduce well.
    Enhanced for better visibility in CMYK format.
    """
    h, w = image.shape[:2]
    result = image.copy().astype(np.float32)
    
    # Create rainbow-like gradient overlay
    y, x = np.ogrid[:h, :w]
    angle = np.arctan2(y - h/2, x - w/2)
    
    # Multiple gradient layers for richer holographic effect
    gradient1 = (np.sin(angle * 3) + 1) / 2 * 20  # Increased from 15 to 20
    gradient2 = (np.sin(angle * 5 + np.pi/4) + 1) / 2 * 15  # Additional layer
    gradient3 = (np.cos(angle * 2) + 1) / 2 * 12  # Third layer
    
    combined_gradient = gradient1 + gradient2 * 0.6 + gradient3 * 0.4
    
    # Apply gradient to each color channel with slight variations
    for c in range(3):
        channel_gradient = combined_gradient * (1.0 + 0.1 * np.sin(c * np.pi / 3))
        result[:, :, c] = np.clip(result[:, :, c] + channel_gradient, 0, 255)
    
    return result.astype(np.uint8)


def enhance_qr_with_security(qr_img: np.ndarray, qr_data: str, is_small_code: bool = False) -> np.ndarray:
    """
    Enhance QR code with advanced security features that make it unreproducible.
    Features are designed to break when photocopied while maintaining QR readability.
    Enhanced for better visibility in CMYK format.
    Optimized for small codes (28×14mm) with adjusted parameters.
    
    Args:
        qr_img: QR code image to enhance
        qr_data: QR code data (product_id)
        is_small_code: If True, uses adjusted thresholds and parameters for small codes
    """
    h, w = qr_img.shape[:2]
    enhanced = qr_img.copy()
    
    # Use product_id as seed for deterministic randomness
    seed = int(hashlib.md5(qr_data.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    
    # Adjust parameters for small codes
    # Small codes need more subtle patterns to maintain QR readability
    noise_std = 4 if is_small_code else 6
    moire_spacing = 6 if is_small_code else 5
    moire_intensity = 3 if is_small_code else 5
    freq_intensity = 2 if is_small_code else 3
    border_width = 1 if is_small_code else 2
    corner_size_ratio = 10 if is_small_code else 8  # Smaller corners for small codes
    
    # 1. Add subtle noise pattern that doesn't affect QR readability but breaks photocopies
    # Reduced noise for small codes to maintain readability
    noise = np.random.normal(0, noise_std, (h, w, 3))
    enhanced = np.clip(enhanced.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 2. Add fine border pattern around QR code with security features
    # Thinner border for small codes
    # Create border with embedded pattern
    border_pattern = np.random.randint(100, 156, (border_width, w, 3), dtype=np.uint8)
    enhanced[:border_width, :] = border_pattern
    enhanced[-border_width:, :] = border_pattern
    border_pattern_v = np.random.randint(100, 156, (h, border_width, 3), dtype=np.uint8)
    enhanced[:, :border_width] = border_pattern_v
    enhanced[:, -border_width:] = border_pattern_v
    
    # 3. Add subtle moiré pattern to QR background (doesn't affect scanning)
    # Only apply to white areas to preserve QR code readability
    # Adjusted spacing and intensity for small codes
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    white_mask = gray > 200  # White areas
    
    for i in range(0, h, moire_spacing):
        if i % (moire_spacing * 2) < moire_spacing // 2:
            # Only modify white background areas
            mask = white_mask[i, :]
            if mask.any():  # Check if mask has any True values
                enhanced[i, :][mask] = np.clip(enhanced[i, :][mask].astype(np.int16) - moire_intensity, 0, 255).astype(np.uint8)
    
    # 4. Add frequency interference to white areas only
    # Adjusted frequency and intensity for small codes
    y, x = np.ogrid[:h, :w]
    freq_multiplier = 0.18 if is_small_code else 0.15  # Slightly higher frequency for small codes
    freq_pattern = np.sin(x * freq_multiplier) * np.cos(y * freq_multiplier) * freq_intensity
    # Apply only to white areas
    for c in range(3):
        channel_white = white_mask
        if channel_white.any():  # Check if mask has any True values
            enhanced[:, :, c] = np.where(
                channel_white,
                np.clip(
                    enhanced[:, :, c].astype(np.int16) + freq_pattern,
                    0, 255
                ).astype(np.uint8),
                enhanced[:, :, c]
            )
    
    # 5. Add embedded security marks in corners (becomes obvious when copied)
    # Smaller corners for small codes to maintain QR readability
    corner_size = max(2, min(h, w) // corner_size_ratio)  # Ensure minimum size
    corner_pattern = np.random.randint(80, 176, (corner_size, corner_size, 3), dtype=np.uint8)
    enhanced[:corner_size, :corner_size] = corner_pattern
    enhanced[-corner_size:, :corner_size] = corner_pattern
    enhanced[:corner_size, -corner_size:] = corner_pattern
    enhanced[-corner_size:, -corner_size:] = corner_pattern
    
    return enhanced


def generate_cdp_with_security(cdp_id: str = None, cdp_size: tuple = None, 
                                physical_size_mm: tuple = (28, 14), dpi: int = 1200) -> Tuple[np.ndarray, str]:
    """
    Generate CDP pattern with comprehensive anti-photocopy features.
    Uses cryptographically secure randomness (NOT derived from product_id).
    Each CDP is unique and non-derivable.
    Includes multiple layers of security to make it unreproducible.
    Optimized for small codes (28×14mm) at high DPI.
    
    Args:
        cdp_id: Optional CDP identifier (UUID). If None, generates new one.
        cdp_size: CDP size in pixels (width, height) - REQUIRED
        physical_size_mm: Physical dimensions in mm (width, height) - default 28×14mm
        dpi: Printing resolution (default: 1200 DPI)
    
    Returns:
        Tuple of (cdp_pattern, cdp_id)
    """
    if cdp_size is None:
        raise ValueError("cdp_size is required for CDP generation")
    
    # Generate or use provided CDP ID (cryptographically random UUID)
    if cdp_id is None:
        cdp_id = str(uuid.uuid4())
    
    # Use cryptographically secure random seed (NOT derived from product_id)
    # This ensures each CDP is unique and non-derivable
    seed_bytes = hashlib.sha256(cdp_id.encode()).digest()
    seed_int = int.from_bytes(seed_bytes[:8], byteorder='big')
    # Constrain seed to numpy's valid range (0 to 2^32 - 1)
    seed_int = seed_int % (2**32)
    np.random.seed(seed_int)
    
    # Generate base random pattern using secure randomness
    # Use secrets module for additional cryptographic randomness
    random_bytes = secrets.token_bytes(cdp_size[0] * cdp_size[1] * 3)
    random_array = np.frombuffer(random_bytes, dtype=np.uint8)
    cdp_pattern = random_array.reshape((cdp_size[1], cdp_size[0], 3))
    
    # Ensure values are in valid range
    cdp_pattern = np.clip(cdp_pattern, 0, 255)
    
    # Apply comprehensive anti-photocopy patterns (in order of application)
    # 1. Base anti-photocopy patterns (includes frequency interference, color-shift, etc.)
    # Use cdp_id instead of product_id for pattern generation
    cdp_pattern = add_anti_photocopy_pattern(cdp_pattern, cdp_id, physical_size_mm, dpi)
    
    # 2. Holographic effect (gradient overlays)
    cdp_pattern = add_holographic_effect(cdp_pattern)
    
    # 3. Guilloche patterns (complex curved patterns) - already included in add_anti_photocopy_pattern
    # But apply again with optimized params for additional layer
    pattern_params = calculate_optimal_pattern_params(cdp_size, physical_size_mm, dpi)
    cdp_pattern = add_guilloche_pattern(cdp_pattern, cdp_id, pattern_params)
    
    # 4. Screen frequency interference - already included in add_anti_photocopy_pattern
    # But apply again for additional layer
    cdp_pattern = add_screen_frequency_interference(cdp_pattern, pattern_params)
    
    # 5. Add microprinting with CDP ID hash (multiple locations)
    # For small codes, use smaller font and fewer locations
    hash_snippet = hashlib.md5(cdp_id.encode()).hexdigest()[:8]
    if pattern_params['is_small_code']:
        # Smaller codes: fewer microprinting locations, smaller positions
        cdp_pattern = add_microprinting(cdp_pattern, hash_snippet[:6], (5, 5))
        cdp_pattern = add_microprinting(cdp_pattern, hash_snippet[6:], (cdp_size[0] - 60, cdp_size[1] - 15))
    else:
        # Normal codes: multiple locations
        cdp_pattern = add_microprinting(cdp_pattern, hash_snippet, (10, 10))
        cdp_pattern = add_microprinting(cdp_pattern, hash_snippet, (cdp_size[0] - 80, 10))
        cdp_pattern = add_microprinting(cdp_pattern, hash_snippet, (10, cdp_size[1] - 20))
    
    return cdp_pattern, cdp_id


def generate_qr_cdp(serial_id: str, qr_size=None, cdp_size=None,
                    padding=20, border_thickness=25, 
                    physical_size_mm: tuple = (28, 14), dpi: int = 1200,
                    size_suffix: str = None, cdp_id: str = None) -> Tuple[np.ndarray, str]:
    """
    Generates a QR code and a random CDP pattern placed horizontally side-by-side
    with padding and a bold red border.
    Enhanced with anti-photocopy and anti-image-scanning features.
    
    QR code encodes serial_id (pointer), not product_id directly.
    CDP is cryptographically random and non-derivable.
    
    Args:
        serial_id: Serial ID to encode in QR code (pointer to backend)
        cdp_id: Optional CDP ID (if None, generates new one)
    
    Returns:
        Tuple of (combined_image, cdp_id)
    
    If qr_size and cdp_size are not specified, they will be calculated to achieve
    the target physical_size_mm when printed at the specified dpi.
    """
    # Calculate pixel dimensions for target physical size at specified DPI
    if qr_size is None or cdp_size is None:
        # Calculate total image dimensions in pixels
        pixels_per_mm = dpi / 25.4
        total_width_px = int(physical_size_mm[0] * pixels_per_mm)
        total_height_px = int(physical_size_mm[1] * pixels_per_mm)
        
        # Calculate content area (excluding border)
        content_width = total_width_px - 2 * border_thickness
        content_height = total_height_px - 2 * border_thickness
        
        # Split width between QR and CDP with padding
        if qr_size is None:
            qr_width = content_width // 2
            qr_size = (qr_width, content_height)
        
        if cdp_size is None:
            cdp_width = content_width - qr_size[0] - padding
            cdp_size = (cdp_width, content_height)

    # --- Detect if this is a small code for adaptive parameters ---
    is_small_code = physical_size_mm[0] < 30 or physical_size_mm[1] < 20  # 28×14mm or smaller
    
    # --- Generate QR Code ---
    # QR code encodes serial_id (pointer), not product_id
    # Adjust QR code parameters for small codes
    # Small codes need higher error correction and appropriate box size
    if is_small_code:
        # For small codes, use version 1 with high error correction
        # Smaller box_size helps maintain readability at small physical sizes
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=3,  # Smaller box size for small codes
            border=1,
        )
    else:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=4,
            border=1,
        )
    
    qr.add_data(serial_id)  # Encode serial_id, not product_id
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_img = qr_img.convert("RGB").resize(qr_size)
    qr_cv = np.array(qr_img)
    qr_cv = cv2.cvtColor(qr_cv, cv2.COLOR_RGB2BGR)
    
    # --- Enhance QR Code with Security Features ---
    # Pass is_small_code flag for adaptive security features
    qr_cv = enhance_qr_with_security(qr_cv, serial_id, is_small_code=is_small_code)

    # --- Generate CDP Pattern with Security Features ---
    # CDP is cryptographically random, not derived from product_id
    # Pass physical dimensions for small code optimization (28×14mm)
    cdp_pattern, cdp_id = generate_cdp_with_security(cdp_id, cdp_size, physical_size_mm, dpi)

    # --- Combine Horizontally with Padding ---
    h = max(qr_cv.shape[0], cdp_pattern.shape[0])
    w = qr_cv.shape[1] + cdp_pattern.shape[1] + padding

    combined = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background

    # Place QR
    combined[0:qr_cv.shape[0], 0:qr_cv.shape[1]] = qr_cv
    # Place CDP with padding
    combined[0:cdp_pattern.shape[0], qr_cv.shape[1] + padding:w] = cdp_pattern

    # --- Add Red Border Around Entire Image ---
    bordered = cv2.copyMakeBorder(
        combined,
        top=border_thickness,
        bottom=border_thickness,
        left=border_thickness,
        right=border_thickness,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 255]  # Bright red in BGR format
    )

    # --- Save the combined QR+CDP image in CMYK format (primary) ---
    # Save as CMYK TIFF for printing, with RGB PNG as fallback
    # Include size suffix in filename if provided
    if size_suffix:
        base_filename = f"{serial_id}_{size_suffix}"
    else:
        base_filename = f"{serial_id}"
    file_path = os.path.join(OUTPUT_DIR, base_filename)
    cmyk_tiff_path = save_cmyk_image(bordered, file_path, save_rgb_fallback=True, dpi=dpi)
    print(f"[INFO] QR+CDP image saved in CMYK format: {cmyk_tiff_path}")

    # --- Extract and save CDP portion separately in CMYK format ---
    # Remove red border first
    combined_no_border = bordered[border_thickness:-border_thickness, border_thickness:-border_thickness]
    
    # Extract CDP region (right side after QR code)
    qr_width = qr_cv.shape[1]
    cdp_start = qr_width + padding
    cdp_region = combined_no_border[:, cdp_start:]
    
    # Save CDP to separate folder in CMYK format (primary)
    # Use cdp_id in filename instead of serial_id
    if size_suffix:
        cdp_base_filename = f"{cdp_id}_{size_suffix}"
    else:
        cdp_base_filename = f"{cdp_id}"
    cdp_file_path = os.path.join(CDP_DIR, cdp_base_filename)
    cdp_cmyk_tiff_path = save_cmyk_image(cdp_region, cdp_file_path, save_rgb_fallback=True, dpi=dpi)
    print(f"[INFO] CDP image saved in CMYK format: {cdp_cmyk_tiff_path}")

    return bordered, cdp_id


def cv2_to_base64(img: np.ndarray) -> str:
    """Convert OpenCV image to Base64 PNG string"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode("utf-8")
