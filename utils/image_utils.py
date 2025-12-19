import cv2
import numpy as np
import base64
import os
import sys
from datetime import datetime
from contextlib import contextmanager

# Try to import pyzbar, with fallback to OpenCV if it fails
PYZBAR_AVAILABLE = False
pyzbar_decode = None

@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output (for zbar warnings)"""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        try:
            sys.stderr = devnull
            yield
        finally:
            sys.stderr = old_stderr

# Set library path for macOS (Homebrew installation) before importing pyzbar
if sys.platform == 'darwin':
    homebrew_lib = '/opt/homebrew/lib'
    if os.path.exists(homebrew_lib):
        # Set DYLD_LIBRARY_PATH (though this may not work due to macOS security restrictions)
        current_dyld = os.environ.get('DYLD_LIBRARY_PATH', '')
        if homebrew_lib not in current_dyld:
            os.environ['DYLD_LIBRARY_PATH'] = f"{homebrew_lib}:{current_dyld}" if current_dyld else homebrew_lib

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    PYZBAR_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] pyzbar not available ({str(e)}), falling back to OpenCV QRCodeDetector", flush=True)
    pyzbar_decode = None
except Exception as e:
    print(f"[WARNING] pyzbar failed to load ({str(e)}), falling back to OpenCV QRCodeDetector", flush=True)
    pyzbar_decode = None

def base64_to_image(base64_str):
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode()

def analyze_image_statistics(image):
    """
    Comprehensive image analysis to extract detailed statistics.
    Helps differentiate between digital and original/physical images.
    
    Args:
        image: Input image (BGR format)
    
    Returns:
        dict: Comprehensive statistics about the image
    """
    if image is None or image.size == 0:
        return None
    
    stats = {}
    
    # Convert to different color spaces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    h, w = gray.shape
    total_pixels = h * w
    
    # ========== BASIC STATISTICS ==========
    # Brightness (mean intensity)
    stats['brightness'] = float(np.mean(gray))
    
    # Contrast (standard deviation)
    stats['contrast'] = float(np.std(gray))
    
    # Dynamic range (max - min)
    stats['dynamic_range'] = float(np.max(gray) - np.min(gray))
    
    # ========== SHARPNESS METRICS ==========
    # Laplacian variance (blur detection)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    stats['sharpness_laplacian'] = float(laplacian.var())
    
    # Gradient magnitude (edge strength)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    stats['edge_strength_mean'] = float(np.mean(grad_magnitude))
    stats['edge_strength_std'] = float(np.std(grad_magnitude))
    stats['edge_strength_max'] = float(np.max(grad_magnitude))
    
    # ========== HISTOGRAM ANALYSIS ==========
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_normalized = hist / (total_pixels + 1e-6)
    
    # Histogram peak (indicates uniformity - screens/photocopies have high peaks)
    stats['histogram_peak'] = float(np.max(hist_normalized))
    
    # Histogram entropy (measure of information content)
    hist_nonzero = hist_normalized[hist_normalized > 0]
    stats['histogram_entropy'] = float(-np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-10)))
    
    # Histogram spread (how spread out the values are)
    hist_mean = np.sum(np.arange(256) * hist_normalized.flatten())
    hist_std = np.sqrt(np.sum((np.arange(256) - hist_mean)**2 * hist_normalized.flatten()))
    stats['histogram_std'] = float(hist_std)
    
    # ========== NOISE ANALYSIS ==========
    # Estimate noise using median absolute deviation of Laplacian
    laplacian_abs = np.abs(laplacian)
    stats['noise_level'] = float(np.median(laplacian_abs))
    
    # High-frequency content (noise + fine details) - downsample for speed
    gray_fft = gray
    if h > 256 or w > 256:
        scale = min(256 / h, 256 / w)
        h_fft, w_fft = int(h * scale), int(w * scale)
        gray_fft = cv2.resize(gray, (w_fft, h_fft), interpolation=cv2.INTER_AREA)
    
    fft = np.fft.fft2(gray_fft.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    fft_magnitude = np.abs(fft_shift)
    
    # High frequency energy (outer 50% of frequency domain)
    h_fft, w_fft = gray_fft.shape
    center_y, center_x = h_fft // 2, w_fft // 2
    y, x = np.ogrid[:h_fft, :w_fft]
    radius = min(h_fft, w_fft) // 4
    high_freq_mask = ((x - center_x)**2 + (y - center_y)**2) > radius**2
    stats['high_frequency_energy'] = float(np.sum(fft_magnitude[high_freq_mask]))
    
    # ========== EDGE DETECTION METRICS ==========
    # Canny edges
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    stats['edge_density'] = float(edge_pixels / total_pixels)
    stats['edge_count'] = int(edge_pixels)
    
    # Edge connectivity (how connected edges are)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats['contour_count'] = len(contours)
    
    # ========== COLOR ANALYSIS ==========
    # Color diversity (unique colors with tolerance)
    pixels_bgr = image.reshape(-1, 3)
    # Quantize to reduce noise impact
    pixels_quantized = (pixels_bgr // 10) * 10
    unique_colors = len(np.unique(pixels_quantized, axis=0))
    stats['color_diversity'] = float(unique_colors / total_pixels)
    stats['unique_colors'] = int(unique_colors)
    
    # Color saturation (from HSV)
    saturation = hsv[:, :, 1]
    stats['saturation_mean'] = float(np.mean(saturation))
    stats['saturation_std'] = float(np.std(saturation))
    
    # Color channels statistics
    b, g, r = cv2.split(image)
    stats['channel_b_mean'] = float(np.mean(b))
    stats['channel_g_mean'] = float(np.mean(g))
    stats['channel_r_mean'] = float(np.mean(r))
    stats['channel_b_std'] = float(np.std(b))
    stats['channel_g_std'] = float(np.std(g))
    stats['channel_r_std'] = float(np.std(r))
    
    # ========== TEXTURE ANALYSIS ==========
    # Local Binary Pattern (LBP) - texture measure - optimized with vectorization
    def calculate_lbp_fast(img):
        """Fast vectorized LBP calculation for texture analysis"""
        h, w = img.shape
        # Downsample for speed (LBP is texture-based, works fine at lower resolution)
        if h > 128 or w > 128:
            scale = min(128 / h, 128 / w)
            h_new, w_new = int(h * scale), int(w * scale)
            img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
            h, w = h_new, w_new
        
        # Vectorized LBP calculation
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        center = img[1:h-1, 1:w-1]
        lbp |= (img[0:h-2, 0:w-2] >= center).astype(np.uint8) << 7
        lbp |= (img[0:h-2, 1:w-1] >= center).astype(np.uint8) << 6
        lbp |= (img[0:h-2, 2:w] >= center).astype(np.uint8) << 5
        lbp |= (img[1:h-1, 2:w] >= center).astype(np.uint8) << 4
        lbp |= (img[2:h, 2:w] >= center).astype(np.uint8) << 3
        lbp |= (img[2:h, 1:w-1] >= center).astype(np.uint8) << 2
        lbp |= (img[2:h, 0:w-2] >= center).astype(np.uint8) << 1
        lbp |= (img[1:h-1, 0:w-2] >= center).astype(np.uint8) << 0
        return lbp
    
    lbp = calculate_lbp_fast(gray)
    lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    lbp_hist_norm = lbp_hist / (lbp_hist.sum() + 1e-6)
    # LBP uniformity (higher = more uniform texture)
    stats['texture_uniformity'] = float(np.sum(lbp_hist_norm**2))
    
    # ========== COMPRESSION ARTIFACTS DETECTION ==========
    # Block artifacts (common in JPEG compression)
    # Check for 8x8 block patterns
    block_size = 8
    blocks_h = h // block_size
    blocks_w = w // block_size
    if blocks_h > 0 and blocks_w > 0:
        # Calculate variance within each block
        block_variances = []
        for i in range(blocks_h):
            for j in range(blocks_w):
                block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_variances.append(np.var(block))
        # Low variance blocks indicate compression artifacts
        stats['compression_artifacts'] = float(np.mean(block_variances))
    else:
        stats['compression_artifacts'] = 0.0
    
    # ========== IMAGE QUALITY SCORES ==========
    # Overall quality score (0-1)
    sharpness_threshold = 100
    contrast_threshold = 30
    hist_peak_threshold = 0.3
    
    is_sharp = stats['sharpness_laplacian'] > sharpness_threshold
    has_contrast = stats['contrast'] > contrast_threshold
    not_screen_photo = stats['histogram_peak'] < hist_peak_threshold
    
    stats['quality_ok'] = bool(is_sharp and has_contrast and not_screen_photo)
    stats['quality_score'] = float(
        (1.0 if is_sharp else stats['sharpness_laplacian'] / sharpness_threshold) * 0.4 +
        (1.0 if has_contrast else stats['contrast'] / contrast_threshold) * 0.3 +
        (1.0 if not_screen_photo else 1.0 - stats['histogram_peak'] / hist_peak_threshold) * 0.3
    )
    
    # ========== ADDITIONAL METRICS ==========
    # Image dimensions
    stats['width'] = int(w)
    stats['height'] = int(h)
    stats['aspect_ratio'] = float(w / h) if h > 0 else 0.0
    stats['total_pixels'] = int(total_pixels)
    
    # Brightness distribution
    stats['brightness_min'] = float(np.min(gray))
    stats['brightness_max'] = float(np.max(gray))
    stats['brightness_median'] = float(np.median(gray))
    
    return stats


def check_image_quality_for_qr(image):
    """
    Check if image quality is sufficient for QR code scanning.
    Low quality or heavily processed images may indicate photocopy/scan.
    
    Args:
        image: Input image containing QR code
    
    Returns:
        bool: True if quality is sufficient, False otherwise
    """
    stats = analyze_image_statistics(image)
    if stats is None:
        return False
    
    # Use the quality_ok flag from comprehensive analysis
    quality_ok = stats.get('quality_ok', False)
    
    # Print detailed statistics in format similar to user's example
    print(f"QR quality check - "
          f"Sharpness: {stats['sharpness_laplacian']:.2f} (threshold=100), "
          f"Contrast: {stats['contrast']:.2f}, "
          f"Hist peak: {stats['histogram_peak']:.3f}, "
          f"Quality OK: {quality_ok}", flush=True)
    
    # Print additional detailed statistics for digital vs original differentiation
    print(f"Image Statistics - "
          f"Edge density: {stats['edge_density']:.4f}, "
          f"Edge strength: {stats['edge_strength_mean']:.2f}, "
          f"Noise level: {stats['noise_level']:.2f}, "
          f"High freq energy: {stats['high_frequency_energy']:.2f}, "
          f"Color diversity: {stats['color_diversity']:.4f}, "
          f"Unique colors: {stats['unique_colors']}, "
          f"Saturation mean: {stats['saturation_mean']:.2f}, "
          f"Texture uniformity: {stats['texture_uniformity']:.4f}, "
          f"Compression artifacts: {stats['compression_artifacts']:.2f}, "
          f"Histogram entropy: {stats['histogram_entropy']:.2f}, "
          f"Dynamic range: {stats['dynamic_range']:.2f}, "
          f"Brightness: {stats['brightness']:.2f}, "
          f"Quality score: {stats['quality_score']:.3f}", flush=True)
    
    return quality_ok


def decode_qr_code(image, require_quality=True, save_folder=None):
    """
    Decode QR code from image to extract product ID.
    Uses pyzbar (ZBar backend) if available, otherwise falls back to OpenCV QRCodeDetector.
    pyzbar is more robust than OpenCV for QR detection, handling blurry/low-contrast QRs better.
    
    Args:
        image: Input image containing QR code (can be combined QR+CDP or just QR)
        require_quality: If True, only decode if image quality passes checks (currently not enforced)
        save_folder: Optional folder path to save the image. If provided, the image will be saved with a timestamp-based filename.
    
    Returns:
        Decoded string or None if QR code not found
    """
    if image is None or image.size == 0:
        print("[WARNING] Invalid image provided for QR decoding", flush=True)
        return None
    
    # Save image to folder if specified
    if save_folder:
        try:
            os.makedirs(save_folder, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"qr_decode_{timestamp}.png"
            file_path = os.path.join(save_folder, filename)
            cv2.imwrite(file_path, image)
            print(f"[INFO] Image saved to {file_path}", flush=True)
        except Exception as e:
            print(f"[WARNING] Failed to save image to {save_folder}: {str(e)}", flush=True)
    
    # Convert to grayscale (both pyzbar and OpenCV work best with grayscale)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Use pyzbar if available, otherwise fall back to OpenCV
    if PYZBAR_AVAILABLE and pyzbar_decode:
        # Try 1: Decode with pyzbar on grayscale image
        try:
            with suppress_stderr():
                decoded = pyzbar_decode(gray)
            if decoded:
                data = decoded[0].data.decode("utf-8")
                print(f"[INFO] Successfully decoded QR code (pyzbar): {data}", flush=True)
                return data
        except Exception as e:
            print(f"[DEBUG] pyzbar decode failed: {str(e)}", flush=True)
        
        # Try 2: Crop red border if present (for QR+CDP combined images)
        try:
            from services.cdp_service import detect_yellow_border
            cropped = detect_yellow_border(image.copy())
            if cropped.shape != image.shape:
                cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped
                with suppress_stderr():
                    decoded = pyzbar_decode(cropped_gray)
                if decoded:
                    data = decoded[0].data.decode("utf-8")
                    print(f"[INFO] Successfully decoded QR code from cropped image (pyzbar): {data}", flush=True)
                    return data
                gray = cropped_gray  # Use cropped image for further processing
        except Exception as e:
            print(f"[DEBUG] Red border detection failed: {str(e)}", flush=True)
        
        # Try 3: Extract QR region from combined images (left half for horizontal images)
        h, w = gray.shape[:2]
        if w > h * 1.5:  # Likely a combined horizontal image
            qr_region = gray[:, :w//2]
            try:
                with suppress_stderr():
                    decoded = pyzbar_decode(qr_region)
                if decoded:
                    data = decoded[0].data.decode("utf-8")
                    print(f"[INFO] Successfully decoded QR code from extracted region (pyzbar): {data}", flush=True)
                    return data
            except Exception:
                pass
        
        # Try 4: Simple contrast enhancement (CLAHE) - helps with low-contrast images
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            with suppress_stderr():
                decoded = pyzbar_decode(enhanced)
            if decoded:
                data = decoded[0].data.decode("utf-8")
                print(f"[INFO] Successfully decoded QR code from enhanced image (pyzbar): {data}", flush=True)
                return data
        except Exception:
            pass
        
        # Try 5: Upscale for very small images
        if min(h, w) < 150:
            try:
                upscaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                with suppress_stderr():
                    decoded = pyzbar_decode(upscaled)
                if decoded:
                    data = decoded[0].data.decode("utf-8")
                    print(f"[INFO] Successfully decoded QR code from upscaled image (pyzbar): {data}", flush=True)
                    return data
            except Exception:
                pass
    
    # Fallback to OpenCV QRCodeDetector if pyzbar is not available or failed
    detector = cv2.QRCodeDetector()
    
    # Try 1: Original image
    data, bbox, _ = detector.detectAndDecode(image)
    if data:
        print(f"[INFO] Successfully decoded QR code (OpenCV): {data}", flush=True)
        return data
    
    # Try 2: Grayscale
    data, bbox, _ = detector.detectAndDecode(gray)
    if data:
        print(f"[INFO] Successfully decoded QR code from grayscale (OpenCV): {data}", flush=True)
        return data
    
    # Try 3: Crop red border if present
    try:
        from services.cdp_service import detect_yellow_border
        cropped = detect_yellow_border(image.copy())
        if cropped.shape != image.shape:
            data, bbox, _ = detector.detectAndDecode(cropped)
            if data:
                print(f"[INFO] Successfully decoded QR code from cropped image (OpenCV): {data}", flush=True)
                return data
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if len(cropped.shape) == 3 else cropped
    except Exception:
        pass
    
    # Try 4: Extract QR region from combined images
    h, w = gray.shape[:2]
    if w > h * 1.5:
        qr_region = gray[:, :w//2]
        data, bbox, _ = detector.detectAndDecode(qr_region)
        if data:
            print(f"[INFO] Successfully decoded QR code from extracted region (OpenCV): {data}", flush=True)
            return data
    
    # Try 5: Contrast enhancement
    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        data, bbox, _ = detector.detectAndDecode(enhanced)
        if data:
            print(f"[INFO] Successfully decoded QR code from enhanced image (OpenCV): {data}", flush=True)
            return data
    except Exception:
        pass
    
    # All attempts failed
    print("[WARNING] Could not decode QR code from image", flush=True)
    return None