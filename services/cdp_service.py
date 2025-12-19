import cv2
import numpy as np
import os
from datetime import datetime
from numpy.fft import fft2, fftshift
from PIL import Image

EXTRACTED_CDP_DIR = "extracted_cdp"
os.makedirs(EXTRACTED_CDP_DIR, exist_ok=True)


def detect_yellow_border(image):
    """
    Detect red border and return bounding rectangle containing QR+CDP.
    Enhanced to be more robust for large QR codes and various lighting conditions.
    Uses contour analysis to find the actual card border, not just any red region.
    Note: Function name kept as detect_yellow_border for backward compatibility.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV (expanded range for better detection)
    # Red wraps around 0/180 in HSV, so we need two ranges
    # Lower saturation threshold to catch faded red borders
    lower_red1 = np.array([0, 100, 100])    # Red near 0
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])  # Red near 180
    upper_red2 = np.array([180, 255, 255])

    # Threshold to get only red areas (combine both ranges)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Apply morphological operations to fill gaps and connect border segments
    # This helps when the border is partially occluded or has gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

    # Find contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # No red detected, return original image
        print("[WARNING] No red border detected, using full image")
        return image

    # Filter contours to find the actual card border
    # The card border should be:
    # 1. Large enough (at least 5% of image area)
    # 2. Have a reasonable aspect ratio (not too elongated)
    # 3. Be roughly rectangular (card shape)
    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * 0.05  # At least 5% of image
    max_area = image_area * 0.95  # Not more than 95% (shouldn't be entire image)
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Check aspect ratio - card should be roughly rectangular
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            # Aspect ratio should be reasonable (between 1.2 and 3.0 for typical cards)
            if 1.2 <= aspect_ratio <= 3.0:
                # Check how rectangular the contour is
                rect_area = w * h
                extent = area / max(rect_area, 1)
                # Extent should be > 0.7 (fairly rectangular)
                if extent > 0.7:
                    valid_contours.append((contour, area, extent))
    
    if not valid_contours:
        # No valid contours found, use largest one as fallback
        print("[WARNING] No valid card-shaped contours found, using largest red region")
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        # Use the contour with best combination of area and rectangularity
        # Prefer larger, more rectangular contours
        best_contour = max(valid_contours, key=lambda x: x[1] * x[2])[0]
        largest_contour = best_contour
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Validate that we found a reasonable border
    # Border should be at least 10% of image size in both dimensions
    min_border_size = min(image.shape[0], image.shape[1]) * 0.1
    if w < min_border_size or h < min_border_size:
        print(f"[WARNING] Detected border too small ({w}x{h}), using full image")
        return image
    
    # Additional validation: border shouldn't be too close to image edges
    # (unless the card is actually at the edge)
    edge_margin = 5
    if (x < edge_margin and w > image.shape[1] - edge_margin * 2) or \
       (y < edge_margin and h > image.shape[0] - edge_margin * 2):
        # Border spans almost entire image, might be wrong
        print(f"[WARNING] Detected border spans entire image, might be incorrect")
    
    # Add small padding inside the red border to avoid including border pixels
    # Reduced padding to preserve more of the image content
    padding = 3
    x = max(x + padding, 0)
    y = max(y + padding, 0)
    w = min(w - 2*padding, image.shape[1] - x)
    h = min(h - 2*padding, image.shape[0] - y)
    
    # Ensure we have valid dimensions
    if w <= 0 or h <= 0:
        print(f"[WARNING] Invalid border dimensions after padding, using full image")
        return image

    cropped = image[y:y+h, x:x+w]
    print(f"[INFO] Red border detected: bbox=({x}, {y}, {w}, {h}), cropped shape={cropped.shape}")
    return cropped


def detect_cdp_region_by_texture(image, qr_rightmost_x, qr_width, padding=20):
    """
    Detect CDP region by analyzing texture and visual characteristics.
    CDP has high variance, texture, and specific color patterns.
    Uses QR code width to determine CDP width (they should be the same size).
    Returns the optimal CDP region boundaries.
    """
    h, w = image.shape[:2]
    
    # CDP should start after QR code + padding
    cdp_start = qr_rightmost_x + padding
    
    # CDP should be the same size as QR code (both are square and same height)
    # Use QR code width to determine CDP width
    expected_cdp_width = qr_width
    
    # Calculate CDP end position based on QR code width
    # CDP should be the same size as QR code
    cdp_end = cdp_start + expected_cdp_width
    
    # Ensure we don't go beyond image bounds
    if cdp_end > w:
        # If expected CDP extends beyond image, use image edge
        cdp_end = w
        print(f"[INFO] CDP extends to image edge: end={cdp_end} (expected: {cdp_start + expected_cdp_width})")
    elif cdp_end < w - 10:  # If there's space beyond expected end
        # Check if there's more CDP beyond the expected end
        # Scan a bit beyond to see if texture continues (up to 20% more)
        scan_end = min(cdp_start + int(expected_cdp_width * 1.2), w)
        
        # Calculate variance for different end positions
        best_end = cdp_end
        best_variance = 0
        
        # Sample every 5 pixels to find optimal end
        for test_end in range(cdp_start + expected_cdp_width // 2, scan_end, 5):
            test_region = image[:, cdp_start:test_end]
            if test_region.shape[1] < 10:
                continue
            test_gray = cv2.cvtColor(test_region, cv2.COLOR_BGR2GRAY)
            test_variance = np.var(test_gray)
            
            # Prefer regions with high variance (CDP texture)
            if test_variance > best_variance:
                best_variance = test_variance
                best_end = test_end
        
        # Only use texture-based end if it's close to expected width
        # (within 20% tolerance) and has good variance
        width_diff = abs(best_end - cdp_end) / max(expected_cdp_width, 1)
        if best_variance > 400 and width_diff < 0.2:
            cdp_end = best_end
            print(f"[INFO] Found optimal CDP end by texture: variance={best_variance:.2f}, end={cdp_end}")
        else:
            # Use expected width based on QR code
            cdp_end = cdp_start + expected_cdp_width
            print(f"[INFO] Using QR-based CDP width: {expected_cdp_width}px, end={cdp_end}")
    else:
        # Use expected width based on QR code
        cdp_end = cdp_start + expected_cdp_width
        print(f"[INFO] Using QR-based CDP width: {expected_cdp_width}px, end={cdp_end}")
    
    # Final validation: ensure CDP region has good texture
    if cdp_end > cdp_start:
        candidate_region = image[:, cdp_start:cdp_end]
        gray_region = cv2.cvtColor(candidate_region, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray_region)
        
        # If variance is too low, might have included background
        if variance < 300:
            print(f"[WARNING] CDP region has low variance ({variance:.2f}), but using it anyway")
    
    return cdp_start, cdp_end


def extract_cdp_region(combined_image, save_file=True, output_dir=None, custom_filename=None):
    """
    Extracts the CDP region from a horizontally combined QR+CDP image.
    The image structure: [Red Border][QR Code (left)][Padding][CDP (right, square)][Red Border]
    CDP is square and has the same height as the image.
    Detects red border automatically, then extracts the square CDP from the right side.
    Optionally saves extracted CDP image to disk.
    
    Args:
        combined_image: Combined QR+CDP image with red border
        save_file: If True, saves the extracted CDP
        output_dir: Custom directory to save to (default: EXTRACTED_CDP_DIR)
        custom_filename: Custom filename (default: auto-generated with timestamp)
    """
    # Crop to red border first - this ensures we work with the correct region
    cropped = detect_yellow_border(combined_image)

    # Save the cropped red border image to 'cropped_qr_cdp' folder
    save_cropped_dir = "cropped_qr_cdp"
    os.makedirs(save_cropped_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    cropped_filename = f"cropped_qr_cdp_{timestamp}.png"
    cropped_path = os.path.join(save_cropped_dir, cropped_filename)
    try:
        cv2.imwrite(cropped_path, cropped)
        print(f"[INFO] Cropped red border image saved: {cropped_path}")
    except Exception as e:
        print(f"[WARNING] Failed to save cropped yellow border image: {str(e)}")
    
    # Validate cropped region - should be reasonable size
    if cropped.shape[0] < 50 or cropped.shape[1] < 100:
        print(f"[WARNING] Cropped region too small: {cropped.shape}, using full image")
        cropped = combined_image

    h, w = cropped.shape[:2]
    
    # CDP is square and has the same height as the image
    # The structure is: [QR Code (left)][Padding][CDP (right, square)]
    # We need to extract only the CDP, not the QR code
    
    # Method: Try to detect where QR code ends by analyzing the image
    # QR codes have high contrast patterns (black/white), while CDP has more uniform texture
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # Analyze vertical strips to find the transition from QR (high variance) to CDP (lower variance)
    # Sample vertical strips across the image
    strip_width = max(10, w // 20)  # Sample strips
    variances = []
    x_positions = []
    
    for x in range(0, w - strip_width, strip_width):
        strip = gray[:, x:x + strip_width]
        variance = np.var(strip)
        variances.append(variance)
        x_positions.append(x)
    
    # Find where variance drops significantly (transition from QR to CDP)
    # QR code has high variance (black/white patterns), CDP has lower variance
    if len(variances) > 2:
        # Find the point where variance drops (QR ends, CDP begins)
        # Look for a significant drop in variance from left to right
        max_variance = max(variances)
        min_variance = min(variances)
        threshold = (max_variance + min_variance) / 2
        
        # Find the rightmost position where variance is still high (end of QR)
        qr_end_x = w  # Default to end
        for i in range(len(variances) - 1, -1, -1):
            if variances[i] > threshold:
                qr_end_x = x_positions[i] + strip_width
                break
        
        # CDP should start after QR + some padding
        # Estimate padding (typically 20 pixels based on generation code)
        estimated_padding = 20
        cdp_start_x = qr_end_x + estimated_padding
        
        print(f"[INFO] Detected QR end at x={qr_end_x}, CDP estimated to start at x={cdp_start_x}")
    else:
        # Fallback: assume QR takes up about half the width
        cdp_start_x = w // 2
        print(f"[INFO] Using fallback: assuming CDP starts at midpoint x={cdp_start_x}")
    
    # CDP is square (height x height)
    cdp_size = h
    
    # Ensure CDP start position leaves enough room for a square CDP
    if cdp_start_x + cdp_size > w:
        # Adjust: extract the rightmost square that fits
        cdp_start_x = max(0, w - cdp_size)
        print(f"[INFO] Adjusted CDP start to x={cdp_start_x} to fit square CDP")
    
    # Extract square CDP region from right side only (after QR code)
    cdp = cropped[:, cdp_start_x:cdp_start_x + cdp_size]
    
    # Ensure extracted CDP is square by matching dimensions
    if cdp.shape[0] != cdp.shape[1]:
        # Make it square by using the height as the reference
        if cdp.shape[0] > cdp.shape[1]:
            # Height is larger, crop height to match width
            cdp = cdp[:cdp.shape[1], :]
        else:
            # Width is larger, crop width to match height
            cdp = cdp[:, :cdp.shape[0]]
        print(f"[INFO] Adjusted CDP to square: final shape={cdp.shape}")
    
    print(f"[INFO] Extracted square CDP: image width={w}, height={h}, CDP start={cdp_start_x}, CDP size={cdp_size}, extracted shape={cdp.shape}")
    
    # Validate extracted CDP region
    if cdp.shape[0] == 0 or cdp.shape[1] == 0:
        print(f"[ERROR] Extracted CDP region is empty! Image shape: {cropped.shape}, CDP start: {cdp_start_x}, size: {cdp_size}")
        # Last resort: return right third of image
        cdp = cropped[:, 2 * w // 3:]
        print(f"[WARNING] Using right third as fallback: CDP shape={cdp.shape}")
    else:
        # Check texture of extracted region to verify it looks like CDP
        gray_cdp = cv2.cvtColor(cdp, cv2.COLOR_BGR2GRAY)
        cdp_variance = np.var(gray_cdp)
        print(f"[INFO] Extracted CDP region: width={cdp.shape[1]}, height={cdp.shape[0]}, variance={cdp_variance:.2f}")
        
        # CDP should have reasonable texture/variance
        if cdp_variance < 100:
            print(f"[WARNING] Extracted region has very low variance ({cdp_variance:.2f}), might not be CDP")

    # Save extracted CDP image if requested
    if save_file:
        if custom_filename:
            filename = custom_filename
        else:
            filename = f"cdp_extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        if output_dir:
            file_path = os.path.join(output_dir, filename)
        else:
            file_path = os.path.join(EXTRACTED_CDP_DIR, filename)
        
        cv2.imwrite(file_path, cdp)
        print(f"[INFO] Extracted CDP saved: {file_path} (shape: {cdp.shape})")

    return cdp


def detect_pattern_degradation(scanned_gray, reference_gray):
    """
    Detect degradation of specific security patterns that indicate photocopying.
    Checks for loss of frequency interference and Guilloche patterns.
    Note: Microdot checks removed as requested.
    
    Returns:
        dict with degradation scores for different pattern types
    """
    degradation_scores = {}
    
    # Downsample for faster FFT operations (maintains frequency characteristics)
    h_orig, w_orig = scanned_gray.shape
    if h_orig > 256 or w_orig > 256:
        scale = min(256 / h_orig, 256 / w_orig)
        h_new, w_new = int(h_orig * scale), int(w_orig * scale)
        scanned_gray = cv2.resize(scanned_gray, (w_new, h_new), interpolation=cv2.INTER_AREA)
        reference_gray = cv2.resize(reference_gray, (w_new, h_new), interpolation=cv2.INTER_AREA)
    
    # 1. Frequency pattern analysis
    # Check if high-frequency patterns are preserved
    f_scanned = fftshift(fft2(scanned_gray.astype(np.float32)))
    f_reference = fftshift(fft2(reference_gray.astype(np.float32)))
    
    # High frequency energy (patterns at 2-4 pixel periods)
    h, w = scanned_gray.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    
    # Check fewer frequency bands for speed (reduced from 4 to 2)
    freq_bands = [
        (2.5, 3.0),   # Medium-high
        (3.0, 3.5),   # Medium (Guilloche patterns)
    ]
    
    freq_preservation = []
    for min_freq, max_freq in freq_bands:
        # Create mask for this frequency band
        radius_min = min_freq * min(h, w) / 10
        radius_max = max_freq * min(h, w) / 10
        mask = ((x - center_x)**2 + (y - center_y)**2 >= radius_min**2) & \
               ((x - center_x)**2 + (y - center_y)**2 <= radius_max**2)
        
        # Compare energy in this band
        ref_energy = np.sum(np.abs(f_reference[mask]))
        scan_energy = np.sum(np.abs(f_scanned[mask]))
        
        if ref_energy > 0:
            preservation = scan_energy / ref_energy
            freq_preservation.append(preservation)
    
    # Average frequency preservation (photocopies lose high frequencies)
    degradation_scores['frequency_loss'] = 1.0 - np.mean(freq_preservation) if freq_preservation else 0.0
    
    # 2. Pattern structure analysis (Guilloche patterns)
    # Guilloche patterns have specific curvature characteristics
    # Use gradient magnitude to detect pattern structure - downsample for speed
    gray_grad_scale = 0.5
    h_grad, w_grad = int(h * gray_grad_scale), int(w * gray_grad_scale)
    ref_gray_small = cv2.resize(reference_gray, (w_grad, h_grad), interpolation=cv2.INTER_AREA)
    scan_gray_small = cv2.resize(scanned_gray, (w_grad, h_grad), interpolation=cv2.INTER_AREA)
    
    ref_grad_x = cv2.Sobel(ref_gray_small, cv2.CV_64F, 1, 0, ksize=3)
    ref_grad_y = cv2.Sobel(ref_gray_small, cv2.CV_64F, 0, 1, ksize=3)
    ref_grad_mag = np.sqrt(ref_grad_x**2 + ref_grad_y**2)
    
    scan_grad_x = cv2.Sobel(scan_gray_small, cv2.CV_64F, 1, 0, ksize=3)
    scan_grad_y = cv2.Sobel(scan_gray_small, cv2.CV_64F, 0, 1, ksize=3)
    scan_grad_mag = np.sqrt(scan_grad_x**2 + scan_grad_y**2)
    
    # Compare gradient structure (photocopies lose fine structure)
    grad_correlation = cv2.matchTemplate(
        (ref_grad_mag / (ref_grad_mag.max() + 1e-6)).astype(np.float32),
        (scan_grad_mag / (scan_grad_mag.max() + 1e-6)).astype(np.float32),
        cv2.TM_CCOEFF_NORMED
    )[0][0]
    degradation_scores['structure_loss'] = 1.0 - max(0, grad_correlation)
    
    return degradation_scores


def compare_cdp(scanned_image, reference_image):
    """
    Compare scanned CDP with reference CDP region.
    Returns similarity score between 0 and 1.
    Enhanced to work with new security patterns (frequency interference, Guilloche, etc.)
    Uses multiple robust comparison methods for better discrimination between products.
    
    Args:
        scanned_image: Can be combined QR+CDP image or just CDP
        reference_image: Can be combined QR+CDP image or just CDP
    """
    # Extract CDP portion from scanned image if it's a combined image
    # Check if it looks like a combined image (wider than tall, or has red border)
    if scanned_image.shape[1] > scanned_image.shape[0] * 1.5:
        scanned_cdp = extract_cdp_region(scanned_image, save_file=False)
    else:
        scanned_cdp = scanned_image
    
    # Extract CDP from reference image if combined
    if reference_image.shape[1] > reference_image.shape[0] * 1.5:
        reference_cdp = extract_cdp_region(reference_image, save_file=False)
    else:
        reference_cdp = reference_image
    
    # Resize both to same size for comparison
    # Use larger size for better pattern matching (256x256 matches generation size)
    target_size = (256, 256)
    scanned_cdp = cv2.resize(scanned_cdp, target_size, interpolation=cv2.INTER_CUBIC)
    reference_cdp = cv2.resize(reference_cdp, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale for more robust comparison (works with CMYK patterns)
    scanned_gray = cv2.cvtColor(scanned_cdp, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_cdp, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing: Enhance contrast and normalize
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better comparison
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    scanned_gray = clahe.apply(scanned_gray)
    reference_gray = clahe.apply(reference_gray)
    
    # Normalize images to have similar brightness/contrast
    scanned_gray = cv2.normalize(scanned_gray, None, 0, 255, cv2.NORM_MINMAX)
    reference_gray = cv2.normalize(reference_gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Detect pattern degradation (indicates photocopying)
    degradation = detect_pattern_degradation(scanned_gray, reference_gray)
    
    # High degradation indicates photocopy - penalize score
    # Note: Microdot checks removed, so only frequency and structure loss are used
    degradation_penalty = (degradation.get('frequency_loss', 0) * 0.5 + 
                          degradation.get('structure_loss', 0) * 0.5)
    
    # Use multiple comparison methods for better accuracy with security patterns
    # 1. Structural Similarity Index (SSIM) - better for patterns
    try:
        from scipy.ndimage import uniform_filter
        
        # Calculate SSIM
        def ssim(img1, img2):
            """Calculate Structural Similarity Index"""
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            
            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            
            mu1 = uniform_filter(img1, size=11)
            mu2 = uniform_filter(img2, size=11)
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = uniform_filter(img1 * img1, size=11) - mu1_sq
            sigma2_sq = uniform_filter(img2 * img2, size=11) - mu2_sq
            sigma12 = uniform_filter(img1 * img2, size=11) - mu1_mu2
            
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()
        
        ssim_score = ssim(scanned_gray, reference_gray)
    except ImportError:
        # Fallback if scipy not available - use simpler correlation
        ssim_score = 0.5
    
    # 2. Normalized correlation coefficient
    scanned_norm = scanned_gray.astype(np.float32) / 255.0
    reference_norm = reference_gray.astype(np.float32) / 255.0
    correlation = cv2.matchTemplate(scanned_norm, reference_norm, cv2.TM_CCOEFF_NORMED)[0][0]
    correlation = max(0, correlation)  # Ensure non-negative
    
    # 3. Histogram comparison (captures texture/pattern distribution)
    hist_scan = cv2.calcHist([scanned_gray], [0], None, [256], [0, 256])
    hist_ref = cv2.calcHist([reference_gray], [0], None, [256], [0, 256])
    # Normalize histograms
    hist_scan = hist_scan / (hist_scan.sum() + 1e-6)
    hist_ref = hist_ref / (hist_ref.sum() + 1e-6)
    # Use correlation coefficient for histogram comparison
    hist_correlation = cv2.compareHist(hist_scan, hist_ref, cv2.HISTCMP_CORREL)
    hist_correlation = max(0, hist_correlation) if not np.isnan(hist_correlation) else 0.0
    
    # 4. Edge-based comparison (pattern structure)
    # Use Canny edge detection to compare pattern structures
    scanned_edges = cv2.Canny(scanned_gray, 50, 150)
    reference_edges = cv2.Canny(reference_gray, 50, 150)
    # Compare edge maps using correlation
    edge_correlation = cv2.matchTemplate(
        scanned_edges.astype(np.float32) / 255.0,
        reference_edges.astype(np.float32) / 255.0,
        cv2.TM_CCOEFF_NORMED
    )[0][0]
    edge_correlation = max(0, edge_correlation)
    
    # 5. Local Binary Pattern (LBP) for texture comparison - optimized with vectorization
    def calculate_lbp_fast(image):
        """Calculate Local Binary Pattern using vectorized operations for speed"""
        h, w = image.shape
        # Downsample for speed (LBP is texture-based, works fine at lower resolution)
        if h > 128 or w > 128:
            scale = min(128 / h, 128 / w)
            h_new, w_new = int(h * scale), int(w * scale)
            image = cv2.resize(image, (w_new, h_new), interpolation=cv2.INTER_AREA)
            h, w = h_new, w_new
        
        # Vectorized LBP calculation
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        center = image[1:h-1, 1:w-1]
        lbp |= (image[0:h-2, 0:w-2] >= center).astype(np.uint8) << 7
        lbp |= (image[0:h-2, 1:w-1] >= center).astype(np.uint8) << 6
        lbp |= (image[0:h-2, 2:w] >= center).astype(np.uint8) << 5
        lbp |= (image[1:h-1, 2:w] >= center).astype(np.uint8) << 4
        lbp |= (image[2:h, 2:w] >= center).astype(np.uint8) << 3
        lbp |= (image[2:h, 1:w-1] >= center).astype(np.uint8) << 2
        lbp |= (image[2:h, 0:w-2] >= center).astype(np.uint8) << 1
        lbp |= (image[1:h-1, 0:w-2] >= center).astype(np.uint8) << 0
        return lbp
    
    scanned_lbp = calculate_lbp_fast(scanned_gray)
    reference_lbp = calculate_lbp_fast(reference_gray)
    # Compare LBP histograms
    lbp_hist_scan = cv2.calcHist([scanned_lbp], [0], None, [256], [0, 256])
    lbp_hist_ref = cv2.calcHist([reference_lbp], [0], None, [256], [0, 256])
    lbp_hist_scan = lbp_hist_scan / (lbp_hist_scan.sum() + 1e-6)
    lbp_hist_ref = lbp_hist_ref / (lbp_hist_ref.sum() + 1e-6)
    lbp_correlation = cv2.compareHist(lbp_hist_scan, lbp_hist_ref, cv2.HISTCMP_CORREL)
    lbp_correlation = max(0, lbp_correlation) if not np.isnan(lbp_correlation) else 0.0
    
    # 6. Pixel difference (tolerance for slight variations)
    diff = cv2.absdiff(scanned_gray, reference_gray)
    tolerance = 10
    pixel_match = np.sum(diff <= tolerance) / diff.size
    
    # 7. Frequency domain comparison (for frequency interference patterns)
    f_scanned = fftshift(fft2(scanned_gray.astype(np.float32)))
    f_reference = fftshift(fft2(reference_gray.astype(np.float32)))
    
    # Compare frequency domain (normalized)
    f_scanned_norm = np.abs(f_scanned) / (np.abs(f_scanned).max() + 1e-6)
    f_reference_norm = np.abs(f_reference) / (np.abs(f_reference).max() + 1e-6)
    freq_correlation = np.corrcoef(f_scanned_norm.ravel(), f_reference_norm.ravel())[0, 1]
    freq_correlation = max(0, freq_correlation) if not np.isnan(freq_correlation) else 0.0
    
    # 8. Multi-scale comparison (compare at different resolutions) - reduced to 2 scales for speed
    scales = [0.75, 1.0]  # Reduced from 3 to 2 scales
    scale_scores = []
    for scale in scales:
        if scale != 1.0:
            h, w = scanned_gray.shape
            scan_scaled = cv2.resize(scanned_gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            ref_scaled = cv2.resize(reference_gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
            # Resize back to original for comparison
            scan_scaled = cv2.resize(scan_scaled, (w, h), interpolation=cv2.INTER_AREA)
            ref_scaled = cv2.resize(ref_scaled, (w, h), interpolation=cv2.INTER_AREA)
        else:
            scan_scaled = scanned_gray
            ref_scaled = reference_gray
        
        scale_corr = cv2.matchTemplate(
            scan_scaled.astype(np.float32) / 255.0,
            ref_scaled.astype(np.float32) / 255.0,
            cv2.TM_CCOEFF_NORMED
        )[0][0]
        scale_scores.append(max(0, scale_corr))
    multi_scale_score = np.mean(scale_scores)
    
    # Combine scores with improved weights for better discrimination
    # SSIM (20%), correlation (15%), histogram (15%), edge (15%), LBP (15%), 
    # pixel match (10%), frequency (5%), multi-scale (5%)
    base_score = (
        ssim_score * 0.20 +
        correlation * 0.15 +
        hist_correlation * 0.15 +
        edge_correlation * 0.15 +
        lbp_correlation * 0.15 +
        pixel_match * 0.10 +
        freq_correlation * 0.05 +
        multi_scale_score * 0.05
    )
    
    # Apply degradation penalty (reduces score if patterns are degraded)
    combined_score = base_score * (1.0 - degradation_penalty * 0.5)
    
    # Ensure score is in valid range
    combined_score = max(0.0, min(1.0, combined_score))
    
    return round(float(combined_score), 3)


def identify_product_from_cdp(scanned_image, cdp_dir, min_match_score=0.6):
    """
    Identify product_id from CDP pattern by matching against all reference CDPs.
    This allows validation to work purely from the CDP itself without needing QR code.
    
    Args:
        scanned_image: Scanned image containing CDP (can be combined QR+CDP or just CDP)
        cdp_dir: Directory containing reference CDP files
        min_match_score: Minimum similarity score to consider a match (default: 0.6)
    
    Returns:
        tuple: (product_id, match_score) if match found, (None, 0.0) otherwise
    """
    # Extract CDP region from scanned image
    if scanned_image.shape[1] > scanned_image.shape[0] * 1.5:
        scanned_cdp = extract_cdp_region(scanned_image, save_file=True)
    else:
        scanned_cdp = scanned_image
    
    if scanned_cdp is None or scanned_cdp.size == 0:
        print("[ERROR] Failed to extract CDP from scanned image")
        return None, 0.0
    
    # Get all reference CDP files
    if not os.path.exists(cdp_dir):
        print(f"[ERROR] CDP directory not found: {cdp_dir}")
        return None, 0.0
    
    # Find all CDP files (prefer TIFF, fallback to PNG)
    cdp_files = []
    for filename in os.listdir(cdp_dir):
        if filename.endswith('.tiff') or filename.endswith('.png'):
            # Extract product_id from filename (remove extension)
            product_id = os.path.splitext(filename)[0]
            filepath = os.path.join(cdp_dir, filename)
            cdp_files.append((product_id, filepath, filename.endswith('.tiff')))
    
    if not cdp_files:
        print(f"[ERROR] No reference CDP files found in {cdp_dir}")
        return None, 0.0
    
    print(f"[INFO] Matching scanned CDP against {len(cdp_files)} reference CDPs...")
    
    best_match = None
    best_score = 0.0
    
    # Compare against each reference CDP
    for product_id, filepath, is_tiff in cdp_files:
        try:
            # Load reference CDP
            if is_tiff:
                # Use PIL to properly read CMYK TIFF
                try:
                    pil_img = Image.open(filepath)
                    if pil_img.mode == 'CMYK':
                        pil_img = pil_img.convert('RGB')
                    ref_img = np.array(pil_img)
                    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"[WARNING] Failed to load TIFF with PIL, trying OpenCV: {str(e)}")
                    ref_img = cv2.imread(filepath)
            else:
                ref_img = cv2.imread(filepath)
            
            if ref_img is None:
                print(f"[WARNING] Failed to load reference CDP: {filepath}")
                continue
            
            # Compare CDP patterns
            score = compare_cdp(scanned_cdp, ref_img)
            
            if score > best_score:
                best_score = score
                best_match = product_id
            
            print(f"[DEBUG] Product {product_id}: similarity score = {score:.3f}")
            
        except Exception as e:
            print(f"[WARNING] Error comparing with {product_id}: {str(e)}")
            continue
    
    # Return best match if it meets minimum threshold
    if best_match and best_score >= min_match_score:
        print(f"[INFO] Best match: product_id={best_match}, score={best_score:.3f}")
        return best_match, best_score
    else:
        print(f"[WARNING] No match found above threshold (best score: {best_score:.3f}, threshold: {min_match_score})")
        return None, best_score
