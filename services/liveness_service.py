import cv2
import numpy as np
from numpy.fft import fft2, fftshift

def detect_yellow_border(image):
    """
    Detect red border and return bounding rectangle containing QR+CDP.
    Returns the cropped image and the bounding box info.
    Note: Function name kept as detect_yellow_border for backward compatibility.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color range in HSV
    # Red wraps around 0/180 in HSV, so we need two ranges
    lower_red1 = np.array([0, 100, 100])    # Red near 0
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])  # Red near 180
    upper_red2 = np.array([180, 255, 255])

    # Combine both red ranges
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # No red detected, fallback to full image
        return image, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Optional padding inside border
    padding = 5
    x = max(x + padding, 0)
    y = max(y + padding, 0)
    w = min(w - 2*padding, image.shape[1] - x)
    h = min(h - 2*padding, image.shape[0] - y)

    bbox_info = {'x': x, 'y': y, 'w': w, 'h': h, 'area': w * h}
    return image[y:y+h, x:x+w], bbox_info


def detect_photocopy(image):
    """
    Detect if image is from a photocopy or scanned document.
    Enhanced to detect degradation of new security patterns:
    - Frequency interference patterns
    - Guilloche patterns
    - Color-shift security
    - Embedded security marks
    Note: Microdot checks removed as requested
    
    Photocopies typically have:
    - Reduced color depth (fewer distinct colors)
    - Loss of fine details (moiré patterns, microprinting, frequency patterns)
    - Uniform background texture
    - Reduced dynamic range
    - Loss of specific security pattern characteristics
    
    Args:
        image: Input image containing QR+CDP
    """
    region, _ = detect_yellow_border(image)
    
    # Resize region to smaller size for faster FFT operations (maintains frequency characteristics)
    h_orig, w_orig = region.shape[:2]
    if h_orig > 512 or w_orig > 512:
        scale = min(512 / h_orig, 512 / w_orig)
        h_new, w_new = int(h_orig * scale), int(w_orig * scale)
        region = cv2.resize(region, (w_new, h_new), interpolation=cv2.INTER_AREA)
    
    # 1. Check color depth (photocopies have fewer distinct colors)
    # Reshape to list of pixels
    pixels = region.reshape(-1, 3)
    # Count unique colors (with some tolerance for noise) - use sampling for speed
    sample_size = min(10000, pixels.shape[0])
    if pixels.shape[0] > sample_size:
        indices = np.random.choice(pixels.shape[0], sample_size, replace=False)
        pixels_sample = pixels[indices]
    else:
        pixels_sample = pixels
    unique_colors = len(np.unique(pixels_sample // 10, axis=0))
    total_pixels = pixels_sample.shape[0]
    color_diversity = unique_colors / total_pixels
    
    color_threshold = 0.05
    is_photocopy_color = color_diversity < color_threshold
    
    # 2. Check for loss of fine details (high frequency content)
    # This is critical for detecting loss of frequency interference patterns
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    f = fftshift(fft2(gray))
    ps = np.abs(f) ** 2
    
    # High frequency energy (edges, fine details, frequency patterns)
    h, w = ps.shape
    center_y, center_x = h // 2, w // 2
    # Outer ring represents high frequencies
    y, x = np.ogrid[:h, :w]
    mask_high_freq = ((x - center_x)**2 + (y - center_y)**2) > (min(h, w) * 0.3)**2
    high_freq_energy = np.sum(ps[mask_high_freq])
    total_energy = np.sum(ps)
    high_freq_ratio = high_freq_energy / (total_energy + 1e-6)
    
    # Check specific frequency bands (matching generation frequencies: 2.0-4.0 pixel periods)
    # Reduced to 3 bands instead of 5 for speed
    freq_bands_detected = 0
    for freq_period in [2.5, 3.0, 3.5]:  # Reduced from 5 to 3 bands
        radius = freq_period * min(h, w) / 10
        mask_band = ((x - center_x)**2 + (y - center_y)**2 >= (radius * 0.8)**2) & \
                   ((x - center_x)**2 + (y - center_y)**2 <= (radius * 1.2)**2)
        band_energy = np.sum(ps[mask_band])
        if band_energy > total_energy * 0.01:  # Band has significant energy
            freq_bands_detected += 1
    
    # Photocopies lose high frequency details and specific frequency bands
    detail_threshold = 0.1
    is_photocopy_detail = high_freq_ratio < detail_threshold
    # If fewer than 2 frequency bands detected, likely photocopy (reduced from 3)
    is_photocopy_freq_bands = freq_bands_detected < 2
    
    # 3. Check dynamic range (photocopies often have compressed dynamic range)
    gray_min, gray_max = np.min(gray), np.max(gray)
    dynamic_range = gray_max - gray_min
    range_threshold = 150
    is_photocopy_range = dynamic_range < range_threshold
    
    # 4. Check for uniform texture (photocopies have more uniform backgrounds)
    # Use smaller kernel and downsampled image for speed
    gray_small = cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2), interpolation=cv2.INTER_AREA)
    kernel = np.ones((3, 3), np.float32) / 9  # Smaller kernel
    local_mean = cv2.filter2D(gray_small.astype(np.float32), -1, kernel)
    local_var = cv2.filter2D((gray_small.astype(np.float32) - local_mean)**2, -1, kernel)
    avg_variance = np.mean(local_var)
    texture_threshold = 200
    is_photocopy_texture = avg_variance < texture_threshold
    
    # 5. Check for pattern structure (Guilloche patterns should have specific curvature)
    # Use gradient analysis to detect pattern structure - downsample for speed
    gray_grad = cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2), interpolation=cv2.INTER_AREA)
    grad_x = cv2.Sobel(gray_grad, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_grad, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    # Photocopies lose fine gradient structure
    grad_variance = np.var(grad_mag)
    grad_threshold = 600
    is_photocopy_structure = grad_variance < grad_threshold
    
    # Combine checks (if 3 or more indicators suggest photocopy for better accuracy)
    # Increased threshold from 2 to 3 to reduce false positives
    # Note: Microdot checks removed as requested
    photocopy_indicators = [
        is_photocopy_color, 
        is_photocopy_detail, 
        is_photocopy_freq_bands,
        is_photocopy_range, 
        is_photocopy_texture,
        is_photocopy_structure
    ]
    photocopy_score = sum(photocopy_indicators)
    is_photocopy = photocopy_score >= 3  # Need 3+ indicators (out of 6 now)
    
    print(f"[DEBUG] Photocopy detection - "
          f"Color:{is_photocopy_color}, Detail:{is_photocopy_detail}, "
          f"FreqBands:{is_photocopy_freq_bands}, Range:{is_photocopy_range}, "
          f"Texture:{is_photocopy_texture}, Structure:{is_photocopy_structure}, Score:{photocopy_score}/6", flush=True)
    
    return not is_photocopy  # Return True if NOT a photocopy (i.e., authentic)


def extract_cdp_region(image):
    """
    Extract CDP region from horizontally combined QR+CDP image with red border.
    The image structure: [Red Border][QR Code (left half)][CDP (right half)][Red Border]
    Right half is the CDP region.
    """
    cropped, _ = detect_yellow_border(image)
    # Split horizontally: left half = QR, right half = CDP
    mid = cropped.shape[1] // 2
    cdp = cropped[:, mid:]
    return cdp


def check_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def check_fft_screen(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = fftshift(fft2(gray))
    ps = np.log1p(np.abs(f))
    cy, cx = ps.shape[0] // 2, ps.shape[1] // 2
    ps[cy-5:cy+5, cx-5:cx+5] = 0
    ratio = np.sort(ps.ravel())[-10:].sum() / (ps.sum() + 1e-6)
    return ratio < 0.003  # True if likely printed (not screen)


def detect_static_image(video_frames):
    """
    Detect if video frames are actually static (suggesting a photo of an image).
    Real scanning should have some motion/variation between frames.
    """
    if len(video_frames) < 2:
        return True  # Need at least 2 frames to check
    
    # Limit to max 5 frames for performance (sample evenly)
    max_frames = 5
    if len(video_frames) > max_frames:
        step = len(video_frames) // max_frames
        video_frames = video_frames[::step][:max_frames]
    
    # Calculate frame differences - downsample for speed
    diffs = []
    for i in range(1, len(video_frames)):
        gray1 = cv2.cvtColor(video_frames[i-1], cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(video_frames[i], cv2.COLOR_BGR2GRAY)
        # Downsample to 1/4 size for faster comparison
        h, w = gray1.shape
        gray1_small = cv2.resize(gray1, (w//2, h//2), interpolation=cv2.INTER_AREA)
        gray2_small = cv2.resize(gray2, (w//2, h//2), interpolation=cv2.INTER_AREA)
        diff = cv2.absdiff(gray1_small, gray2_small)
        mean_diff = np.mean(diff)
        diffs.append(mean_diff)
    
    avg_diff = np.mean(diffs)
    # If average difference is very low, likely a static image
    is_static = avg_diff < 5.0
    
    print(f"[DEBUG] Static image detection - Avg frame diff: {avg_diff:.2f}, "
          f"Is static: {is_static}", flush=True)
    
    return is_static


def detect_small_code(image):
    """
    Detect if the scanned QR CDP code is small (e.g., 28×14 mm).
    Small codes typically have lower frame coverage even at optimal distance.
    Returns True if code appears to be small based on frame coverage.
    """
    # Get image dimensions
    image_height = image.shape[0]
    image_width = image.shape[1]
    image_area = image_height * image_width
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Red wraps around 0/180 in HSV, so we need two ranges
    lower_red1 = np.array([0, 100, 100])    # Red near 0
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])  # Red near 180
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"[DEBUG] detect_small_code: No red border detected. Image size: {image_width}x{image_height}", flush=True)
        return False
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    frame_area = w * h
    coverage_ratio = frame_area / image_area
    
    # Small codes (28×14 mm) typically have coverage between 12-35% even at optimal distance
    # If coverage is in this range and code is detected, it's likely a small printed code
    # Also check aspect ratio - small codes are rectangular (width:height = 2:1, so 28:14 = 2:1)
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
    
    # Detect small code: coverage 12-35% and rectangular aspect ratio (1.5-2.5:1)
    # For 28×14mm codes, width > height, so aspect ratio should be around 2:1
    is_small = (0.12 <= coverage_ratio <= 0.35) and (1.5 <= aspect_ratio <= 2.5)
    
    # Debug output
    print(f"[DEBUG] detect_small_code: Image dimensions: {image_width}x{image_height} (area: {image_area})", flush=True)
    print(f"[DEBUG] detect_small_code: Detected red border: width={w}, height={h} (area: {frame_area})", flush=True)
    print(f"[DEBUG] detect_small_code: Coverage ratio: {coverage_ratio:.3f} (12-35% for small codes)", flush=True)
    print(f"[DEBUG] detect_small_code: Aspect ratio: {aspect_ratio:.3f} (1.5-2.5 for small codes)", flush=True)
    print(f"[DEBUG] detect_small_code: Coverage check (0.12 <= {coverage_ratio:.3f} <= 0.35): {0.12 <= coverage_ratio <= 0.35}", flush=True)
    print(f"[DEBUG] detect_small_code: Aspect ratio check (1.5 <= {aspect_ratio:.3f} <= 2.5): {1.5 <= aspect_ratio <= 2.5}", flush=True)
    print(f"[DEBUG] detect_small_code: Final result - is_small_code: {is_small}", flush=True)
    
    return is_small


def detect_qr_code_size(image):
    """
    Detect QR code size category and provide warnings if too big or too small.
    Categorizes QR codes into: too_small, small, normal, large, too_large.
    
    Args:
        image: Input image containing QR CDP with red border
    
    Returns:
        dict with size assessment including:
        - size_category: 'too_small', 'small', 'normal', 'large', 'too_large'
        - coverage_ratio: Frame coverage as percentage
        - width_pixels: Width of detected frame in pixels
        - height_pixels: Height of detected frame in pixels
        - aspect_ratio: Width/height ratio
        - has_warnings: Boolean indicating if warnings exist
        - warnings: List of warning objects
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    warnings = []
    size_info = {}
    
    if not contours:
        warnings.append({
            'type': 'no_frame_detected',
            'severity': 'critical',
            'message': 'Yellow scanning frame not detected. Cannot determine QR code size.',
            'value': 0,
            'recommended': 'Ensure QR code with yellow border is visible in frame'
        })
        return {
            'size_info': {
                'size_category': 'unknown',
                'coverage_ratio': 0.0,
                'width_pixels': 0,
                'height_pixels': 0,
                'aspect_ratio': 0.0,
                'detected': False
            },
            'warnings': warnings,
            'has_warnings': True
        }
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    frame_area = w * h
    image_area = image.shape[0] * image.shape[1]
    coverage_ratio = (frame_area / image_area) * 100  # as percentage
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
    
    size_info['coverage_ratio'] = round(float(coverage_ratio), 2)
    size_info['width_pixels'] = int(w)
    size_info['height_pixels'] = int(h)
    size_info['aspect_ratio'] = round(float(aspect_ratio), 2)
    size_info['detected'] = True
    
    # Size categories based on coverage ratio:
    # - Too Small: < 5% (QR code is extremely small, likely too far or very small print)
    # - Small: 5-12% (Small printed code, may need closer scanning)
    # - Normal: 12-45% (Optimal size range for most QR codes)
    # - Large: 45-70% (Large QR code, may need to move further)
    # - Too Large: > 70% (QR code takes up most of frame, too close)
    
    if coverage_ratio < 5:
        size_category = 'too_small'
        warnings.append({
            'type': 'qr_too_small',
            'severity': 'critical',
            'message': f'QR code is extremely small ({coverage_ratio:.1f}% of frame). The code may be too far away, or the printed label is very small. Move closer or use a larger printed label.',
            'value': float(round(coverage_ratio, 1)),
            'recommended': '5-45% of frame'
        })
    elif coverage_ratio < 12:
        size_category = 'small'
        warnings.append({
            'type': 'qr_small',
            'severity': 'warning',
            'message': f'QR code is small ({coverage_ratio:.1f}% of frame). This may be a small printed label. Move closer for better results, or ensure the label is at least 2cm × 2cm.',
            'value': float(round(coverage_ratio, 1)),
            'recommended': '12-45% of frame for optimal scanning'
        })
    elif coverage_ratio <= 45:
        size_category = 'normal'
        # No warning for normal size
    elif coverage_ratio <= 70:
        size_category = 'large'
        warnings.append({
            'type': 'qr_large',
            'severity': 'warning',
            'message': f'QR code is large ({coverage_ratio:.1f}% of frame). Move camera further away for better focus and to capture the entire code properly.',
            'value': float(round(coverage_ratio, 1)),
            'recommended': '12-45% of frame for optimal scanning'
        })
    else:  # coverage_ratio > 70
        size_category = 'too_large'
        warnings.append({
            'type': 'qr_too_large',
            'severity': 'critical',
            'message': f'QR code is too large ({coverage_ratio:.1f}% of frame). Camera is too close. Move further away to capture the entire code and improve focus.',
            'value': float(round(coverage_ratio, 1)),
            'recommended': '12-45% of frame'
        })
    
    size_info['size_category'] = size_category
    
    # Additional check for aspect ratio (rectangular codes)
    if aspect_ratio > 2.5:
        warnings.append({
            'type': 'unusual_aspect_ratio',
            'severity': 'warning',
            'message': f'QR code has unusual aspect ratio ({aspect_ratio:.2f}:1). This may indicate a very rectangular label or scanning angle issue.',
            'value': float(round(aspect_ratio, 2)),
            'recommended': '1.0-2.5:1 (square to slightly rectangular)'
        })
    
    return {
        'size_info': size_info,
        'warnings': warnings,
        'has_warnings': len(warnings) > 0
    }


def check_scanning_frame_distance(image):
    """
    Check if scanning frame (yellow border) is at optimal distance.
    Returns distance assessment with warnings if too close or too far.
    
    Args:
        image: Input image containing QR CDP with yellow border
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    warnings = []
    frame_info = {}
    
    if not contours:
        warnings.append({
            'type': 'no_frame_detected',
            'severity': 'critical',
            'message': 'Yellow scanning frame not detected. Please ensure the QR code with yellow border is visible.',
            'value': 0,
            'recommended': 'Frame should cover 15-40% of image'
        })
        frame_info['detected'] = False
        frame_info['coverage_ratio'] = 0.0
        return {'frame_info': frame_info, 'warnings': warnings, 'has_warnings': True}
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    frame_area = w * h
    image_area = image.shape[0] * image.shape[1]
    coverage_ratio = frame_area / image_area
    
    frame_info['detected'] = True
    frame_info['coverage_ratio'] = round(float(coverage_ratio * 100), 2)  # as percentage
    frame_info['frame_area'] = frame_area
    frame_info['image_area'] = image_area
    
    # Standard thresholds for normal codes (≥20×20 mm): 20-45%
    min_coverage = 0.20
    optimal_min = 0.25
    optimal_max = 0.40
    max_coverage = 0.45
    recommended_range = '20-45% of image'
    distance_msg = '10-20 cm distance recommended'
    
    # Check distance and generate warnings
    if coverage_ratio < min_coverage:
        warnings.append({
            'type': 'frame_too_far',
            'severity': 'critical',
            'message': f'Camera is too far from QR code. Please move closer ({distance_msg}). Frame should cover {recommended_range}.',
            'value': float(round(coverage_ratio * 100, 1)),
            'recommended': recommended_range
        })
    elif coverage_ratio < optimal_min:
        warnings.append({
            'type': 'frame_too_far',
            'severity': 'warning',
            'message': f'Camera is somewhat far. Move closer for better results and more reliable detection.',
            'value': float(round(coverage_ratio * 100, 1)),
            'recommended': recommended_range
        })
    elif coverage_ratio > max_coverage:
        warnings.append({
            'type': 'frame_too_close',
            'severity': 'critical',
            'message': f'Camera is too close to QR code. Please move further away ({distance_msg}). Frame should cover {recommended_range}.',
            'value': float(round(coverage_ratio * 100, 1)),
            'recommended': recommended_range
        })
    elif coverage_ratio > optimal_max:
        warnings.append({
            'type': 'frame_too_close',
            'severity': 'warning',
            'message': f'Camera is somewhat close. Move further away for better focus and lighting consistency.',
            'value': float(round(coverage_ratio * 100, 1)),
            'recommended': recommended_range
        })
    
    return {
        'frame_info': frame_info,
        'warnings': warnings,
        'has_warnings': len(warnings) > 0
    }


def check_lighting_steadiness(video_frames):
    """
    Check if lighting is steady across multiple video frames.
    Unsteady lighting (flickering, shadows moving) indicates poor scanning conditions.
    """
    if len(video_frames) < 2:
        return {
            'steadiness_info': {'is_steady': True, 'variance': 0.0},
            'warnings': [],
            'has_warnings': False
        }
    
    # Limit to max 5 frames for performance (sample evenly)
    max_frames = 5
    if len(video_frames) > max_frames:
        step = len(video_frames) // max_frames
        video_frames = video_frames[::step][:max_frames]
    
    # Calculate brightness for each frame
    brightnesses = []
    contrasts = []
    
    # Cache yellow border detection for first frame, reuse bbox for others
    first_region, bbox_info = detect_yellow_border(video_frames[0])
    first_gray = cv2.cvtColor(first_region, cv2.COLOR_BGR2GRAY)
    brightnesses.append(np.mean(first_gray))
    contrasts.append(np.std(first_gray))
    
    # For remaining frames, use same bbox if available (faster)
    for frame in video_frames[1:]:
        if bbox_info:
            x, y, w, h = bbox_info['x'], bbox_info['y'], bbox_info['w'], bbox_info['h']
            region = frame[y:y+h, x:x+w] if y+h <= frame.shape[0] and x+w <= frame.shape[1] else frame
        else:
            region, _ = detect_yellow_border(frame)
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        brightnesses.append(np.mean(gray))
        contrasts.append(np.std(gray))
    
    # Calculate variance in brightness across frames
    brightness_variance = np.var(brightnesses)
    brightness_std = np.std(brightnesses)
    contrast_variance = np.var(contrasts)
    
    steadiness_info = {
        'brightness_mean': round(float(np.mean(brightnesses)), 2),
        'brightness_variance': round(float(brightness_variance), 2),
        'brightness_std': round(float(brightness_std), 2),
        'contrast_variance': round(float(contrast_variance), 2),
        'frame_count': len(video_frames)
    }
    
    warnings = []
    
    # Check brightness steadiness (should be relatively stable)
    # Calibrated thresholds:
    # - Critical (> 12): Significant flickering or moving shadows, unreliable lighting
    # - Warning (> 8): Moderate variation, may affect accuracy
    # - Good (< 8): Steady lighting across frames
    # Note: When frame distance is suboptimal, lighting variance tends to increase
    if brightness_std > 12:
        warnings.append({
            'type': 'unsteady_lighting',
            'severity': 'critical',
            'message': 'Lighting is unsteady (flickering or moving shadows detected). Please hold camera steady, ensure stable lighting, and maintain optimal distance (20-45% frame coverage).',
            'value': float(round(brightness_std, 2)),
            'recommended': '< 8'
        })
        steadiness_info['is_steady'] = False
    elif brightness_std > 8:
        warnings.append({
            'type': 'unsteady_lighting',
            'severity': 'warning',
            'message': 'Lighting is somewhat unsteady. Try to hold camera more steady, ensure stable lighting, and check frame distance.',
            'value': float(round(brightness_std, 2)),
            'recommended': '< 8'
        })
        steadiness_info['is_steady'] = False
    else:
        steadiness_info['is_steady'] = True
    
    # Check contrast steadiness
    # Higher threshold to account for natural variation
    if contrast_variance > 80:
        warnings.append({
            'type': 'unsteady_contrast',
            'severity': 'warning',
            'message': 'Image contrast varies between frames. This may affect scanning quality. Check frame distance and lighting stability.',
            'value': float(round(contrast_variance, 2)),
            'recommended': '< 50'
        })
    
    return {
        'steadiness_info': steadiness_info,
        'warnings': warnings,
        'has_warnings': len(warnings) > 0
    }


def assess_lighting_conditions(image, video_frames=None):
    """
    Assess lighting conditions and return warnings if inadequate.
    Enhanced with lighting steadiness check across frames if provided.
    Returns a dictionary with lighting assessment and warnings.
    """
    region, _ = detect_yellow_border(image)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    
    warnings = []
    lighting_info = {}
    
    # 1. Overall Brightness (mean pixel value)
    mean_brightness = np.mean(gray)
    lighting_info['brightness'] = float(round(float(mean_brightness), 2))
    
    # Calibrated brightness thresholds that account for frame distance:
    # - When frame is far, brightness appears lower due to smaller region
    # - Adjusted to be more lenient but still detect actual low light
    if mean_brightness < 45:
        warnings.append({
            'type': 'low_brightness',
            'severity': 'critical',
            'message': 'Lighting is too dim. Please move to a brighter area, add more light, or move camera closer to QR code.',
            'value': float(round(mean_brightness, 2)),
            'recommended': '> 75'
        })
    elif mean_brightness < 75:
        warnings.append({
            'type': 'low_brightness',
            'severity': 'warning',
            'message': 'Lighting is somewhat dim. Better results with brighter lighting or optimal frame distance (20-45% coverage).',
            'value': float(round(mean_brightness, 2)),
            'recommended': '> 75'
        })
    elif mean_brightness > 220:
        warnings.append({
            'type': 'high_brightness',
            'severity': 'warning',
            'message': 'Lighting may be too bright, causing overexposure. Try reducing light or adjusting angle.',
            'value': float(round(mean_brightness, 2)),
            'recommended': '80-200'
        })
    
    # 2. Brightness Variance (uneven lighting)
    brightness_variance = np.var(gray)
    lighting_info['brightness_variance'] = float(round(float(brightness_variance), 2))
    
    if brightness_variance > 3000:
        warnings.append({
            'type': 'uneven_lighting',
            'severity': 'warning',
            'message': 'Lighting appears uneven with harsh shadows. Try to find more even lighting.',
            'value': float(round(brightness_variance, 2)),
            'recommended': '< 2000'
        })
    
    # 3. Contrast
    contrast = np.std(gray)
    lighting_info['contrast'] = float(round(float(contrast), 2))
    
    if contrast < 20:
        warnings.append({
            'type': 'low_contrast',
            'severity': 'critical',
            'message': 'Contrast is too low. QR code may not be readable. Check print quality and lighting.',
            'value': float(round(contrast, 2)),
            'recommended': '> 30'
        })
    elif contrast < 30:
        warnings.append({
            'type': 'low_contrast',
            'severity': 'warning',
            'message': 'Contrast is somewhat low. Better results with higher contrast.',
            'value': float(round(contrast, 2)),
            'recommended': '> 30'
        })
    
    # 4. Dynamic Range
    gray_min, gray_max = np.min(gray), np.max(gray)
    dynamic_range = gray_max - gray_min
    lighting_info['dynamic_range'] = float(round(float(dynamic_range), 2))
    lighting_info['min_brightness'] = int(gray_min.item()) if hasattr(gray_min, 'item') else int(gray_min)
    lighting_info['max_brightness'] = int(gray_max.item()) if hasattr(gray_max, 'item') else int(gray_max)
    
    if dynamic_range < 100:
        warnings.append({
            'type': 'low_dynamic_range',
            'severity': 'warning',
            'message': 'Dynamic range is limited. Image may be over or underexposed.',
            'value': float(round(dynamic_range, 2)),
            'recommended': '> 150'
        })
    
    # 5. Exposure Assessment
    # Check if image is overexposed (too many bright pixels)
    bright_pixels = np.sum(gray > 240)
    bright_ratio = bright_pixels / gray.size
    lighting_info['overexposure_ratio'] = float(round(float(bright_ratio), 3))
    
    if bright_ratio > 0.3:
        warnings.append({
            'type': 'overexposure',
            'severity': 'warning',
            'message': 'Image appears overexposed. Reduce lighting or adjust camera exposure.',
            'value': float(round(bright_ratio * 100, 1)),
            'recommended': '< 20%'
        })
    
    # Check if image is underexposed (too many dark pixels)
    dark_pixels = np.sum(gray < 30)
    dark_ratio = dark_pixels / gray.size
    lighting_info['underexposure_ratio'] = float(round(float(dark_ratio), 3))
    
    if dark_ratio > 0.3:
        warnings.append({
            'type': 'underexposure',
            'severity': 'critical',
            'message': 'Image appears underexposed. Please add more light or move to a brighter area.',
            'value': float(round(dark_ratio * 100, 1)),
            'recommended': '< 20%'
        })
    
    # Overall lighting quality score (calibrated)
    quality_score = 100
    # Brightness penalties (adjusted thresholds)
    if mean_brightness < 45 or mean_brightness > 220:
        quality_score -= 30
    elif mean_brightness < 75 or mean_brightness > 200:
        quality_score -= 15
    
    # Contrast penalties
    if contrast < 18:
        quality_score -= 30
    elif contrast < 28:
        quality_score -= 15
    
    # Dynamic range penalties
    if dynamic_range < 90:
        quality_score -= 10
    
    # Uneven lighting penalties (adjusted threshold)
    if brightness_variance > 2800:
        quality_score -= 10
    
    quality_score = max(0, quality_score)
    lighting_info['quality_score'] = int(quality_score)
    
    # Determine overall lighting status
    critical_warnings = [w for w in warnings if w['severity'] == 'critical']
    if critical_warnings:
        lighting_info['status'] = 'poor'
    elif warnings:
        lighting_info['status'] = 'fair'
    else:
        lighting_info['status'] = 'good'
    
    result = {
        'lighting_info': lighting_info,
        'warnings': warnings,
        'has_warnings': len(warnings) > 0,
        'has_critical_warnings': len(critical_warnings) > 0
    }
    
    # Add lighting steadiness check if video frames provided
    if video_frames and len(video_frames) > 1:
        steadiness_check = check_lighting_steadiness(video_frames)
        result['steadiness'] = steadiness_check['steadiness_info']
        if steadiness_check['has_warnings']:
            result['warnings'].extend(steadiness_check['warnings'])
            result['has_warnings'] = True
            # Update critical warnings if needed
            new_critical = [w for w in steadiness_check['warnings'] if w['severity'] == 'critical']
            if new_critical:
                critical_warnings.extend(new_critical)
                result['has_critical_warnings'] = True
                if lighting_info['status'] != 'poor':
                    lighting_info['status'] = 'fair'
    
    return result


def liveness_check(image, video_frames=None):
    """
    LIMITED ROLE: Only rejects static screenshots and obvious replay attacks.
    NOT used as proof of authenticity - that comes from CDP feature matching.
    
    Args:
        image: Input image containing QR CDP
        video_frames: Optional list of video frames for motion detection
    
    Returns:
        True if passes basic liveness (not a static screenshot), False otherwise
    """
    # Only check for static images (screenshots, photos of screens)
    # This is a minimal check - main security is from CDP feature matching
    static_check = True
    if video_frames and len(video_frames) > 1:
        static_check = not detect_static_image(video_frames)
    
    print(f"[DEBUG] Liveness (limited role) - Static check: {static_check}", flush=True)
    
    # Only reject if clearly a static image
    return static_check
