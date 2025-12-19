"""
Feature extraction module for production-ready anti-counterfeiting.
Extracts robust signals that are hard to reproduce but stable across captures.
"""
import cv2
import numpy as np
from numpy.fft import fft2, fftshift
from scipy.ndimage import uniform_filter
from typing import Dict, Tuple


def extract_frequency_band_energy(image: np.ndarray) -> Dict[str, float]:
    """
    Extract multi-scale frequency energy ratios.
    These are robust to lighting/angle but sensitive to reproduction quality.
    
    Returns:
        Dictionary with frequency band energy ratios
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Downsample for faster FFT (maintains frequency characteristics)
    h, w = gray.shape
    if h > 256 or w > 256:
        scale = min(256 / h, 256 / w)
        h_new, w_new = int(h * scale), int(w * scale)
        gray = cv2.resize(gray, (w_new, h_new), interpolation=cv2.INTER_AREA)
        h, w = h_new, w_new
    
    # Compute FFT
    f = fftshift(fft2(gray.astype(np.float32)))
    ps = np.abs(f) ** 2  # Power spectrum
    
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    
    # Define frequency bands (in pixel periods)
    # Lower frequencies = larger patterns, higher frequencies = fine details
    bands = {
        'low': (0.0, 1.5),      # Very low frequency (smooth gradients)
        'medium_low': (1.5, 2.5), # Medium-low (coarse patterns)
        'medium': (2.5, 3.5),    # Medium (main patterns)
        'medium_high': (3.5, 4.5), # Medium-high (fine details)
        'high': (4.5, 10.0)      # High frequency (very fine details, noise)
    }
    
    band_energies = {}
    total_energy = np.sum(ps)
    
    for band_name, (min_freq, max_freq) in bands.items():
        # Convert frequency periods to radius
        radius_min = min_freq * min(h, w) / 10
        radius_max = max_freq * min(h, w) / 10
        
        # Create mask for this frequency band
        mask = ((x - center_x)**2 + (y - center_y)**2 >= radius_min**2) & \
               ((x - center_x)**2 + (y - center_y)**2 <= radius_max**2)
        
        band_energy = np.sum(ps[mask])
        band_energies[band_name] = float(band_energy / (total_energy + 1e-6))
    
    # Calculate energy ratios (more stable than absolute values)
    # Explicitly convert to Python float to ensure JSON serializability
    ratios = {
        'low_to_medium': float(band_energies['low'] / (band_energies['medium'] + 1e-6)),
        'medium_to_high': float(band_energies['medium'] / (band_energies['high'] + 1e-6)),
        'medium_high_to_high': float(band_energies['medium_high'] / (band_energies['high'] + 1e-6)),
        'total_high_freq_ratio': float(band_energies['high'] / (total_energy + 1e-6)),
    }
    
    return {**band_energies, **ratios}


def extract_edge_density_metrics(image: np.ndarray) -> Dict[str, float]:
    """
    Extract edge preservation metrics.
    Authentic prints have consistent edge characteristics.
    
    Returns:
        Dictionary with edge density and strength metrics
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Normalize for consistent comparison
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Calculate gradient magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Edge density (proportion of pixels with significant edges)
    edge_threshold = np.percentile(grad_mag, 75)  # Top 25% of gradients
    edge_mask = grad_mag > edge_threshold
    edge_density = float(np.sum(edge_mask) / grad_mag.size)
    
    # Edge strength statistics
    edge_strength_mean = float(np.mean(grad_mag))
    edge_strength_std = float(np.std(grad_mag))
    edge_strength_max = float(np.max(grad_mag))
    
    # Canny edge detection for edge count
    edges = cv2.Canny(gray, 50, 150)
    edge_pixel_count = int(np.sum(edges > 0))
    edge_pixel_density = float(edge_pixel_count / edges.size)
    
    return {
        'edge_density': edge_density,
        'edge_strength_mean': edge_strength_mean,
        'edge_strength_std': edge_strength_std,
        'edge_strength_max': edge_strength_max,
        'edge_pixel_density': edge_pixel_density,
        'edge_pixel_count': edge_pixel_count
    }


def extract_texture_descriptors(image: np.ndarray) -> Dict[str, float]:
    """
    Extract texture consistency using Local Binary Pattern (LBP) histograms.
    LBP captures local texture patterns that are stable across captures.
    
    Returns:
        Dictionary with LBP histogram features
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Downsample for speed (LBP is texture-based, works fine at lower resolution)
    h, w = gray.shape
    if h > 128 or w > 128:
        scale = min(128 / h, 128 / w)
        h_new, w_new = int(h * scale), int(w * scale)
        gray = cv2.resize(gray, (w_new, h_new), interpolation=cv2.INTER_AREA)
        h, w = h_new, w_new
    
    # Calculate LBP using vectorized operations
    lbp = np.zeros((h-2, w-2), dtype=np.uint8)
    center = gray[1:h-1, 1:w-1]
    lbp |= (gray[0:h-2, 0:w-2] >= center).astype(np.uint8) << 7
    lbp |= (gray[0:h-2, 1:w-1] >= center).astype(np.uint8) << 6
    lbp |= (gray[0:h-2, 2:w] >= center).astype(np.uint8) << 5
    lbp |= (gray[1:h-1, 2:w] >= center).astype(np.uint8) << 4
    lbp |= (gray[2:h, 2:w] >= center).astype(np.uint8) << 3
    lbp |= (gray[2:h, 1:w-1] >= center).astype(np.uint8) << 2
    lbp |= (gray[2:h, 0:w-2] >= center).astype(np.uint8) << 1
    lbp |= (gray[1:h-1, 0:w-2] >= center).astype(np.uint8) << 0
    
    # Calculate LBP histogram
    lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    lbp_hist = lbp_hist / (lbp_hist.sum() + 1e-6)  # Normalize
    
    # Extract histogram features
    # Uniformity (higher = more uniform texture)
    uniformity = float(np.sum(lbp_hist**2))
    
    # Entropy (measure of texture complexity)
    hist_nonzero = lbp_hist[lbp_hist > 0]
    entropy = float(-np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-10)))
    
    # Dominant patterns (top 5 most common LBP values)
    top_indices = np.argsort(lbp_hist.flatten())[-5:][::-1]
    top_values = [float(lbp_hist[i]) for i in top_indices]
    
    return {
        'lbp_uniformity': uniformity,
        'lbp_entropy': entropy,
        'lbp_top_5_values': top_values,
        'lbp_top_5_indices': [int(i) for i in top_indices]
    }


def extract_all_features(image: np.ndarray) -> Dict[str, any]:
    """
    Extract all robust features from CDP image.
    These features are used for server-side comparison.
    
    Args:
        image: CDP image (BGR format)
    
    Returns:
        Dictionary with all extracted features
    """
    features = {}
    
    # 1. Frequency band energy ratios
    freq_features = extract_frequency_band_energy(image)
    features.update({f'freq_{k}': v for k, v in freq_features.items()})
    
    # 2. Edge density metrics
    edge_features = extract_edge_density_metrics(image)
    features.update({f'edge_{k}': v for k, v in edge_features.items()})
    
    # 3. Texture descriptors (LBP)
    texture_features = extract_texture_descriptors(image)
    features.update({f'texture_{k}': v for k, v in texture_features.items()})
    
    # 4. Basic image statistics (for normalization)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    features['mean_brightness'] = float(np.mean(gray))
    features['std_brightness'] = float(np.std(gray))
    features['image_height'] = int(image.shape[0])
    features['image_width'] = int(image.shape[1])
    
    return features


def compare_features(reference_features: Dict, scanned_features: Dict) -> float:
    """
    Compare reference and scanned features to compute similarity score.
    Uses weighted combination of feature differences.
    
    Args:
        reference_features: Features from reference CDP
        scanned_features: Features from scanned CDP
    
    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    if not reference_features or not scanned_features:
        return 0.0
    
    scores = []
    weights = []
    
    # 1. Frequency band energy comparison (30% weight)
    freq_keys = [k for k in reference_features.keys() if k.startswith('freq_')]
    freq_diffs = []
    for key in freq_keys:
        if key in scanned_features:
            ref_val = reference_features[key]
            scan_val = scanned_features[key]
            # Normalized difference
            diff = abs(ref_val - scan_val) / (abs(ref_val) + 1e-6)
            freq_diffs.append(1.0 - min(1.0, diff))
    if freq_diffs:
        freq_score = np.mean(freq_diffs)
        scores.append(freq_score)
        weights.append(0.30)
    
    # 2. Edge metrics comparison (25% weight)
    edge_keys = [k for k in reference_features.keys() if k.startswith('edge_')]
    edge_diffs = []
    for key in edge_keys:
        if key in scanned_features:
            ref_val = reference_features[key]
            scan_val = scanned_features[key]
            # Normalized difference
            diff = abs(ref_val - scan_val) / (abs(ref_val) + 1e-6)
            edge_diffs.append(1.0 - min(1.0, diff))
    if edge_diffs:
        edge_score = np.mean(edge_diffs)
        scores.append(edge_score)
        weights.append(0.25)
    
    # 3. Texture descriptors comparison (25% weight)
    texture_keys = [k for k in reference_features.keys() if k.startswith('texture_')]
    texture_diffs = []
    for key in texture_keys:
        if key in scanned_features and key != 'texture_lbp_top_5_values' and key != 'texture_lbp_top_5_indices':
            ref_val = reference_features[key]
            scan_val = scanned_features[key]
            # Normalized difference
            diff = abs(ref_val - scan_val) / (abs(ref_val) + 1e-6)
            texture_diffs.append(1.0 - min(1.0, diff))
    
    # Special handling for LBP histogram top values
    if 'texture_lbp_top_5_values' in reference_features and 'texture_lbp_top_5_values' in scanned_features:
        ref_top = reference_features['texture_lbp_top_5_values']
        scan_top = scanned_features['texture_lbp_top_5_values']
        # Compare histogram distributions
        top_diff = np.mean([abs(r - s) for r, s in zip(ref_top, scan_top)])
        texture_diffs.append(1.0 - min(1.0, top_diff))
    
    if texture_diffs:
        texture_score = np.mean(texture_diffs)
        scores.append(texture_score)
        weights.append(0.25)
    
    # 4. Brightness normalization check (20% weight) - ensures images are comparable
    if 'mean_brightness' in reference_features and 'mean_brightness' in scanned_features:
        ref_bright = reference_features['mean_brightness']
        scan_bright = scanned_features['mean_brightness']
        # Allow some brightness variation (normalized)
        brightness_diff = abs(ref_bright - scan_bright) / 255.0
        brightness_score = 1.0 - min(1.0, brightness_diff * 2)  # Allow up to 50% brightness difference
        scores.append(brightness_score)
        weights.append(0.20)
    
    # Weighted average
    if scores and weights:
        total_weight = sum(weights)
        if total_weight > 0:
            weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            return float(max(0.0, min(1.0, weighted_score)))
    
    return 0.0

