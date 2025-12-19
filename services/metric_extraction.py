"""
Metric Extraction Service
Extracts the 15 metrics needed for ML training from CDP images.
These metrics are used to train the authenticity classifier.
"""
import cv2
import numpy as np
from typing import Dict, Optional
from numpy.fft import fft2, fftshift


def extract_sharpness(image: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.
    Higher values indicate sharper images.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(laplacian_var)


def extract_contrast(image: np.ndarray) -> float:
    """
    Calculate image contrast as standard deviation of pixel values.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    contrast = np.std(gray)
    return float(contrast)


def extract_histogram_peak(image: np.ndarray) -> float:
    """
    Calculate the peak value in the histogram (most common pixel value).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    peak_idx = np.argmax(hist)
    peak_value = hist[peak_idx] / (gray.size)  # Normalize
    return float(peak_value)


def extract_edge_density(image: np.ndarray) -> float:
    """
    Calculate edge density as proportion of pixels with significant edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150)
    edge_pixels = np.sum(edges > 0)
    density = edge_pixels / edges.size
    return float(density)


def extract_edge_strength(image: np.ndarray) -> float:
    """
    Calculate average edge strength using gradient magnitude.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    mean_strength = np.mean(grad_mag)
    return float(mean_strength)


def extract_noise_level(image: np.ndarray) -> float:
    """
    Estimate noise level using high-frequency content.
    Higher values indicate more noise (photocopies, screenshots).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Downsample for faster FFT
    h, w = gray.shape
    if h > 256 or w > 256:
        scale = min(256 / h, 256 / w)
        h_new, w_new = int(h * scale), int(w * scale)
        gray = cv2.resize(gray, (w_new, h_new), interpolation=cv2.INTER_AREA)
    
    # FFT to get frequency content
    f = fftshift(fft2(gray.astype(np.float32)))
    ps = np.abs(f) ** 2
    
    # High frequency energy (noise)
    h, w = ps.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    mask_high_freq = ((x - center_x)**2 + (y - center_y)**2) > (min(h, w) * 0.3)**2
    high_freq_energy = np.sum(ps[mask_high_freq])
    total_energy = np.sum(ps)
    noise_ratio = high_freq_energy / (total_energy + 1e-6)
    
    # Scale to reasonable range
    noise_level = noise_ratio * 100.0
    return float(noise_level)


def extract_high_freq_energy(image: np.ndarray) -> float:
    """
    Calculate total high-frequency energy in the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Downsample for faster FFT
    h, w = gray.shape
    if h > 256 or w > 256:
        scale = min(256 / h, 256 / w)
        h_new, w_new = int(h * scale), int(w * scale)
        gray = cv2.resize(gray, (w_new, h_new), interpolation=cv2.INTER_AREA)
    
    f = fftshift(fft2(gray.astype(np.float32)))
    ps = np.abs(f) ** 2
    
    h, w = ps.shape
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    mask_high_freq = ((x - center_x)**2 + (y - center_y)**2) > (min(h, w) * 0.3)**2
    high_freq_energy = np.sum(ps[mask_high_freq])
    
    return float(high_freq_energy)


def extract_color_diversity(image: np.ndarray) -> float:
    """
    Calculate color diversity as ratio of unique colors to total pixels.
    Lower values indicate fewer colors (photocopies).
    """
    if len(image.shape) != 3:
        return 0.0
    
    # Sample pixels for performance
    pixels = image.reshape(-1, 3)
    sample_size = min(10000, pixels.shape[0])
    if pixels.shape[0] > sample_size:
        indices = np.random.choice(pixels.shape[0], sample_size, replace=False)
        pixels_sample = pixels[indices]
    else:
        pixels_sample = pixels
    
    # Count unique colors (with tolerance)
    unique_colors = len(np.unique(pixels_sample // 10, axis=0))
    diversity = unique_colors / pixels_sample.shape[0]
    return float(diversity)


def extract_unique_colors(image: np.ndarray) -> int:
    """
    Count number of unique colors in the image.
    """
    if len(image.shape) != 3:
        return 0
    
    # Sample for performance
    pixels = image.reshape(-1, 3)
    sample_size = min(50000, pixels.shape[0])
    if pixels.shape[0] > sample_size:
        indices = np.random.choice(pixels.shape[0], sample_size, replace=False)
        pixels_sample = pixels[indices]
    else:
        pixels_sample = pixels
    
    unique_colors = len(np.unique(pixels_sample // 5, axis=0))
    return int(unique_colors)


def extract_saturation(image: np.ndarray) -> float:
    """
    Calculate average saturation of the image.
    """
    if len(image.shape) != 3:
        return 0.0
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    return float(saturation)


def extract_texture_uniformity(image: np.ndarray) -> float:
    """
    Calculate texture uniformity using Local Binary Pattern variance.
    Lower values indicate more uniform texture (photocopies).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Downsample for speed
    h, w = gray.shape
    if h > 128 or w > 128:
        scale = min(128 / h, 128 / w)
        h_new, w_new = int(h * scale), int(w * scale)
        gray = cv2.resize(gray, (w_new, h_new), interpolation=cv2.INTER_AREA)
        h, w = h_new, w_new
    
    # Calculate LBP
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
    
    # Calculate histogram
    hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-6)
    
    # Uniformity (inverse of entropy)
    uniformity = np.sum(hist**2)
    return float(uniformity)


def extract_compression_artifacts(image: np.ndarray) -> float:
    """
    Detect compression artifacts (blocking, ringing).
    Higher values indicate compression artifacts (screenshots).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Downsample for speed
    h, w = gray.shape
    if h > 256 or w > 256:
        scale = min(256 / h, 256 / w)
        h_new, w_new = int(h * scale), int(w * scale)
        gray = cv2.resize(gray, (w_new, h_new), interpolation=cv2.INTER_AREA)
    
    # Detect block artifacts (8x8 blocks typical for JPEG)
    block_size = 8
    artifacts = 0.0
    
    for i in range(0, gray.shape[0] - block_size, block_size):
        for j in range(0, gray.shape[1] - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            # Check for blocking (sudden changes at block boundaries)
            if i + block_size < gray.shape[0]:
                edge_diff = np.abs(np.mean(block[-1, :]) - np.mean(gray[i+block_size, j:j+block_size]))
                artifacts += edge_diff
            if j + block_size < gray.shape[1]:
                edge_diff = np.abs(np.mean(block[:, -1]) - np.mean(gray[i:i+block_size, j+block_size]))
                artifacts += edge_diff
    
    return float(artifacts)


def extract_histogram_entropy(image: np.ndarray) -> float:
    """
    Calculate histogram entropy (measure of information content).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / (hist.sum() + 1e-6)
    
    # Calculate entropy
    hist_nonzero = hist[hist > 0]
    entropy = -np.sum(hist_nonzero * np.log2(hist_nonzero + 1e-10))
    return float(entropy)


def extract_dynamic_range(image: np.ndarray) -> float:
    """
    Calculate dynamic range (difference between max and min pixel values).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    min_val = np.min(gray)
    max_val = np.max(gray)
    dynamic_range = max_val - min_val
    return float(dynamic_range)


def extract_brightness(image: np.ndarray) -> float:
    """
    Calculate average brightness of the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    brightness = np.mean(gray)
    return float(brightness)


def extract_all_metrics(image: np.ndarray) -> Dict[str, float]:
    """
    Extract all 15 metrics from a CDP image.
    
    Args:
        image: CDP image in BGR format (OpenCV format)
    
    Returns:
        Dictionary with all 15 metrics
    """
    metrics = {
        'Sharpness': extract_sharpness(image),
        'Contrast': extract_contrast(image),
        'HistogramPeak': extract_histogram_peak(image),
        'EdgeDensity': extract_edge_density(image),
        'EdgeStrength': extract_edge_strength(image),
        'NoiseLevel': extract_noise_level(image),
        'HighFreqEnergy': extract_high_freq_energy(image),
        'ColorDiversity': extract_color_diversity(image),
        'UniqueColors': extract_unique_colors(image),
        'Saturation': extract_saturation(image),
        'TextureUniformity': extract_texture_uniformity(image),
        'CompressionArtifacts': extract_compression_artifacts(image),
        'HistogramEntropy': extract_histogram_entropy(image),
        'DynamicRange': extract_dynamic_range(image),
        'Brightness': extract_brightness(image)
    }
    
    return metrics

