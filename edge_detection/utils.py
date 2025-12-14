"""
Utility functions for image processing and edge detection.
"""

import numpy as np
from typing import Tuple, Optional


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.
    
    Args:
        size: Size of the kernel (should be odd)
        sigma: Standard deviation of the Gaussian distribution
    
    Returns:
        2D numpy array representing the Gaussian kernel
    """
    if size % 2 == 0:
        size += 1
    
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize the kernel
    kernel = kernel / np.sum(kernel)
    return kernel


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    
    Args:
        image: Input grayscale image
        kernel_size: Size of the Gaussian kernel
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        Blurred image
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve2d(image, kernel)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Perform 2D convolution between image and kernel.
    
    Args:
        image: Input image
        kernel: Convolution kernel
    
    Returns:
        Convolved image
    """
    # Get dimensions
    img_h, img_w = image.shape
    kernel_h, kernel_w = kernel.shape
    
    # Calculate padding
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    # Pad the image
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # Initialize output
    output = np.zeros_like(image)
    
    # Perform convolution
    for i in range(img_h):
        for j in range(img_w):
            output[i, j] = np.sum(padded[i:i+kernel_h, j:j+kernel_w] * kernel)
    
    return output


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to 0-255 range.
    
    Args:
        image: Input image
    
    Returns:
        Normalized image
    """
    img_min = np.min(image)
    img_max = np.max(image)
    
    if img_max == img_min:
        return np.zeros_like(image)
    
    normalized = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return normalized


def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to grayscale.
    
    Args:
        image: Input RGB image (H, W, 3)
    
    Returns:
        Grayscale image (H, W)
    """
    if len(image.shape) == 2:
        return image
    
    # Standard RGB to grayscale conversion weights
    grayscale = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return grayscale.astype(np.uint8)


def orientation_to_color(orientation: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Convert edge orientation to color visualization.
    Uses HSV color space where hue represents orientation.
    
    Args:
        orientation: Gradient orientation map in radians
        edges: Binary edge map (0 or 255)
    
    Returns:
        RGB image with colored edges based on orientation
    """
    h, w = edges.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Normalize orientation to [0, 2π] then to [0, 1] for hue
    # Convert radians to degrees, normalize to [0, 360), then to [0, 1]
    orientation_deg = np.degrees(orientation) % 360
    hue = orientation_deg / 360.0  # Normalize to [0, 1]
    
    # Create HSV image
    hsv = np.zeros((h, w, 3), dtype=np.float32)
    hsv[:, :, 0] = hue  # Hue based on orientation
    hsv[:, :, 1] = 1.0  # Full saturation
    hsv[:, :, 2] = (edges > 0).astype(np.float32)  # Value: 1 for edges, 0 for background
    
    # Convert HSV to RGB
    # HSV to RGB conversion
    c = hsv[:, :, 2] * hsv[:, :, 1]  # Chroma
    x = c * (1 - np.abs((hsv[:, :, 0] * 6) % 2 - 1))
    m = hsv[:, :, 2] - c
    
    r_prime = np.zeros_like(hue)
    g_prime = np.zeros_like(hue)
    b_prime = np.zeros_like(hue)
    
    # Determine RGB based on hue sector
    h6 = hsv[:, :, 0] * 6
    sector = h6.astype(int) % 6
    
    mask0 = (sector == 0)
    mask1 = (sector == 1)
    mask2 = (sector == 2)
    mask3 = (sector == 3)
    mask4 = (sector == 4)
    mask5 = (sector == 5)
    
    r_prime[mask0] = c[mask0]
    g_prime[mask0] = x[mask0]
    b_prime[mask0] = 0
    
    r_prime[mask1] = x[mask1]
    g_prime[mask1] = c[mask1]
    b_prime[mask1] = 0
    
    r_prime[mask2] = 0
    g_prime[mask2] = c[mask2]
    b_prime[mask2] = x[mask2]
    
    r_prime[mask3] = 0
    g_prime[mask3] = x[mask3]
    b_prime[mask3] = c[mask3]
    
    r_prime[mask4] = x[mask4]
    g_prime[mask4] = 0
    b_prime[mask4] = c[mask4]
    
    r_prime[mask5] = c[mask5]
    g_prime[mask5] = 0
    b_prime[mask5] = x[mask5]
    
    # Add m to each component
    color_image[:, :, 0] = ((r_prime + m) * 255).astype(np.uint8)
    color_image[:, :, 1] = ((g_prime + m) * 255).astype(np.uint8)
    color_image[:, :, 2] = ((b_prime + m) * 255).astype(np.uint8)
    
    return color_image


