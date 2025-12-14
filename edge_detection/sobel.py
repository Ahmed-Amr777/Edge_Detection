"""
Sobel Edge Detection Implementation
From scratch using NumPy only.
"""

import numpy as np
from typing import Tuple, Optional
from .utils import apply_gaussian_blur, normalize_image, rgb_to_grayscale, convolve2d


def sobel_edge_detection(
    image: np.ndarray,
    blur: bool = True,
    kernel_size: int = 5,
    sigma: float = 1.0,
    threshold: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Sobel edge detection to an image.
    
    Args:
        image: Input image (grayscale or RGB)
        blur: Whether to apply Gaussian blur before edge detection
        kernel_size: Size of Gaussian kernel if blur is True
        sigma: Standard deviation for Gaussian kernel
        threshold: Optional threshold value for binary output (0-255)
    
    Returns:
        Tuple of (magnitude, gradient_x, gradient_y)
        - magnitude: Edge magnitude map
        - gradient_x: Gradient in x-direction
        - gradient_y: Gradient in y-direction
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = rgb_to_grayscale(image)
    else:
        gray = image.copy()
    
    # Convert to float for calculations
    gray = gray.astype(np.float64)
    
    # Apply Gaussian blur if requested
    if blur:
        gray = apply_gaussian_blur(gray, kernel_size, sigma)
    
    # Sobel kernels (using 0.5, -0.5, 0 pattern)
    sobel_x = np.array([
        [-0.5, 0, 0.5],
        [-1, 0, 1],
        [-0.5, 0, 0.5]
    ], dtype=np.float64)
    
    sobel_y = np.array([
        [-0.5, -1, -0.5],
        [0, 0, 0],
        [0.5, 1, 0.5]
    ], dtype=np.float64)
    
    # Calculate gradients
    gradient_x = convolve2d(gray, sobel_x)
    gradient_y = convolve2d(gray, sobel_y)
    
    # Calculate magnitude
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize to 0-255 range
    magnitude = normalize_image(magnitude)
    gradient_x = normalize_image(gradient_x)
    gradient_y = normalize_image(gradient_y)
    
    # Apply threshold if provided
    if threshold is not None:
        magnitude = (magnitude > threshold).astype(np.uint8) * 255
    
    return magnitude, gradient_x, gradient_y

