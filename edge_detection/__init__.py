"""
Edge Detection Package
A comprehensive implementation of edge detection algorithms from scratch using NumPy.
"""

from .sobel import sobel_edge_detection
from .canny import canny_edge_detection
from .prewitt import prewitt_edge_detection

__all__ = ['sobel_edge_detection', 'canny_edge_detection', 'prewitt_edge_detection']


