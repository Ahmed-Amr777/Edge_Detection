"""
Canny Edge Detection Implementation
From scratch using NumPy only.
"""

import numpy as np
from typing import Tuple, Optional, Union
from .utils import apply_gaussian_blur, normalize_image, rgb_to_grayscale, convolve2d


def canny_edge_detection(
    image: np.ndarray,
    low_threshold: float = 50,
    high_threshold: float = 150,
    kernel_size: int = 5,
    sigma: float = 1.0,
    return_orientation: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply Canny edge detection to an image.
    
    Args:
        image: Input image (grayscale or RGB)
        low_threshold: Lower threshold for hysteresis
        high_threshold: Upper threshold for hysteresis
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation for Gaussian kernel
        return_orientation: If True, also return orientation map
    
    Returns:
        Binary edge map (0 or 255), or tuple of (edges, orientation) if return_orientation=True
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = rgb_to_grayscale(image)
    else:
        gray = image.copy()
    
    # Convert to float for calculations
    gray = gray.astype(np.float64)
    
    # Step 1: Apply Gaussian blur
    blurred = apply_gaussian_blur(gray, kernel_size, sigma)
    
    # Step 2: Calculate gradients using Sobel operators
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float64)
    
    gradient_x = convolve2d(blurred, sobel_x)
    gradient_y = convolve2d(blurred, sobel_y)
    
    # Calculate magnitude and direction
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    direction = np.arctan2(gradient_y, gradient_x)
    
    # Step 3: Non-maximum suppression
    suppressed = non_maximum_suppression(magnitude, direction)
    
    # Step 4: Double thresholding and hysteresis
    edges = hysteresis_thresholding(suppressed, low_threshold, high_threshold)
    
    edges = edges.astype(np.uint8)
    
    if return_orientation:
        # Create orientation map (only for edge pixels)
        orientation_map = np.zeros_like(direction, dtype=np.float32)
        orientation_map[edges == 255] = direction[edges == 255]
        return edges, orientation_map
    
    return edges


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Apply non-maximum suppression to thin edges.
    Uses all 8 directions (including diagonals).
    
    Args:
        magnitude: Gradient magnitude
        direction: Gradient direction in radians
    
    Returns:
        Suppressed magnitude
    """
    h, w = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    
    # Convert direction to degrees and normalize to [0, 180)
    angle = np.degrees(direction) % 180
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Determine neighbors based on gradient direction (all 8 directions)
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] < 180):
                # Horizontal edge
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elif 22.5 <= angle[i, j] < 67.5:
                # Diagonal (top-left to bottom-right)
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            elif 67.5 <= angle[i, j] < 112.5:
                # Vertical edge
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            else:  # 112.5 <= angle[i, j] < 157.5
                # Diagonal (top-right to bottom-left)
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            
            # Keep pixel if it's a local maximum
            if magnitude[i, j] >= max(neighbors):
                suppressed[i, j] = magnitude[i, j]
    
    return suppressed


def double_threshold(
    magnitude: np.ndarray,
    low_threshold: float,
    high_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply double thresholding to classify edges as strong or weak.
    
    Args:
        magnitude: Gradient magnitude after non-maximum suppression
        low_threshold: Lower threshold
        high_threshold: Upper threshold
    
    Returns:
        Tuple of (edges, weak_edges_mask)
        - edges: Binary edge map with strong edges marked as 255
        - weak_edges_mask: Boolean mask of weak edge pixels
    """
    edges = np.zeros_like(magnitude)
    
    # Strong edges (above high threshold)
    strong_edges = magnitude >= high_threshold
    # Weak edges (between low and high threshold)
    weak_edges = (magnitude >= low_threshold) & (magnitude < high_threshold)
    
    # Mark strong edges
    edges[strong_edges] = 255
    
    return edges, weak_edges


def connect_weak_edges(
    edges: np.ndarray,
    weak_edges: np.ndarray
) -> np.ndarray:
    """
    Connect weak edges to strong edges using 8-connected neighbors with BFS.
    Uses Breadth-First Search to explore connected weak edge components.
    
    Args:
        edges: Binary edge map with strong edges already marked (255)
        weak_edges: Boolean mask of weak edge pixels
    
    Returns:
        Binary edge map (0 or 255) with connected weak edges
    """
    # ===== OLD IMPLEMENTATION (Top-to-bottom, left-to-right with stack) =====
    # h, w = edges.shape
    # 
    # # Stack for deferred weak edges
    # stack = []
    # 
    # # Process from top to bottom, left to right
    # for i in range(1, h - 1):
    #     for j in range(1, w - 1):
    #         if weak_edges[i, j]:
    #             # Check 8 neighbors: up, down, left, right, and 4 diagonals
    #             neighbors_8 = [
    #                 (i-1, j-1),  # top-left
    #                 (i-1, j),    # up
    #                 (i-1, j+1),  # top-right
    #                 (i, j-1),    # left
    #                 (i, j+1),    # right
    #                 (i+1, j-1),  # bottom-left
    #                 (i+1, j),    # down
    #                 (i+1, j+1)  # bottom-right
    #             ]
    #             
    #             # Check if any neighbor is a strong edge (255)
    #             has_strong_neighbor = False
    #             has_weak_neighbor = False
    #             
    #             for ni, nj in neighbors_8:
    #                 # Check bounds
    #                 if ni < 0 or ni >= h or nj < 0 or nj >= w:
    #                     continue
    #                 
    #                 if edges[ni, nj] == 255:
    #                     # Found strong edge neighbor
    #                     has_strong_neighbor = True
    #                     break
    #                 elif weak_edges[ni, nj] and edges[ni, nj] == 0:
    #                     # Found another weak edge neighbor
    #                     has_weak_neighbor = True
    #             
    #             if has_strong_neighbor:
    #                 # Mark as edge (255) if connected to strong edge
    #                 edges[i, j] = 255
    #             elif not has_weak_neighbor:
    #                 # No strong neighbors and no weak neighbors nearby - mark as 0
    #                 edges[i, j] = 0
    #             else:
    #                 # Has weak neighbor but no strong neighbor - add to stack for later
    #                 stack.append((i, j))
    # 
    # # Process stack: check weak edges that were deferred
    # while stack:
    #     i, j = stack.pop(0)  # Process from front (FIFO)
    #     
    #     # Skip if already processed
    #     if edges[i, j] != 0:
    #         continue
    #     
    #     # Check 8 neighbors again
    #     neighbors_8 = [
    #         (i-1, j-1),  # top-left
    #         (i-1, j),    # up
    #         (i-1, j+1),  # top-right
    #         (i, j-1),    # left
    #         (i, j+1),    # right
    #         (i+1, j-1),  # bottom-left
    #         (i+1, j),    # down
    #         (i+1, j+1)  # bottom-right
    #     ]
    #     
    #     has_strong_neighbor = False
    #     has_weak_neighbor = False
    #     
    #     for ni, nj in neighbors_8:
    #         # Check bounds
    #         if ni < 0 or ni >= h or nj < 0 or nj >= w:
    #             continue
    #         
    #         if edges[ni, nj] == 255:
    #             # Found strong edge neighbor
    #             has_strong_neighbor = True
    #             break
    #         elif weak_edges[ni, nj] and edges[ni, nj] == 0:
    #             # Found another weak edge neighbor
    #             has_weak_neighbor = True
    #     
    #     if has_strong_neighbor:
    #         # Mark as edge (255) if connected to strong edge
    #         edges[i, j] = 255
    #     elif not has_weak_neighbor:
    #         # No strong neighbors and no weak neighbors nearby - mark as 0
    #         edges[i, j] = 0
    #     # If still has weak neighbor but no strong, it stays 0 (will be checked in next iteration if needed)
    # 
    # return edges
    # ===== END OLD IMPLEMENTATION =====
    
    # ===== NEW IMPLEMENTATION (BFS) =====
    h, w = edges.shape
    visited = np.zeros_like(edges, dtype=bool)
    
    # 8-connected neighbors
    neighbors_8 = [
        (-1, -1),  # top-left
        (-1, 0),   # up
        (-1, 1),   # top-right
        (0, -1),   # left
        (0, 1),    # right
        (1, -1),   # bottom-left
        (1, 0),    # down
        (1, 1)     # bottom-right
    ]
    
    def bfs_connected_component(start_i, start_j):
        """
        Use BFS to explore a connected component of weak edges.
        Returns True if any pixel in the component is connected to a strong edge.
        """
        from collections import deque
        queue = deque([(start_i, start_j)])
        component = []
        connected_to_strong = False
        
        while queue:
            i, j = queue.popleft()  # FIFO for BFS
            
            # Skip if already visited or not a weak edge
            if visited[i, j] or not weak_edges[i, j]:
                continue
            
            visited[i, j] = True
            component.append((i, j))
            
            # Check all 8 neighbors
            for di, dj in neighbors_8:
                ni, nj = i + di, j + dj
                
                # Check bounds
                if ni < 0 or ni >= h or nj < 0 or nj >= w:
                    continue
                
                # If neighbor is a strong edge, mark this component as connected
                if edges[ni, nj] == 255:
                    connected_to_strong = True
                
                # If neighbor is an unvisited weak edge, add to queue
                if weak_edges[ni, nj] and not visited[ni, nj]:
                    queue.append((ni, nj))
        
        # If component is connected to strong edge, mark all pixels as 255
        if connected_to_strong:
            for i, j in component:
                edges[i, j] = 255
        else:
            # Not connected to strong edge, mark as 0
            for i, j in component:
                edges[i, j] = 0
        
        return connected_to_strong
    
    # Process all weak edges using BFS
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if weak_edges[i, j] and not visited[i, j]:
                # Start BFS from this unvisited weak edge
                bfs_connected_component(i, j)
    
    return edges


def hysteresis_thresholding(
    magnitude: np.ndarray,
    low_threshold: float,
    high_threshold: float
) -> np.ndarray:
    """
    Apply double thresholding and hysteresis to connect edges.
    Combines double_threshold and connect_weak_edges.
    
    Args:
        magnitude: Gradient magnitude after non-maximum suppression
        low_threshold: Lower threshold
        high_threshold: Upper threshold
    
    Returns:
        Binary edge map (0 or 255)
    """
    # Step 1: Apply double threshold
    edges, weak_edges = double_threshold(magnitude, low_threshold, high_threshold)
    
    # Step 2: Connect weak edges to strong edges
    edges = connect_weak_edges(edges, weak_edges)
    
    return edges

