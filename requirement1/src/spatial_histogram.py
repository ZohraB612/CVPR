"""
Spatial histogram computation module.
Implements spatial grid histogram extraction combining color and texture features.

Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

# Import packages
import cv2
import numpy as np

def compute_spatial_histogram(img, r_bins=8, g_bins=8, b_bins=8, grid_size=(2, 2)):
    """
    Compute spatial color histogram for an image.
    
    Args:
        img: Input image (BGR format)
        r_bins: number of bins for red channel
        g_bins: number of bins for green channel
        b_bins: number of bins for blue channel
        grid_size: tuple of (rows, cols) for spatial grid
    
    Returns:
        np.ndarray: flattened normalized histogram
    """
    height, width = img.shape[:2]
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]
    
    # Initialize the final histogram
    total_bins = r_bins * g_bins * b_bins
    spatial_hist = np.zeros(total_bins * grid_size[0] * grid_size[1])
    
    # Compute histogram for each cell
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Get cell coordinates
            y_start = i * cell_height
            y_end = (i + 1) * cell_height if i < grid_size[0] - 1 else height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width if j < grid_size[1] - 1 else width
            
            # Extract cell
            cell = img[y_start:y_end, x_start:x_end]
            
            # Compute histogram for cell
            cell_hist = cv2.calcHist(
                [cell], 
                [0, 1, 2], 
                None, 
                [b_bins, g_bins, r_bins], 
                [0, 256, 0, 256, 0, 256]
            )
            
            # Normalize cell histogram
            cell_hist = cell_hist.flatten()
            cell_hist = cell_hist / (cell_hist.sum() + 1e-7)  # Add small constant to avoid division by zero
            
            # Add to spatial histogram
            idx_start = (i * grid_size[1] + j) * total_bins
            idx_end = idx_start + total_bins
            spatial_hist[idx_start:idx_end] = cell_hist
    
    return spatial_hist

def compute_texture_features(gray_cell, gabor_filters):
    """This is a function that is used to compute texture features for a cell."""
    texture_features = []
    for gabor_kernel in gabor_filters:
        filtered = cv2.filter2D(gray_cell, cv2.CV_8UC3, gabor_kernel)
        texture_features.extend([
            np.mean(filtered),
            np.std(filtered)
        ])
    return np.array(texture_features)