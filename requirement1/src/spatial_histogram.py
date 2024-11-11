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

def compute_spatial_histogram(img, r_bins=8, g_bins=8, b_bins=4, grid_size=(4, 4)):
    """
    Function used to compute spatial grid histogram by combining colour and texture features.
    
    Args:
        img: input image (BGR format)
        r_bins, g_bins, b_bins: colour quantisation bins
        grid_size: tuple of (rows, cols) for spatial grid
    
    Returns:
        np.ndarray: concatenated features from all grid cells
    """
    height, width = img.shape[:2]
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]
    
    # We initialise a list to store features from each cell from the grid
    cell_features_list = []
    
    # Convert to different color spaces
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Create Gabor filters for texture
    gabor_filters = []
    num_orientations = 4  # Reduced from 8 to save computation
    for theta in np.arange(0, np.pi, np.pi / num_orientations):
        kernel = cv2.getGaborKernel(
            ksize=(16, 16),
            sigma=4.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )
        kernel /= 1.5 * kernel.sum()
        gabor_filters.append(kernel)
    
    # next we process each grid cell
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # 1. first we define the cell boundaries
            y1 = i * cell_height
            y2 = (i + 1) * cell_height
            x1 = j * cell_width
            x2 = (j + 1) * cell_width
            
            cell_bgr = img[y1:y2, x1:x2]
            cell_hsv = hsv_img[y1:y2, x1:x2]
            cell_lab = lab_img[y1:y2, x1:x2]
            
            # 1. BGR color histogram
            hist_bgr = cv2.calcHist(
                [cell_bgr], 
                [0, 1, 2], 
                None, 
                [b_bins, g_bins, r_bins], 
                [0, 256, 0, 256, 0, 256]
            ).flatten()
            
            # 2. HSV histogram (focusing on Hue and Saturation)
            hist_hs = cv2.calcHist(
                [cell_hsv], 
                [0, 1], 
                None, 
                [8, 8], 
                [0, 180, 0, 256]
            ).flatten()
            
            # 3. Lab histogram (focusing on a and b channels)
            hist_ab = cv2.calcHist(
                [cell_lab], 
                [1, 2], 
                None, 
                [8, 8], 
                [0, 256, 0, 256]
            ).flatten()
            
            # 4. Texture features using Gabor filters
            gray_cell = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
            texture_features = []
            
            for gabor_kernel in gabor_filters:
                filtered = cv2.filter2D(gray_cell, cv2.CV_8UC3, gabor_kernel)
                texture_features.extend([
                    np.mean(filtered),
                    np.std(filtered)
                ])
            
            # 5. Edge features using Canny
            edges = cv2.Canny(gray_cell, 100, 200)
            edge_features = [
                np.mean(edges),
                np.std(edges),
                np.sum(edges > 0) / edges.size  # Edge density
            ]
            
            # Combine all features for this cell
            cell_features = np.concatenate([
                hist_bgr,
                hist_hs,
                hist_ab,
                texture_features,
                edge_features
            ])
            
            # Add to list
            cell_features_list.append(cell_features)
    
    # Concatenate all cell features
    final_features = np.concatenate(cell_features_list)
    
    # Normalize the final feature vector
    final_features = final_features / (np.sum(final_features) + 1e-7)
    
    return final_features

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