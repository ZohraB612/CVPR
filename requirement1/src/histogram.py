"""
Colour histogram computation module.
Implements global color histogram extraction with variable quantization levels.

Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

import cv2
import numpy as np

def compute_global_histogram(img, r_bins=8, g_bins=8, b_bins=4):
    """
    Compute global color histogram for an image.
    
    Args:
        img: Input image (BGR format)
        r_bins: Number of bins for red channel
        g_bins: Number of bins for green channel
        b_bins: Number of bins for blue channel
        
    Returns:
        np.ndarray: Flattened normalized histogram
    """
    # Compute 3D histogram
    hist = cv2.calcHist(
        [img], 
        [0, 1, 2], 
        None, 
        [b_bins, g_bins, r_bins], 
        [0, 256, 0, 256, 0, 256]
    )
    
    # Normalize and flatten histogram
    hist = hist.flatten()
    hist = hist / hist.sum()
    
    return hist

def euclidean_distance(hist1, hist2):
    """
    Compute Euclidean distance between two histograms.
    
    Args:
        hist1: First histogram
        hist2: Second histogram
        
    Returns:
        float: Euclidean distance
    """
    return np.sqrt(np.sum((hist1 - hist2) ** 2))