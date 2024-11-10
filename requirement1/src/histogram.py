"""
Colour histogram computation module.
Implements global color histogram extraction with variable quantization levels.

Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

import cv2
import numpy as np

def compute_global_histogram(image, r_bins=8, g_bins=8, b_bins=4):
    """
    This function computes a global color histogram with different quantization per channel.
    
    Args:
        image: BGR image (OpenCV format)
        r_bins: Number of bins for Red channel
        g_bins: Number of bins for Green channel
        b_bins: Number of bins for Blue channel
        
    Returns:
        normalized flattened histogram
        
    Note:
        - Uses different quantization levels for each channel
        - normalizes histogram to make it scale-invariant
    """
    # Calculate 3D histogram with specified bins per channel
    hist = cv2.calcHist([image], [0, 1, 2], None, 
                        [r_bins, g_bins, b_bins], 
                        [0, 256, 0, 256, 0, 256])
    
    # normalize histogram to make it scale-invariant
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def euclidean_distance(hist1, hist2):
    """
    Compute Euclidean distance between two histograms.
    
    Args:
        hist1, hist2: Normalized histograms to compare
        
    Returns:
        Float: Euclidean distance between histograms
        
    Note:
        - Used as similarity measure between images
        - Lower distance indicates more similar images
    """
    return np.sqrt(np.sum((hist1 - hist2) ** 2))