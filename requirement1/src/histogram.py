"""
Colour histogram computation module.
Implements global color histogram extraction with variable quantization levels.

Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

# Import packages
import cv2
import numpy as np

def compute_global_histogram(img, r_bins=8, g_bins=8, b_bins=4):
    """
    Compute global color histogram for an image.
    
    Args:
        img: Input image (BGR format)
        r_bins: number of bins for red channel
        g_bins: number of bins for green channel
        b_bins: number of bins for blue channel
        
    Returns:
        np.ndarray: flattened normalised histogram
    """
    # compute 3D histogram
    hist = cv2.calcHist(
        [img], 
        [0, 1, 2], 
        None, 
        [b_bins, g_bins, r_bins], 
        [0, 256, 0, 256, 0, 256]
    )
    
    # normalise and flatten histogram
    hist = hist.flatten()
    hist = hist / hist.sum()
    
    return hist

def euclidean_distance(hist1, hist2):
    """
    Function to compute Euclidean distance between two histograms.
    
    Args:
        hist1: 1st histogram
        hist2: 2nd histogram
        
    Returns:
        float: euclidean distance
    """
    return np.sqrt(np.sum((hist1 - hist2) ** 2))