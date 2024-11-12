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

def manhattan_distance(hist1, hist2):
    """
    Function to compute Manhattan (L1) distance between two histograms.
    
    Args:
        hist1: 1st histogram
        hist2: 2nd histogram
        
    Returns:
        float: manhattan distance
    """
    return np.sum(np.abs(hist1 - hist2))

def chi_square_distance(hist1, hist2):
    """
    Function to compute Chi-Square distance between two histograms.
    
    Args:
        hist1: 1st histogram
        hist2: 2nd histogram
        
    Returns:
        float: chi-square distance
    """
    eps = 1e-10  # small constant to avoid division by zero
    return np.sum((hist1 - hist2) ** 2 / (hist1 + hist2 + eps))

def intersection_distance(hist1, hist2):
    """
    Function to compute Histogram Intersection distance.
    Lower values indicate more similarity.
    
    Args:
        hist1: 1st histogram
        hist2: 2nd histogram
        
    Returns:
        float: intersection distance
    """
    return 1.0 - np.sum(np.minimum(hist1, hist2))

def mahalanobis_distance(hist1, hist2):
    """
    Function to compute Mahalanobis distance between two histograms.
    Note: This implementation treats the histograms as observations and 
    computes a simple covariance. For more accurate results with multiple
    samples, consider pre-computing the covariance matrix from a training set.
    
    Args:
        hist1: 1st histogram
        hist2: 2nd histogram
        
    Returns:
        float: mahalanobis distance
    """
    # Reshape histograms to 2D arrays for covariance calculation
    X = np.vstack([hist1, hist2])
    
    # Compute covariance matrix
    # Add small constant to diagonal for numerical stability
    eps = 1e-8
    covariance = np.cov(X.T) + np.eye(len(hist1)) * eps
    
    try:
        # Compute inverse of covariance matrix
        inv_covariance = np.linalg.inv(covariance)
        
        # Compute difference vector
        diff = hist1 - hist2
        
        # Compute Mahalanobis distance
        dist = np.sqrt(diff.dot(inv_covariance).dot(diff))
        return dist
    except np.linalg.LinAlgError:
        # If covariance matrix is singular, fall back to Euclidean distance
        print("Warning: Covariance matrix is singular, falling back to Euclidean distance")
        return euclidean_distance(hist1, hist2)