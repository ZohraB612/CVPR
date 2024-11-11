"""
Evaluation metrics module.
Implements precision-recall curve computation and other evaluation metrics.

Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

from src.utils import get_image_class
import numpy as np

def compute_pr_curve(query_class, distances, total_images):
    """
    Compute precision-recall curve.
    
    Args:
        query_class: Ground truth class of query image
        distances: List of (distance, image_path) tuples
        total_images: Total number of images
    Returns:
        dict: Dictionary containing precision and recall arrays
    """
    relevant = 0
    precision = []
    recall = []
    
    # Get total relevant (images of same class as query)
    total_relevant = sum(1 for _, img_path in distances 
                        if get_image_class(str(img_path)) == query_class)
    
    if total_relevant == 0:
        return {'precision': np.zeros(len(distances)), 
                'recall': np.zeros(len(distances))}
    
    for i, (_, img_path) in enumerate(distances, 1):
        # Check if retrieved image is of same class
        if get_image_class(str(img_path)) == query_class:
            relevant += 1
        
        precision.append(relevant / i)
        recall.append(relevant / total_relevant)
    
    return {
        'precision': np.array(precision),
        'recall': np.array(recall)
    }

def compute_average_precision(precisions, recalls):
    """
    Compute Average Precision (AP) from precision-recall values.
    
    Args:
        precisions: List of precision values
        recalls: List of recall values
    
    Returns:
        float: Average Precision score
        
    Note:
        - Implements area under PR curve using trapezoidal rule
        - Standard metric in information retrieval
    """
    ap = 0.0
    for i in range(len(recalls) - 1):
        ap += (recalls[i + 1] - recalls[i]) * (precisions[i + 1] + precisions[i]) / 2
    return ap