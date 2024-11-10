"""
Evaluation metrics module.
Implements precision-recall curve computation and other evaluation metrics.

Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

from src.utils import get_image_class

def compute_pr_curve(query_class, distances, total_images):
    """
    Compute precision-recall curve data points.
    
    Args:
        query_class: Class ID of the query image
        distances: List of (distance, image_path) tuples, sorted by distance
        total_images: Total number of images in dataset
    
    Returns:
        tuple: (recalls, precisions) lists for plotting
        
    Note:
        - Adds (0,1) and (1,0) points to complete the curve
        - Computes precision and recall at each rank
    """
    precisions = []
    recalls = []
    
    # Count total relevant images (same class as query)
    relevant_count = sum(1 for _, path in distances 
                        if get_image_class(path) == query_class)
    
    # Compute precision and recall at each rank
    true_positives = 0
    for i, (_, path) in enumerate(distances, 1):
        if get_image_class(path) == query_class:
            true_positives += 1
        
        # Calculate metrics
        precision = true_positives / i
        recall = true_positives / relevant_count
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Add endpoints for complete PR curve
    recalls.insert(0, 0.0)
    precisions.insert(0, 1.0)
    recalls.append(1.0)
    precisions.append(0.0)
    
    return recalls, precisions

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