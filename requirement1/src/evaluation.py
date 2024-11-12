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
    Compute Average Precision (AP) using the formula:
    AP = sum(P(n) × rel(n)) / # relevant documents
    
    Where:
    - P(n) is the precision at rank n
    - rel(n) is 1 if result n is relevant, 0 otherwise
    
    Args:
        precisions: List of precision values
        recalls: List of recall values
    
    Returns:
        float: Average Precision score
    """
    # Get the changes in recall - when recall changes, it means we found a relevant document
    recall_changes = np.diff(recalls, prepend=0)
    # Number of relevant documents is the final recall × length (since recall = relevant/total_relevant)
    num_relevant = int(recalls[-1] * len(recalls))
    
    if num_relevant == 0:
        return 0.0
        
    # Sum precision values where recall changes (i.e., where we found relevant documents)
    ap = np.sum(precisions[recall_changes > 0]) / num_relevant
    
    return ap