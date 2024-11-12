"""
This file contains utility functions for the image retrieval system.

Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

import os
from config.settings import IMAGE_FOLDER, IMAGE_PATH, BASE_PATH

def get_image_class(filename):
    """
    This function extracts the class ID from an image filename.
    
    Args:
        filename: Path to image file (e.g., '3_1_s.bmp')
        
    Returns:
        str: Class ID (e.g., '3')
    """
    return os.path.basename(filename).split('_')[0]

def check_requirements():
    """
    This function checks if all required files and directories exist.
    """
    print("\nDebug Information:")
    print(f"Base Path: {BASE_PATH}")
    print(f"Image Path: {IMAGE_PATH}")
    print(f"Base Path exists?: {os.path.exists(BASE_PATH)}")
    print(f"Image Path exists?: {os.path.exists(IMAGE_PATH)}")
    
    if not os.path.exists(BASE_PATH):
        print(f"Error! Can't find base directory at: {BASE_PATH}")
        return False
        
    if not os.path.exists(IMAGE_PATH):
        print(f"Error! can't find Images directory at: {IMAGE_PATH}")
        print("Check that the dataset structure is correct:")
        print("  - MSRC_ObjCategImageDatabase_v2/")
        print("    |- Images/")
        print("       |- *.bmp files")
        return False
        
    # we check the Images directory for .bmp files
    image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.bmp')]
    if not image_files:
        print(f"Error! No .bmp files found in {IMAGE_PATH}")
        return False
        
    print(f"Found {len(image_files)} .bmp files")
    return True

def create_results_directory(config, timestamp):
    """
    We use this function to create (if it doesn't exist) and return the path to the results directory.
    
    Args:
        config: quantisation configuration dict
        
    Returns:
        str: path to the results directory
    """
    results_dir = f"results/results_R{config['r']}G{config['g']}B{config['b']}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def find_query_image(image_path, query_class):
    """
    Find first image of given class in the dataset.
    
    Args:
        image_path: Path to image directory
        query_class: Class name to search for
        
    Returns:
        str: Path to first image of query class, or None if not found
    """
    # Get all image files
    image_files = [f for f in os.listdir(image_path) if f.endswith('.bmp')]
    
    # Map of class names to their numeric prefixes
    class_prefixes = {
        'building': '3',
        'grass': '1',
        'tree': '3',
        'cow': '4',
        'sheep': '5',
        'sky': '6',
        'airplane': '7',
        'water': '8',
        'face': '9',
        'car': '10',
        'bicycle': '11',
        'flower': '12',
        'sign': '13',
        'bird': '14',
        'book': '15',
        'chair': '16',
        'road': '17',
        'cat': '18',
        'dog': '19',
        'body': '20'
    }
    
    # Get prefix for query class
    prefix = class_prefixes.get(query_class)
    if not prefix:
        print(f"Warning: Unknown query class {query_class}")
        return None
    
    # Find first image with matching prefix
    for filename in image_files:
        if filename.startswith(f"{prefix}_"):
            return os.path.join(image_path, filename)
    
    print(f"Warning: No images found for class {query_class}")
    return None