"""
Utility functions for the image retrieval system.
Provides helper functions for file operations and data processing.

Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

import os
from config.settings import IMAGE_FOLDER, IMAGE_PATH

def get_image_class(filename):
    """
    Extract class ID from image filename.
    
    Args:
        filename: Path to image file (e.g., '3_1_s.bmp')
        
    Returns:
        str: Class ID (e.g., '3')
        
    Note:
        Assumes filename format: 'class_instance_s.bmp'
    """
    return os.path.basename(filename).split('_')[0]

def check_requirements():
    """
    Check if all required files and directories exist.
    """
    from config.settings import BASE_PATH, IMAGE_PATH
    
    print("\nDebug Information:")
    print(f"Base Path: {BASE_PATH}")
    print(f"Image Path: {IMAGE_PATH}")
    print(f"Base Path exists?: {os.path.exists(BASE_PATH)}")
    print(f"Image Path exists?: {os.path.exists(IMAGE_PATH)}")
    
    if not os.path.exists(BASE_PATH):
        print(f"Error: Base directory not found at: {BASE_PATH}")
        return False
        
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Images directory not found at: {IMAGE_PATH}")
        print("Please ensure the dataset structure is correct:")
        print("  - MSRC_ObjCategImageDatabase_v2/")
        print("    |- Images/")
        print("       |- *.bmp files")
        return False
        
    # Check if there are any .bmp files in the Images directory
    image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.bmp')]
    if not image_files:
        print(f"Error: No .bmp files found in {IMAGE_PATH}")
        return False
        
    print(f"Found {len(image_files)} .bmp files")
    return True

def create_results_directory(config, timestamp):
    """
    Create and return path to results directory.
    
    Args:
        config: Quantization configuration dictionary
        timestamp: Current timestamp string
        
    Returns:
        str: Path to results directory
        
    Note:
        Creates directory structure: results_R{r}G{g}B{b}_{timestamp}
    """
    results_dir = f"results_R{config['r']}G{config['g']}B{config['b']}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir