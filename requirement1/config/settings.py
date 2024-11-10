"""
Configuration settings for the image retrieval system.
Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

import os

# Dataset paths
BASE_PATH = os.path.normpath("/Users/zohrab/Desktop/2024/University of Surrey/Semester 1/CVPR/Skeleton_Python_revised/MSRC_ObjCategImageDatabase_v2")
IMAGE_FOLDER = "Images"
IMAGE_PATH = os.path.join(BASE_PATH, IMAGE_FOLDER)

# Test queries for consistent evaluation
# These images will be used throughout all experiments
TEST_QUERIES = {
    'building': os.path.join(BASE_PATH, 'Images/3_1_s.bmp'),
    'face': os.path.join(BASE_PATH, 'Images/17_1_s.bmp'),
    'sheep': os.path.join(BASE_PATH, 'Images/6_1_s.bmp'),
    'street': os.path.join(BASE_PATH, 'Images/9_1_s.bmp')
}

# Different quantization configurations to test
# Each configuration specifies different bin numbers for R,G,B channels
CONFIGS = [
    {'r': 8, 'g': 8, 'b': 4, 'name': '884'},   # Standard with less blue sensitivity
    {'r': 4, 'g': 8, 'b': 4, 'name': '484'},   # More green sensitivity (human eye most sensitive to green)
    {'r': 16, 'g': 16, 'b': 8, 'name': '16168'},# Higher detail
    {'r': 4, 'g': 4, 'b': 4, 'name': '444'},    # Basic configuration (baseline)
    {'r': 8, 'g': 8, 'b': 8, 'name': '888'}, 
    {'r': 16, 'g': 16, 'b': 16, 'name': '161616'} 
]

# Output settings
RESULTS_BASE_DIR = 'results'
DPI = 300  # Resolution for saved images