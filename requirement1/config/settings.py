"""
Configuration settings for the image retrieval system.
Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

import os

# Dataset paths
BASE_PATH = os.path.normpath("/home/zohrab/CVPR/MSRC_ObjCategImageDatabase_v2")
IMAGE_FOLDER = "Images"
IMAGE_PATH = os.path.join(BASE_PATH, IMAGE_FOLDER)

# Test queries for consistent evaluation
# These images will be used throughout all experiments
TEST_QUERIES = {
    'building': os.path.join(BASE_PATH, 'Images/3_1_s.bmp'),
    'street': os.path.join(BASE_PATH, 'Images/17_1_s.bmp'),
    'face': os.path.join(BASE_PATH, 'Images/6_1_s.bmp'),
    'sheep': os.path.join(BASE_PATH, 'Images/9_1_s.bmp')
}

# Different quantization configurations to test
# Each configuration specifies different bin numbers for R,G,B channels
CONFIGS = [
    # Original color histogram configurations
    {'name': 'R8G8B4', 'r': 8, 'g': 8, 'b': 4},
    {'name': 'R4G4B4', 'r': 4, 'g': 4, 'b': 4},
    {'name': 'R16G16B8', 'r': 16, 'g': 16, 'b': 8},
    {'name': 'R8G8B8', 'r': 8, 'g': 8, 'b': 8},
    {'name': 'R16G16B16', 'r': 16, 'g': 16, 'b': 16},
]

# Define spatial grid configurations
SPATIAL_CONFIGS = [
    {
        'name': '884_2x2',
        'r': 8, 'g': 8, 'b': 4,
        'grid_size': (2, 2),
        'use_spatial': True
    },
    {
        'name': '884_4x4',
        'r': 8, 'g': 8, 'b': 4,
        'grid_size': (4, 4),
        'use_spatial': True
    },
    {
        'name': '884_6x6',
        'r': 8, 'g': 8, 'b': 4,
        'grid_size': (6, 6),
        'use_spatial': True
    },
    {
        'name': '884_8x8',
        'r': 8, 'g': 8, 'b': 4,
        'grid_size': (8, 8),
        'use_spatial': True
    },
    {
        'name': '884_16x16',
        'r': 8, 'g': 8, 'b': 4,
        'grid_size': (16, 16),
        'use_spatial': True
    }
]

# PCA configurations
PCA_CONFIGS = [
    {
        'name': 'R8G8B4_PCA32',
        'r': 8, 'g': 8, 'b': 4,
        'use_pca': True,
        'pca_components': 32
    },
    {
        'name': 'R8G8B4_PCA64',
        'r': 8, 'g': 8, 'b': 4,
        'use_pca': True,
        'pca_components': 64
    }
]

# Output settings
RESULTS_BASE_DIR = 'results'
DPI = 300  # Resolution for saved images