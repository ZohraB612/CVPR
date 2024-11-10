"""
Main execution script for image retrieval system using global color histograms.
Implements Requirement 1 of the EEE3032 Computer Vision coursework.

This script:
1. Processes images with different color quantization levels
2. Computes and caches global color histograms
3. Performs image retrieval using Euclidean distance
4. Generates comprehensive visualizations and analysis

Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

import sys
from datetime import datetime
from tqdm import tqdm
import numpy as np
import cv2
import os
from src.histogram import compute_global_histogram  

# Local imports
from config.settings import (
    IMAGE_FOLDER, 
    IMAGE_PATH, 
    TEST_QUERIES, 
    CONFIGS
)
from src.histogram import compute_global_histogram, euclidean_distance
from src.evaluation import compute_pr_curve
from src.visualisation import save_experiment_results
from src.utils import check_requirements, get_image_class, create_results_directory

def process_configuration(config):
    """
    Process a single quantization configuration.
    
    Args:
        config: Dictionary containing r,g,b bin counts and name
        
    Returns:
        tuple: (features, files) or None if error occurs
    """
    # Create cache file in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CACHE_FILE = os.path.join(script_dir, f'histogram_cache_{config["name"]}.npz')
    
    try:
        # Check for cached histograms
        if os.path.exists(CACHE_FILE):
            print(f"Loading histograms from cache for config {config['name']}...")
            data = np.load(CACHE_FILE, allow_pickle=True)
            return data['features'], list(data['files'])
        
        # Compute new histograms
        print(f"Computing histograms for config {config['name']}...")
        features = []
        files = []
        
        # Get all image files
        image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.bmp')]
        if not image_files:
            raise ValueError(f"No image files found in '{IMAGE_PATH}'!")
        
        # Process each image
        for filename in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(IMAGE_PATH, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not load image: {img_path}")
                continue
            
            # Compute histogram with specified quantization
            hist = compute_global_histogram(
                img,
                r_bins=config['r'],
                g_bins=config['g'],
                b_bins=config['b']
            )
            
            files.append(img_path)
            features.append(hist)
        
        if not features:
            raise ValueError("No valid images processed!")
        
        # Convert to numpy array and cache
        features = np.array(features)
        print("Saving histograms to cache...")
        np.savez(CACHE_FILE, features=features, files=files)
        
        return features, files
        
    except Exception as e:
        print(f"Error processing configuration {config['name']}: {str(e)}")
        return None

def process_query(query_path, features, files, config, results_dir):
    """
    Process a single query image.
    
    Args:
        query_path: Path to query image
        features: Array of precomputed features
        files: List of file paths
        config: Current configuration
        results_dir: Directory to save results
    """
    try:
        # Convert files to a list if it's not already
        files_list = files.tolist() if isinstance(files, np.ndarray) else files
        
        # Get query index and class
        query_idx = files_list.index(query_path)
        query_class = get_image_class(query_path)
        
        # Compute distances to all images
        distances = []
        for i, feat in enumerate(features):
            dist = euclidean_distance(features[query_idx], feat)
            distances.append((dist, files_list[i]))
        
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        
        # Compute PR curve
        pr_data = compute_pr_curve(query_class, distances, len(files))
        
        # Save results
        save_experiment_results(query_path, distances, pr_data, config, results_dir)
        
    except Exception as e:
        print(f"Error processing query {query_path}: {str(e)}")
        raise  # Add this to see the full error traceback

def main():
    """Main execution function"""
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    print("Starting image retrieval system...")
    print(f"Using dataset: {IMAGE_FOLDER}")
    print(f"Testing {len(CONFIGS)} different quantization configurations")
    
    # Process each configuration
    for config in CONFIGS:
        print(f"\nProcessing configuration: R{config['r']}G{config['g']}B{config['b']}")
        
        # Process configuration
        result = process_configuration(config)
        if result is None:
            continue
            
        features, files = result
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = create_results_directory(config, timestamp)
        
        # Process each test query
        print("\nProcessing test queries...")
        for query_name, query_path in TEST_QUERIES.items():
            print(f"Processing query: {query_name}")
            process_query(query_path, features, files, config, results_dir)
        
        print(f"Results saved to: {results_dir}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()