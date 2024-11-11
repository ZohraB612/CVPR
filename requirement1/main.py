"""
Main execution script for image retrieval system using global color histograms.
Implements Requirement 1 of the EEE3032 Computer Vision coursework.

Author: Zohra Bouchamaoui
Student ID: 6848526
"""

import sys
from datetime import datetime
from tqdm import tqdm
import numpy as np
import cv2
import os
import argparse

# Local imports
from config.settings import (
    IMAGE_FOLDER, 
    IMAGE_PATH, 
    TEST_QUERIES, 
    CONFIGS,
    SPATIAL_CONFIGS
)
from src.histogram import compute_global_histogram, euclidean_distance
from src.evaluation import compute_pr_curve
from src.visualisation import save_experiment_results
from src.utils import check_requirements, get_image_class, create_results_directory
from src.spatial_histogram import compute_spatial_histogram
from src.pca_retrieval import PCARetrieval
from src.analysis import compare_pca_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Image Retrieval System')
    parser.add_argument('--spatial_grid', action='store_true',
                       help='Use spatial grid features')
    parser.add_argument('--pca_experiment', action='store_true',
                       help='Run PCA experiment')
    parser.add_argument('--pca_components', type=int, default=32,
                       help='Number of PCA components')
    parser.add_argument('--compare_pca', action='store_true',
                       help='Compare PCA results')
    return parser.parse_args()

def process_configuration(config):
    """Process a single quantization configuration."""
    # Create cache filename
    cache_file = f"cache_{config['name']}.npy"
    
    if os.path.exists(cache_file):
        print(f"Loading histograms from cache for config {config['name']}...")
        features = np.load(cache_file)
        return features
    
    print(f"Computing histograms for config {config['name']}...")
    features = []
    
    # Process each image
    image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.bmp')]
    for filename in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(IMAGE_PATH, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not load image: {img_path}")
            continue
        
        # Compute histogram based on configuration
        if config.get('use_spatial', False):
            hist = compute_spatial_histogram(
                img,
                r_bins=config['r'],
                g_bins=config['g'],
                b_bins=config['b'],
                grid_size=config['grid_size']
            )
        else:
            hist = compute_global_histogram(
                img,
                r_bins=config['r'],
                g_bins=config['g'],
                b_bins=config['b']
            )
        features.append(hist)
    
    features = np.array(features)
    np.save(cache_file, features)
    
    return features

def process_query(query_path, features, config, results_dir):
    """Process a single query image."""
    try:
        # Get all image files
        image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.bmp')]
        files = [os.path.join(IMAGE_PATH, f) for f in image_files]
        
        # Compute query image features
        query_img = cv2.imread(query_path)
        
        # Compute histogram based on configuration
        if config.get('use_spatial', False):
            query_hist = compute_spatial_histogram(
                query_img,
                r_bins=config['r'],
                g_bins=config['g'],
                b_bins=config['b'],
                grid_size=config['grid_size']
            )
        else:
            query_hist = compute_global_histogram(
                query_img,
                r_bins=config['r'],
                g_bins=config['g'],
                b_bins=config['b']
            )
        
        # Compute distances
        distances = []
        for i, feat in enumerate(features):
            dist = euclidean_distance(query_hist, feat)
            distances.append((dist, files[i]))
        
        # Sort distances
        distances_sorted = sorted(distances)  # Sort based on first element (distance)
        
        # Get query class
        query_class = get_image_class(query_path)
        
        # Compute PR curve
        pr_data = compute_pr_curve(query_class, distances_sorted, len(files))
        
        # Save results
        save_experiment_results(query_path, distances_sorted, pr_data, config, results_dir)
        
    except Exception as e:
        print(f"Error processing query {query_path}: {str(e)}")
        raise

def main():
    """Main execution function"""
    args = parse_args()
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    print("Starting image retrieval system...")
    print(f"Using dataset: {IMAGE_FOLDER}")
    
    if args.compare_pca:
        compare_pca_results()
        return
    
    # Run appropriate experiment based on arguments
    if args.pca_experiment:
        pca = PCARetrieval(n_components=args.pca_components)
        pca.run_experiment(
            image_path=IMAGE_PATH,
            results_base_dir='results'
        )
    else:
        if args.spatial_grid:
            configs_to_run = SPATIAL_CONFIGS
            print("Running spatial grid experiments")
        else:
            configs_to_run = CONFIGS
            print("Running standard histogram experiments")
        
        print(f"Testing {len(configs_to_run)} different configurations")
        
        # Process each configuration
        for config in configs_to_run:
            print(f"\nProcessing configuration: R{config['r']}G{config['g']}B{config['b']}")
            
            # Process configuration
            features = process_configuration(config)
            
            # Create results directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = create_results_directory(config, timestamp)
            
            # Process test queries
            print("\nProcessing test queries...")
            for query_name, query_path in TEST_QUERIES.items():
                print(f"Processing query: {query_name}")
                try:
                    process_query(query_path, features, config, results_dir)
                except Exception as e:
                    print(f"Error processing query {query_path}: {str(e)}")
                    continue
            
            print(f"Results saved to: {results_dir}")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()