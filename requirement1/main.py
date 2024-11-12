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
    SPATIAL_CONFIGS,
    BASE_PATH
)
from src.histogram import compute_global_histogram, euclidean_distance
from src.evaluation import compute_pr_curve
from src.visualisation import save_experiment_results
from src.utils import check_requirements, get_image_class, create_results_directory
from src.spatial_histogram import compute_spatial_histogram
from src.pca_retrieval import PCARetrieval
from src.analysis import compare_pca_results
from src.bovw_retrieval import BoVWRetrieval

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
    parser.add_argument('--bovw', action='store_true',
                       help='Run BoVW retrieval experiment')
    parser.add_argument('--codebook_size', type=int, default=1000,
                       help='Size of BoVW codebook')
    return parser.parse_args()

def process_configuration(config):
    """Process a single quantization configuration."""
    # Create cache filename
    cache_file = f"cache/cache_{config['name']}.npy"
    
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
        query_hist = compute_global_histogram(query_img, config['r'], config['g'], config['b'])
        
        # Compute distances
        distances = []
        for i, feat in enumerate(features):
            dist = euclidean_distance(query_hist, feat)
            distances.append((dist, files[i]))
        
        # Sort distances
        distances_sorted = sorted(distances)
        
        # Compute precision-recall curve
        query_class = get_image_class(query_path)
        relevant_count = sum(1 for _, path in distances_sorted 
                           if get_image_class(path) == query_class)
        
        # Calculate precision and recall at each position
        precisions = []
        recalls = []
        relevant_so_far = 0
        
        for i, (_, path) in enumerate(distances_sorted, 1):
            if get_image_class(path) == query_class:
                relevant_so_far += 1
            precision = relevant_so_far / i
            recall = relevant_so_far / relevant_count if relevant_count > 0 else 0
            precisions.append(precision)
            recalls.append(recall)
        
        # Convert to numpy arrays
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        
        # Create pr_data tuple
        pr_data = (recalls, precisions)
        
        # Save results
        save_experiment_results(query_path, distances_sorted, pr_data, config, results_dir)
        
        # Print P@10 for this query
        p10 = sum(1 for _, path in distances_sorted[:10] 
                 if get_image_class(path) == query_class) / 10
        print(f"{query_class}: P@10 = {p10:.3f}")
        
    except Exception as e:
        print(f"Error processing query {query_path}: {str(e)}")
        raise

def main():
    """
    Main function to run experiments
    """
    args = parse_args()
    
    print("\nDebug Information:")
    print(f"Base Path: {BASE_PATH}")
    print(f"Image Path: {IMAGE_PATH}")
    print(f"Base Path exists?: {os.path.exists(BASE_PATH)}")
    print(f"Image Path exists?: {os.path.exists(IMAGE_PATH)}")
    
    # Count .bmp files
    bmp_count = len([f for f in os.listdir(IMAGE_PATH) if f.endswith('.bmp')])
    print(f"Found {bmp_count} .bmp files")
    
    print("Starting image retrieval system...")
    print(f"Using dataset: {IMAGE_FOLDER}")
    
    if args.bovw:
        print("\nRunning BoVW retrieval experiment...")
        bovw_config = {
            'codebook_size': args.codebook_size,
            'detector': 'sift'
        }
        
        bovw = BoVWRetrieval(bovw_config)
        
        # Build codebook and compute features
        image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.bmp')]
        image_paths = [os.path.join(IMAGE_PATH, f) for f in image_files]
        
        try:
            bovw.build_codebook(image_paths, args.codebook_size)
            bovw.compute_bovw_features(image_paths)
            
            # Process queries
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join('results', f'bovw_{args.codebook_size}_{timestamp}')
            os.makedirs(results_dir, exist_ok=True)
            
            for query_name, query_path in TEST_QUERIES.items():
                print(f"Processing query: {query_name}")
                bovw.process_query(query_path, results_dir)
            
            print(f"\nResults saved to: {results_dir}")
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
            return
        except Exception as e:
            print(f"\nError in BoVW processing: {str(e)}")
            return
    
    elif args.pca_experiment:
        print("\nRunning PCA experiment...")
        pca = PCARetrieval(CONFIGS['default'], args.pca_components)
        
        # Process all images
        image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.bmp')]
        image_paths = [os.path.join(IMAGE_PATH, f) for f in image_files]
        pca.process_images(image_paths)
        
        # Process queries
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = create_results_directory(f'pca_{args.pca_components}_{timestamp}')
        
        for query_name, query_path in TEST_QUERIES.items():
            print(f"Processing query: {query_name}")
            pca.process_query(query_path, results_dir)
        
        print(f"\nResults saved to: {results_dir}")
        
        if args.compare_pca:
            compare_pca_results(results_dir)
    
    else:
        configs = SPATIAL_CONFIGS if args.spatial_grid else CONFIGS
        print("\nRunning histogram experiments")
        print(f"Testing {len(configs)} different configurations")
        
        for config in configs:  # Changed this line to iterate over list
            print(f"\nProcessing configuration: {config['name']}")
            
            # Process all images
            features = process_configuration(config)
            
            # Process queries
            results_dir = create_results_directory(config["name"])
            
            for query_name, query_path in TEST_QUERIES.items():
                print(f"Processing query: {query_name}")
                process_query(query_path, features, config, results_dir)
            
            print(f"Results saved to: {results_dir}\n")

if __name__ == "__main__":
    main()