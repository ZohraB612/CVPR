"""
Script to run image classification experiments using multiple approaches:
1. SVM with different features (BoVW, Spatial Histogram, Region-based)
2. CNN with transfer learning

Author: Zohra Bouchamaoui
Student ID: 6848526
"""

import os
from datetime import datetime
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from src.bovw_retrieval import BoVWRetrieval
from src.spatial_histogram import compute_spatial_histogram
from src.region_features import RegionFeatureExtractor
from src.image_classifier import evaluate_classifier
from src.cnn_classifier import CNNClassifier
from config.settings import (
    IMAGE_FOLDER, 
    IMAGE_PATH, 
    CONFIGS,
    SPATIAL_CONFIGS
)

def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification System')
    parser.add_argument('--classifier', type=str, 
                       choices=['svm', 'cnn', 'both'],
                       default='both', help='Classifier type to use')
    parser.add_argument('--feature_type', type=str, 
                       choices=['bovw', 'spatial', 'region', 'all'],
                       default='spatial', help='Feature type for SVM')
    parser.add_argument('--model_name', type=str,
                       choices=['resnet18', 'resnet50', 'vgg16'],
                       default='resnet18', help='CNN model to use')
    return parser.parse_args()

def run_svm_classification(image_paths, feature_type, results_dir):
    """Run SVM classification with specified feature type."""
    print(f"\nRunning SVM classification with {feature_type} features...")
    
    if feature_type == 'bovw':
        bovw_config = {
            'codebook_size': 1000,
            'detector': 'sift'
        }
        bovw = BoVWRetrieval(bovw_config)
        classifier = evaluate_classifier(
            feature_extractor=bovw,
            image_paths=image_paths,
            results_dir=os.path.join(results_dir, "svm_bovw")
        )
        
    elif feature_type == 'spatial':
        classifier = evaluate_classifier(
            feature_extractor=compute_spatial_histogram,
            image_paths=image_paths,
            results_dir=os.path.join(results_dir, "svm_spatial")
        )
        
    elif feature_type == 'region':
        region_extractor = RegionFeatureExtractor(n_segments=100)
        classifier = evaluate_classifier(
            feature_extractor=region_extractor,
            image_paths=image_paths,
            results_dir=os.path.join(results_dir, "svm_region")
        )
    
    return classifier

def run_cnn_classification(image_paths, model_name, results_dir):
    """Run CNN classification with visualizations."""
    print(f"\nRunning CNN classification with {model_name}...")
    
    try:
        # Split data
        train_paths, val_paths = train_test_split(
            image_paths, test_size=0.2, random_state=42,
            stratify=[os.path.basename(p).split('_')[0] for p in image_paths]
        )
        
        # Initialize and train CNN
        cnn = CNNClassifier(
            model_name=model_name,
            results_dir=os.path.join(results_dir, f"cnn_{model_name}")
        )
        
        # Train model
        cnn.train(train_paths, val_paths)
        
        print("\nGenerating visualizations...")
        # Generate additional visualizations
        cnn.visualize_predictions(val_paths, num_samples=5, results_dir=cnn.results_dir)
        cnn.visualize_feature_space(image_paths, cnn.results_dir)
        
        return cnn
        
    except Exception as e:
        print(f"\nError in CNN processing: {str(e)}")
        print("Continuing with available results...")
        return None

def compare_results(results_dir, svm_results, cnn_results):
    """Compare and visualize results from different classifiers."""
    comparison_path = os.path.join(results_dir, 'comparison_results.txt')
    
    with open(comparison_path, 'w') as f:
        f.write("Classification Results Comparison\n")
        f.write("================================\n\n")
        
        # SVM Results
        if svm_results:
            f.write("SVM Classification Results:\n")
            f.write("--------------------------\n")
            f.write(f"Feature type: {args.feature_type}\n")
            f.write(f"Best accuracy: {svm_results.best_score_:.4f}\n")
            f.write(f"Best parameters: {svm_results.best_params_}\n\n")
        
        # CNN Results
        if cnn_results:
            f.write("CNN Classification Results:\n")
            f.write("--------------------------\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Final validation accuracy: {cnn_results.best_val_acc:.4f}\n\n")

def main():
    args = parse_args()
    
    print("Starting image classification system...")
    print(f"Using dataset: {IMAGE_FOLDER}\n")
    
    # Get all image paths
    image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.bmp')]
    image_paths = [os.path.join(IMAGE_PATH, f) for f in image_files]
    
    # Create timestamp for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base_dir = os.path.join('results', f'classification_{timestamp}')
    
    try:
        svm_results = None
        cnn_results = None
        
        # Run SVM classification
        if args.classifier in ['svm', 'both']:
            if args.feature_type == 'all':
                for feat_type in ['bovw', 'spatial', 'region']:
                    svm_results = run_svm_classification(
                        image_paths, feat_type, results_base_dir
                    )
            else:
                svm_results = run_svm_classification(
                    image_paths, args.feature_type, results_base_dir
                )
        
        # Run CNN classification
        if args.classifier in ['cnn', 'both']:
            cnn_results = run_cnn_classification(
                image_paths, args.model_name, results_base_dir
            )
        
        # Compare results if both classifiers were run
        if args.classifier == 'both':
            compare_results(results_base_dir, svm_results, cnn_results)
        
        print("\nClassification experiments completed!")
        print("Results saved in:", results_base_dir)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        return
    except Exception as e:
        print(f"\nError in classification processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()