"""
Quantization analysis script.
Evaluates different RGB quantization configurations.

Author: Zohra Bouchamaoui
Student ID: 6848526
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.histogram import compute_global_histogram, euclidean_distance
from src.evaluation import compute_pr_curve
from config.settings import TEST_QUERIES, CONFIGS, IMAGE_PATH
from src.utils import get_image_class

def evaluate_configuration(config, query_path):
    """Evaluate a single configuration on one query"""
    query_img = cv2.imread(query_path)
    query_class = get_image_class(query_path)
    query_hist = compute_global_histogram(query_img, config['r'], config['g'], config['b'])
    
    # Process all images
    distances = []
    for img_file in os.listdir(IMAGE_PATH):
        if img_file.endswith('.bmp'):
            img_path = os.path.join(IMAGE_PATH, img_file)
            img = cv2.imread(img_path)
            hist = compute_global_histogram(img, config['r'], config['g'], config['b'])
            dist = euclidean_distance(query_hist, hist)
            distances.append((dist, img_path))
    
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    
    # Compute metrics
    pr_data = compute_pr_curve(query_class, distances, len(distances))
    p_at_10 = pr_data['precision'][9] if len(pr_data['precision']) > 9 else 0
    
    return p_at_10

def main():
    # Results dictionary
    results = {config['name']: {} for config in CONFIGS}
    
    # Evaluate each configuration for each query
    for config in CONFIGS:
        print(f"\nEvaluating configuration: {config['name']}")
        for category, query_path in TEST_QUERIES.items():
            p_at_10 = evaluate_configuration(config, query_path)
            results[config['name']][category] = p_at_10
            print(f"{category}: P@10 = {p_at_10:.3f}")
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    data = np.array([[results[config['name']][cat] for cat in TEST_QUERIES.keys()] 
                     for config in CONFIGS])
    
    sns.heatmap(data, 
                annot=True, 
                fmt='.3f',
                xticklabels=TEST_QUERIES.keys(),
                yticklabels=[config['name'] for config in CONFIGS],
                cmap='YlOrRd')
    
    plt.title('Precision@10 for Different Quantization Configurations')
    plt.xlabel('Image Category')
    plt.ylabel('Configuration')
    plt.tight_layout()
    plt.savefig('quantization_analysis.png')
    plt.close()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for config in CONFIGS:
        avg_p10 = np.mean(list(results[config['name']].values()))
        std_p10 = np.std(list(results[config['name']].values()))
        print(f"\n{config['name']}:")
        print(f"Average P@10: {avg_p10:.3f} ± {std_p10:.3f}")
        print(f"Dimensions: {config['r'] * config['g'] * config['b']}")
        print(f"Memory per image: {config['r'] * config['g'] * config['b'] * 4 / 1024:.2f}KB")

if __name__ == "__main__":
    main() 