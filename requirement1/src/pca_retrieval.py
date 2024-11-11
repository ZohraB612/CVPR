"""
PCA-based image retrieval implementation.
Extends the base system with PCA and Mahalanobis distance.

Author: Zohra Bouchamaoui
Student ID: 6848526
"""

import os
import cv2
import numpy as np
from datetime import datetime
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from src.histogram import compute_global_histogram, euclidean_distance
from src.utils import get_image_class
from src.evaluation import compute_pr_curve
from src.visualisation import save_experiment_results
import matplotlib.pyplot as plt

class PCARetrieval:
    def __init__(self, n_components=32):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.features = None
        self.files = None
        self.mean = None
        self.covariance = None
        self.inv_covariance = None
        self.config = {
            'name': f'PCA_{n_components}',
            'r': 8,  # Using standard RGB quantization
            'g': 8,
            'b': 4,
            'distance': 'mahalanobis',
            'components': n_components,
            'use_pca': True
        }
        
    def compute_features(self, image_path):
        """Compute color histograms and apply PCA."""
        print(f"\nComputing features with PCA ({self.n_components} components)...")
        
        # Get list of images
        image_files = [f for f in os.listdir(image_path) if f.endswith('.bmp')]
        self.files = [os.path.join(image_path, f) for f in image_files]
        
        # Compute histograms
        features = []
        for filename in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(image_path, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Warning: Could not load image: {img_path}")
                continue
                
            hist = compute_global_histogram(
                img, 
                r_bins=self.config['r'], 
                g_bins=self.config['g'], 
                b_bins=self.config['b']
            ).flatten()
            features.append(hist)
            
        # Convert to numpy array
        features = np.array(features)
        
        # Apply PCA
        self.mean = np.mean(features, axis=0)
        features_centered = features - self.mean
        
        # Fit PCA and transform features
        self.features = self.pca.fit_transform(features_centered)
        
        # Compute covariance for Mahalanobis distance
        self.covariance = np.cov(self.features.T)
        try:
            self.inv_covariance = np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError:
            print("Warning: Singular covariance matrix, using pseudo-inverse")
            self.inv_covariance = np.linalg.pinv(self.covariance)
            
        # Print explained variance
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"Total explained variance: {explained_var:.3f}")
        
    def process_query(self, query_path, results_dir):
        """Process a single query image."""
        try:
            # Load and compute query features
            query_img = cv2.imread(query_path)
            query_hist = compute_global_histogram(
                query_img, 
                r_bins=self.config['r'], 
                g_bins=self.config['g'], 
                b_bins=self.config['b']
            ).flatten()
            
            # Transform query to PCA space
            query_centered = query_hist - self.mean
            query_pca = self.pca.transform(query_centered.reshape(1, -1))[0]
            
            # Compute Mahalanobis distances
            distances = []
            for i, feat in enumerate(self.features):
                dist = mahalanobis(query_pca, feat, self.inv_covariance)
                distances.append((dist, self.files[i]))
            
            # Sort distances
            distances.sort()
            
            # Get query class and compute PR curve
            query_class = get_image_class(query_path)
            pr_data = compute_pr_curve(query_class, distances, len(self.files))
            
            # Save results using existing visualization function
            save_experiment_results(query_path, distances, pr_data, self.config, results_dir)
            
        except Exception as e:
            print(f"Error processing PCA query {query_path}: {str(e)}")
            
    def run_experiment(self, image_path, results_base_dir):
        """Run complete PCA experiment."""
        # Compute features
        self.compute_features(image_path)
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(
            results_base_dir, 
            f'pca_{self.n_components}_{timestamp}'
        )
        os.makedirs(results_dir, exist_ok=True)
        
        # Process test queries
        test_queries = {
            'building': '3_1_s.bmp',
            'face': '17_1_s.bmp',
            'sheep': '6_1_s.bmp',
            'street': '9_1_s.bmp'
        }
        
        print("\nProcessing test queries...")
        for query_name, query_file in test_queries.items():
            print(f"Processing query: {query_name}")
            query_path = os.path.join(image_path, query_file)
            self.process_query(query_path, results_dir)
            
        print(f"\nResults saved to: {results_dir}")
    
    def analyze_components(self, image_path, max_components=256):
        """
        Analyze PCA components and their explained variance.
        
        Args:
            image_path: Path to image directory
            max_components: Maximum number of components to analyze
        """
        # Compute histograms first
        print("\nComputing histograms for PCA analysis...")
        image_files = [f for f in os.listdir(image_path) if f.endswith('.bmp')]
        features = []
        
        for filename in tqdm(image_files, desc="Processing images"):
            img_path = os.path.join(image_path, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            hist = compute_global_histogram(
                img, 
                r_bins=self.config['r'], 
                g_bins=self.config['g'], 
                b_bins=self.config['b']
            ).flatten()
            features.append(hist)
        
        features = np.array(features)
        
        # Center the data
        mean = np.mean(features, axis=0)
        features_centered = features - mean
        
        # Compute full PCA
        pca_full = PCA(n_components=min(max_components, len(features_centered)))
        pca_full.fit(features_centered)
        
        # Compute cumulative explained variance
        cumulative_variance_ratio = np.cumsum(pca_full.explained_variance_ratio_)
        
        # Create plots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Explained variance ratio for each component
        plt.subplot(2, 1, 1)
        plt.plot(pca_full.explained_variance_ratio_, 'b-', label='Individual')
        plt.title('Explained Variance Ratio by Component')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)
        plt.legend()
        
        # Plot 2: Cumulative explained variance
        plt.subplot(2, 1, 2)
        plt.plot(cumulative_variance_ratio, 'r-', label='Cumulative')
        plt.title('Cumulative Explained Variance Ratio')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.grid(True)
        
        # Add horizontal lines at 90%, 95%, and 99%
        for threshold in [0.9, 0.95, 0.99]:
            n_components = np.where(cumulative_variance_ratio >= threshold)[0][0] + 1
            plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
            plt.text(max_components/2, threshold, f'{threshold*100}% - {n_components} components', 
                    verticalalignment='bottom')
        
        plt.legend()
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('pca_analysis.png')
        plt.close()
        
        # Print analysis
        print("\nPCA Component Analysis:")
        for threshold in [0.9, 0.95, 0.99]:
            n_components = np.where(cumulative_variance_ratio >= threshold)[0][0] + 1
            print(f"Components needed for {threshold*100}% variance: {n_components}")
        
        # Find elbow point using kneedle algorithm
        try:
            from kneed import KneeLocator
            x = np.arange(len(cumulative_variance_ratio))
            kneedle = KneeLocator(x, cumulative_variance_ratio, 
                                 S=1.0, curve='concave', direction='increasing')
            if kneedle.knee is not None:
                print(f"\nOptimal number of components (elbow method): {kneedle.knee}")
        except ImportError:
            print("\nInstall kneed package for automatic elbow point detection: pip install kneed")