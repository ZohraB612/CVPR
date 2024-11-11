"""
Bag of Visual Words (BoVW) retrieval implementation.
Uses SIFT features and k-means clustering for visual codebook generation.

Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from src.utils import get_image_class
from src.evaluation import compute_pr_curve

class BoVWRetrieval:
    def __init__(self, config):
        """
        Initialize BoVW retrieval system.
        
        Args:
            config: Dictionary containing configuration parameters
                   - codebook_size: Number of visual words
                   - detector: 'sift' or 'harris'
        """
        self.config = config
        self.sift = cv2.SIFT_create()
        self.codebook = None
        self.features = []
        self.files = []
        
    def extract_features(self, image):
        """
        Extract SIFT features from image.
        
        Args:
            image: Input image (BGR format)
        Returns:
            numpy array: SIFT descriptors
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return descriptors
    
    def build_codebook(self, image_paths, codebook_size=1000):
        """
        Build visual codebook using k-means clustering.
        
        Args:
            image_paths: List of image paths
            codebook_size: Number of visual words
        """
        # Check for cached codebook
        codebook_cache = f'cache_bovw_codebook_{codebook_size}.npy'
        if os.path.exists(codebook_cache):
            print("Loading cached codebook...")
            self.codebook = np.load(codebook_cache)
            return

        print("Building codebook...")
        all_descriptors = []
        
        # Collect SIFT descriptors from all images
        for path in tqdm(image_paths, desc="Extracting SIFT features"):
            img = cv2.imread(path)
            if img is None:
                continue
            
            descriptors = self.extract_features(img)
            if descriptors is not None:
                all_descriptors.append(descriptors)
        
        # Stack all descriptors
        all_descriptors = np.vstack(all_descriptors)
        
        # Perform k-means clustering
        print(f"Running k-means with {codebook_size} clusters...")
        kmeans = KMeans(n_clusters=codebook_size, random_state=42, n_init=10)
        kmeans.fit(all_descriptors)
        
        self.codebook = kmeans.cluster_centers_
        print("Codebook built successfully!")
        
        # Cache the codebook
        np.save(codebook_cache, self.codebook)
        
    def compute_bovw_features(self, image_paths):
        """
        Compute BoVW features for all images.
        
        Args:
            image_paths: List of image paths
        """
        # Check for cached features
        features_cache = f'cache_bovw_features_{self.config["codebook_size"]}.npz'
        if os.path.exists(features_cache):
            print("Loading cached BoVW features...")
            cache = np.load(features_cache)
            self.features = cache['features']
            self.files = cache['files']
            return

        print("Computing BoVW features...")
        self.features = []
        self.files = []
        
        for path in tqdm(image_paths, desc="Computing BoVW histograms"):
            img = cv2.imread(path)
            if img is None:
                continue
                
            # Extract SIFT features
            descriptors = self.extract_features(img)
            if descriptors is None:
                continue
                
            # Assign features to nearest visual words
            assignments = []
            for desc in descriptors:
                distances = np.linalg.norm(self.codebook - desc.reshape(1, -1), axis=1)
                nearest_word = np.argmin(distances)
                assignments.append(nearest_word)
                
            # Create histogram of visual words
            hist, _ = np.histogram(assignments, 
                                 bins=range(len(self.codebook) + 1), 
                                 density=True)
            
            # Normalize histogram
            hist = normalize(hist.reshape(1, -1))[0]
            
            self.features.append(hist)
            self.files.append(path)
        
        # Convert to numpy arrays
        self.features = np.array(self.features)
        self.files = np.array(self.files)
        
        # Cache the features
        np.savez(features_cache, features=self.features, files=self.files)
        
    def process_query(self, query_path, results_dir):
        """
        Process a query image and save results.
        
        Args:
            query_path: Path to query image
            results_dir: Directory to save results
        """
        try:
            # Extract query features
            query_img = cv2.imread(query_path)
            query_descriptors = self.extract_features(query_img)
            
            if query_descriptors is None:
                raise ValueError("No SIFT features found in query image")
            
            # Create query histogram
            assignments = []
            for desc in query_descriptors:
                distances = np.linalg.norm(self.codebook - desc.reshape(1, -1), axis=1)
                nearest_word = np.argmin(distances)
                assignments.append(nearest_word)
                
            query_hist, _ = np.histogram(assignments, 
                                       bins=range(len(self.codebook) + 1), 
                                       density=True)
            query_hist = normalize(query_hist.reshape(1, -1))[0]
            
            # Compute distances
            distances = []
            for i, feat in enumerate(self.features):
                dist = np.linalg.norm(query_hist - feat)
                distances.append((dist, self.files[i]))
                
            # Sort distances
            distances.sort(key=lambda x: x[0])
            
            # Create results directory
            query_name = os.path.splitext(os.path.basename(query_path))[0]
            query_dir = os.path.join(results_dir, f"query_{query_name}")
            os.makedirs(query_dir, exist_ok=True)
            
            # Get query class and compute PR curve
            query_class = get_image_class(query_path)
            pr_data = compute_pr_curve(query_class, distances, len(self.files))
            
            # Plot PR curve
            plt.figure(figsize=(10, 6))
            plt.plot(pr_data['recall'], pr_data['precision'], 'b-')
            plt.title(f'Precision-Recall Curve for {query_name}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.grid(True)
            plt.savefig(os.path.join(query_dir, 'pr_curve.png'))
            plt.close()
            
            # Plot retrieval results
            plt.figure(figsize=(15, 8))
            query_img = cv2.cvtColor(cv2.imread(query_path), cv2.COLOR_BGR2RGB)
            plt.subplot(3, 7, 1)
            plt.imshow(query_img)
            plt.title('Query Image')
            plt.axis('off')
            
            # Plot top 20 retrieved images
            for i, (dist, img_path) in enumerate(distances[:20], 2):
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                plt.subplot(3, 7, i)
                plt.imshow(img)
                retrieved_class = get_image_class(img_path)
                is_correct = '✓' if retrieved_class == query_class else '✗'
                plt.title(f'{is_correct}\nDist: {dist:.2f}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(query_dir, 'search_results.png'))
            plt.close()
            
            # Save analysis
            with open(os.path.join(query_dir, 'analysis.txt'), 'w') as f:
                f.write(f"Analysis Report for Query Image: {os.path.basename(query_path)}\n")
                f.write(f"Query Class: {query_class}\n\n")
                
                f.write("BoVW Configuration:\n")
                f.write(f"Codebook size: {self.config['codebook_size']}\n")
                f.write(f"Feature detector: {self.config['detector']}\n\n")
                
                f.write("Retrieval Performance:\n")
                f.write(f"Average Precision: {np.mean(pr_data['precision']):.4f}\n")
                f.write(f"Precision@10: {pr_data['precision'][9]:.3f}\n")
                f.write(f"Precision@20: {pr_data['precision'][19]:.3f}\n")
                
                # Count correct retrievals
                correct = sum(1 for _, path in distances[:20] 
                            if get_image_class(path) == query_class)
                f.write(f"\nCorrect retrievals in top 20: {correct}\n")
                
                # SIFT feature statistics
                f.write("\nSIFT Feature Statistics:\n")
                f.write(f"Number of SIFT features in query: {len(query_descriptors)}\n")
            
            print(f"Saved results for {query_name}")
            
        except Exception as e:
            print(f"Error processing BoVW query {query_path}: {str(e)}")
            raise 