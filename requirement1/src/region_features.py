"""
Region-based feature extraction using superpixel segmentation.
Combines segmentation with spatial histogram features for improved object recognition.

Author: Zohra Bouchamaoui
Student ID: 6848526
"""

import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.color import label2rgb
import matplotlib.pyplot as plt

class RegionFeatureExtractor:
    def __init__(self, n_segments=100, compactness=10):
        """
        Initialize region-based feature extractor.
        
        Args:
            n_segments: Number of approximate segments to generate
            compactness: Controls compactness of segments
        """
        self.n_segments = n_segments
        self.compactness = compactness
        
    def extract_features(self, img):
        """
        Extract features using region-based approach.
        
        Args:
            img: Input image (BGR format)
        
        Returns:
            np.ndarray: Combined features from significant regions
        """
        # Convert to RGB for SLIC
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Generate superpixels
        segments = slic(img_rgb, n_segments=self.n_segments, 
                       compactness=self.compactness)
        
        # Extract region properties
        regions = regionprops(segments + 1)  # Add 1 to avoid background label 0
        
        # Store features for each region
        region_features = []
        region_weights = []
        
        for region in regions:
            # Create mask for this region
            mask = segments == (region.label - 1)
            
            # Extract region from original image
            region_img = img.copy()
            region_img[~mask] = 0
            
            # Compute region features
            features = self._compute_region_features(region_img, region)
            
            # Compute region importance weight
            weight = self._compute_region_importance(region, mask, img)
            
            region_features.append(features)
            region_weights.append(weight)
        
        # Weight and combine features
        region_features = np.array(region_features)
        region_weights = np.array(region_weights)
        region_weights = region_weights / np.sum(region_weights)  # Normalize weights
        
        # Weighted average of features
        final_features = np.average(region_features, 
                                  weights=region_weights,
                                  axis=0)
        
        return final_features
    
    def _compute_region_features(self, region_img, region):
        """Compute features for a single region."""
        # Get bounding box
        minr, minc, maxr, maxc = region.bbox
        roi = region_img[minr:maxr, minc:maxc]
        
        # Skip if region is too small
        if roi.shape[0] < 8 or roi.shape[1] < 8:
            return np.zeros(512)  # Return zero features
            
        # Compute spatial histogram for region
        from src.spatial_histogram import compute_spatial_histogram
        features = compute_spatial_histogram(roi)
        
        return features
    
    def _compute_region_importance(self, region, mask, img):
        """
        Compute importance weight for region based on:
        - Size
        - Location (central regions more important)
        - Color variance
        """
        # Size importance
        size_score = region.area / (img.shape[0] * img.shape[1])
        
        # Location importance (prefer central regions)
        cy, cx = region.centroid
        h, w = img.shape[:2]
        dist_to_center = np.sqrt(((cy/h - 0.5)**2 + (cx/w - 0.5)**2))
        location_score = 1 - dist_to_center
        
        # Color variance importance
        region_colors = img[mask]
        color_score = np.std(region_colors) / 255
        
        # Combine scores
        importance = (size_score + location_score + color_score) / 3
        return importance
    
    def visualize_segments(self, img, save_path=None):
        """Visualize segmentation results."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        segments = slic(img_rgb, n_segments=self.n_segments, 
                       compactness=self.compactness)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        ax1.imshow(img_rgb)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Segmentation
        img_segments = label2rgb(segments, img_rgb, kind='avg')
        ax2.imshow(img_segments)
        ax2.set_title(f'Superpixel Segmentation\n({self.n_segments} segments)')
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show() 