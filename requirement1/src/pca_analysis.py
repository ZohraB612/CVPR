import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis

class PCAAnalyzer:
    def __init__(self, n_components=None):
        """
        Initialize PCA analyzer.
        
        Args:
            n_components: Number of components to keep (if None, keep all)
        """
        self.pca = PCA(n_components=n_components)
        self.covariance = None
        self.inv_covariance = None
        self.mean = None
    
    @property
    def components_(self):
        return self.pca.components_
        
    @components_.setter
    def components_(self, value):
        self.pca.components_ = value
    
    def fit_transform(self, features):
        """
        Fit PCA to features and transform them.
        
        Args:
            features: Array of shape (n_samples, n_features)
            
        Returns:
            transformed_features: Reduced dimension features
        """
        # Center the data
        self.mean = np.mean(features, axis=0)
        centered_features = features - self.mean
        
        # Fit PCA and transform
        transformed_features = self.pca.fit_transform(centered_features)
        
        # Compute covariance matrix for Mahalanobis distance
        self.covariance = np.cov(transformed_features.T)
        try:
            self.inv_covariance = np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError:
            # Handle singular matrix
            self.inv_covariance = np.linalg.pinv(self.covariance)
        
        return transformed_features
    
    def transform(self, features):
        """Transform new features using fitted PCA."""
        if self.mean is None:
            raise ValueError("PCA not fitted yet!")
        
        centered_features = features - self.mean
        return self.pca.transform(centered_features)
    
    def mahalanobis_distance(self, feature1, feature2):
        """
        Compute Mahalanobis distance between two feature vectors.
        
        Args:
            feature1, feature2: Feature vectors in PCA space
            
        Returns:
            float: Mahalanobis distance
        """
        if self.inv_covariance is None:
            raise ValueError("Covariance not computed yet!")
        
        return mahalanobis(feature1, feature2, self.inv_covariance)
    
    def compute_distances(self, query_feature, database_features):
        """
        Compute Mahalanobis distances between query and all database features.
        
        Args:
            query_feature: Query feature vector
            database_features: Array of database feature vectors
            
        Returns:
            np.array: Array of distances
        """
        distances = []
        for feat in database_features:
            dist = self.mahalanobis_distance(query_feature, feat)
            distances.append(dist)
        return np.array(distances)
    
    def get_explained_variance_ratio(self):
        """Get explained variance ratio for each component."""
        return self.pca.explained_variance_ratio_

def analyze_pca_performance(features, n_components_list=[8, 16, 32, 64]):
    """
    Analyze PCA performance with different numbers of components.
    
    Args:
        features: Original feature array
        n_components_list: List of component numbers to try
        
    Returns:
        dict: Results for each number of components
    """
    results = {}
    
    for n_components in n_components_list:
        # Initialize PCA
        pca_analyzer = PCAAnalyzer(n_components=n_components)
        
        # Transform features
        transformed_features = pca_analyzer.fit_transform(features)
        
        # Get explained variance
        explained_variance = np.sum(pca_analyzer.get_explained_variance_ratio())
        
        results[n_components] = {
            'transformed_features': transformed_features,
            'explained_variance': explained_variance,
            'analyzer': pca_analyzer
        }
    
    return results 