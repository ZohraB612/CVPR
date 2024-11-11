"""
Image classifier implementation using Support Vector Machines (SVM).
Can work with either BoVW features or spatial histogram features.

Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import cv2
from src.augmentation import augment_underrepresented_classes

class ImageClassifier:
    def __init__(self, feature_type='spatial_hist'):
        """
        Initialize the image classifier.
        
        Args:
            feature_type: 'spatial_hist' or 'bovw'
        """
        self.feature_type = feature_type
        self.scaler = StandardScaler()
        self.classifier = None
        self.classes = None
        
    def prepare_dataset(self, features, image_paths):
        """
        Prepare features and labels for classification.
        
        Args:
            features: numpy array of features (n_samples, n_features)
            image_paths: list of image paths to extract classes from
            
        Returns:
            X: scaled features
            y: labels
        """
        # Extract class labels from image paths
        labels = [os.path.basename(path).split('_')[0] for path in image_paths]
        self.classes = np.unique(labels)
        
        # Scale features
        X = self.scaler.fit_transform(features)
        y = np.array(labels)
        
        return X, y
    
    def train(self, features, image_paths, cv_folds=5):
        """
        Train classifier with augmented data for balanced classes.
        """
        # First, augment the dataset
        augmented_paths = augment_underrepresented_classes(image_paths)
        
        # Extract features for augmented images
        print("Extracting features for augmented images...")
        all_features = []
        all_paths = []
        
        # Original features
        all_features.extend(features)
        all_paths.extend(image_paths)
        
        # Augmented features
        augmented_only = [p for p in augmented_paths if p not in image_paths]
        for path in tqdm(augmented_only):
            img = cv2.imread(path)
            if img is not None:
                if self.feature_type == 'spatial':
                    feat = compute_spatial_histogram(img)
                else:  # bovw
                    feat = self.feature_extractor.extract_features(img)
                all_features.append(feat)
                all_paths.append(path)
        
        X, y = self.prepare_dataset(np.array(all_features), all_paths)
        
        # Split dataset with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Enhanced parameter grid
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear'],
            'class_weight': ['balanced', None]
        }
        
        # Initialize SVM with probability estimation
        base_svm = SVC(probability=True, random_state=42)
        
        # Use StratifiedKFold for cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Initialize GridSearchCV with f1_macro scoring
        self.classifier = GridSearchCV(
            base_svm, 
            param_grid, 
            cv=cv,
            n_jobs=-1,
            verbose=1,
            scoring='f1_macro'
        )
        
        print("Training SVM classifier...")
        self.classifier.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test)
        
        # Print detailed results
        print("\nBest parameters:", self.classifier.best_params_)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix with improved visualization
        self._plot_confusion_matrix(y_test, y_pred)
        
        return self.classifier.best_score_
    
    def predict(self, features):
        """
        Predict class probabilities for new features.
        
        Args:
            features: numpy array of features (n_samples, n_features)
            
        Returns:
            predictions: class predictions
            probabilities: class probabilities
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained yet!")
            
        # Scale features
        X = self.scaler.transform(features)
        
        # Get predictions and probabilities
        predictions = self.classifier.predict(X)
        probabilities = self.classifier.predict_proba(X)
        
        return predictions, probabilities
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Enhanced confusion matrix plotting."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=self.classes,
                    yticklabels=self.classes)
        
        plt.title('Confusion Matrix\n(numbers show raw counts, colors show percentages)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

def evaluate_classifier(feature_extractor, image_paths, results_dir):
    """
    Evaluate the classifier and save results.
    
    Args:
        feature_extractor: BoVWRetrieval or spatial histogram extractor
        image_paths: list of image paths
        results_dir: directory to save results
    """
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract features if using BoVW
    if hasattr(feature_extractor, 'compute_bovw_features'):
        feature_extractor.compute_bovw_features(image_paths)
        features = feature_extractor.features
        classifier = ImageClassifier(feature_type='bovw')
    else:
        # For spatial histogram
        features = []
        for path in tqdm(image_paths, desc="Extracting features"):
            img = cv2.imread(path)
            if img is not None:
                feat = feature_extractor(img)
                features.append(feat)
        features = np.array(features)
        classifier = ImageClassifier(feature_type='spatial_hist')
    
    # Train and evaluate
    best_score = classifier.train(features, image_paths)
    
    # Save results
    with open(os.path.join(results_dir, 'classification_results.txt'), 'w') as f:
        f.write(f"Classification Results\n")
        f.write(f"Feature type: {classifier.feature_type}\n")
        f.write(f"Best cross-validation score: {best_score:.4f}\n")
        f.write(f"Best parameters: {classifier.classifier.best_params_}\n")
        
    return classifier 