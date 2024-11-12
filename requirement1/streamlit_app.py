"""
Streamlit interface for image retrieval system using global color histograms.
"""

import streamlit as st
import cv2
import os
from datetime import datetime
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import traceback
from scipy.spatial.distance import mahalanobis
from sklearn.preprocessing import normalize
import torch
import matplotlib.pyplot as plt
from src.visualisation import plot_pr_curve, plot_confusion_matrix, plot_average_pr_curve, plot_class_wise_pr_curves
import time
from sklearn.cluster import MiniBatchKMeans

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Local imports
from config.settings import (
    IMAGE_FOLDER, 
    IMAGE_PATH, 
    TEST_QUERIES, 
    CONFIGS,
    SPATIAL_CONFIGS,
    BASE_PATH
)
from src.histogram import (
    compute_global_histogram, 
    euclidean_distance, 
    manhattan_distance, 
    chi_square_distance, 
    intersection_distance,
    mahalanobis_distance,
    cosine_distance
)
from src.spatial_histogram import compute_spatial_histogram
from src.pca_retrieval import PCARetrieval
from src.bovw_retrieval import BoVWRetrieval
from src.utils import get_image_class
from src.cnn_classifier import CNNClassifier
from src.image_classifier import ImageClassifier
from src.visualisation import plot_histogram_comparison, save_match_visualization

def get_class_from_filename(filename):
    """Extract class number from filename and return corresponding class name"""
    class_num = filename.split('_')[0]
    class_mapping = {
        '1': 'farm animals',
        '2': 'trees',
        '3': 'buildings',
        '4': 'airplanes',
        '5': 'cows',
        '6': 'faces',
        '7': 'cars',
        '8': 'bicycles',
        '9': 'sheep',
        '10': 'flowers',
        '11': 'signs',
        '12': 'birds',
        '13': 'books',
        '14': 'chairs',
        '15': 'cats',
        '16': 'dogs',
        '17': 'streets',
        '18': 'nature',
        '19': 'people',
        '20': 'boats',
    }
    return class_mapping.get(class_num, 'unknown')

def normalize_class_name(class_name):
    """Normalize class names to match the mapping"""
    class_mapping = {
        'buildings': 'building',
        'building': 'building',
        'airplane': 'airplane',
        'animal': 'farm animal',
        'streets': 'streets',
        'street': 'streets',
        'faces': 'face',
        'face': 'face',
        'sheep': 'sheep',
    }
    
    # Convert to lowercase and normalize
    normalized = class_name.lower()
    if normalized not in class_mapping:
        # Remove trailing 's' if present
        if normalized.endswith('s'):
            normalized = normalized[:-1]
    else:
        normalized = class_mapping[normalized]
        
    print(f"Normalizing '{class_name}' to '{normalized}'")
    return normalized

def main():
    # Define class mapping and class_to_idx at the start
    class_mapping = {
        '1': 'building',
        '2': 'tree',
        '3': 'building',
        '4': 'airplane',
        '5': 'cow',
        '6': 'face',
        '7': 'car',
        '8': 'bicycle',
        '9': 'sheep',
        '10': 'flower',
        '11': 'sign',
        '12': 'bird',
        '13': 'book',
        '14': 'chair',
        '15': 'cat',
        '16': 'dog',
        '17': 'streets',
        '18': 'nature',
        '19': 'people',
        '20': 'boat',
    }
    
    # Create class_to_idx dictionary with both singular and plural forms
    unique_classes = sorted(set(class_mapping.values()))
    class_to_idx = {}
    for idx, class_name in enumerate(unique_classes):
        class_to_idx[class_name] = idx

    # Always show debug info temporarily
    st.write("Available classes:", list(class_to_idx.keys()))
    st.write("Class mapping:", class_mapping)

    # Add task selection at the top
    task = st.sidebar.radio(
        "Select Task",
        ["Image Retrieval"]
    )

    if task == "Image Classification":
        st.title("Image Classification")
        
        # Model selection
        model_name = st.sidebar.selectbox(
            "Select Model",
            ["resnet18", "resnet50"],
            index=0,
            help="ResNet18 is faster, ResNet50 might be more accurate"
        )
        
        # Upload image for classification
        uploaded_file = st.file_uploader("Choose an image...", type=['bmp', 'jpg', 'png'])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                st.image(image, caption='Uploaded Image', width=250)
            
            # Create classifier
            classifier = CNNClassifier(model_name=model_name)
            
            # Load pre-trained weights if available
            model_path = os.path.join('results', 'best_model.pth')
            if os.path.exists(model_path):
                classifier.model.load_state_dict(torch.load(model_path, map_location=classifier.device))
                classifier.model.eval()
                
                if st.button("Classify Image"):
                    with st.spinner('Classifying...'):
                        # Process image
                        img_array = np.array(image)
                        if len(img_array.shape) == 2:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
                        elif img_array.shape[2] == 4:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                        
                        # Predict
                        with torch.no_grad():
                            x = classifier.transform(img_array).unsqueeze(0).to(classifier.device)
                            outputs = classifier.model(x)
                            probs = torch.nn.functional.softmax(outputs, dim=1)
                            conf, pred = torch.max(probs, 1)
                        
                        # Display results
                        st.subheader("Classification Results")
                        predicted_class = pred.item() + 1
                        class_name = get_class_from_filename(f"{predicted_class}_")
                        
                        st.write(f"**Predicted Class:** {class_name}")
                        st.write(f"**Confidence:** {conf.item():.2%}")
                        
                        # Display top-3 predictions
                        st.subheader("Top 3 Predictions")
                        probs = probs.cpu().numpy()[0]
                        top3_idx = np.argsort(probs)[-3:][::-1]
                        
                        for idx in top3_idx:
                            class_id = idx + 1
                            class_name = get_class_from_filename(f"{class_id}_")
                            prob = probs[idx]
                            st.write(f"{class_name}: {prob:.2%}")
            else:
                st.warning("No trained model found. Please train the model first.")
                
    else:  # Image Retrieval
        st.title("Image Retrieval System")
        st.sidebar.header("Configuration")

        # Method selection
        retrieval_method = st.sidebar.selectbox(
            "Select Retrieval Method",
            ["Color Histogram", "Spatial Histogram", "PCA", "Bag of Visual Words", "CNN Classification", "SVM"]
        )

        # Query image selection
        query_source = st.radio(
            "Select Query Source",
            ["Test Queries"]
        )
        
        query_img = None
        selected_query = None
        query_path = None
        
        if query_source == "Test Queries":
            display_names = {
                'building': 'Building (3_1_s.bmp)',
                'street': 'Street (17_1_s.bmp)',
                'face': 'Face (6_1_s.bmp)',
                'sheep': 'Sheep (9_1_s.bmp)'
            }
            
            selected_query = st.selectbox(
                "Select Test Query", 
                list(TEST_QUERIES.keys()),
                format_func=lambda x: display_names[x]
            )
            
            if selected_query:
                query_path = TEST_QUERIES[selected_query]
                if os.path.exists(query_path):
                    query_img = cv2.imread(query_path)
                    if query_img is not None:
                        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
                        col1, col2, col3 = st.columns([2,1,2])
                        with col2:
                            st.image(query_img, caption=f"Test Query: {display_names[selected_query]}", width=250)

        if query_img is not None:
            # Initialize config variable
            config = None
            
            if retrieval_method == "Color Histogram":
                r_bins = st.sidebar.selectbox("R Bins", options=[2, 4, 8, 16, 32], index=2)
                g_bins = st.sidebar.selectbox("G Bins", options=[2, 4, 8, 16, 32], index=2)
                b_bins = st.sidebar.selectbox("B Bins", options=[2, 4, 8, 16, 32], index=2)
                
                # Add distance metric selection
                distance_metric = st.sidebar.selectbox(
                    "Distance Metric",
                    ["Euclidean", "Manhattan", "Chi-Square", "Intersection", "Mahalanobis", "Cosine"],
                    help="""
                    Euclidean: Standard L2 distance
                    Manhattan: L1 distance, less sensitive to outliers
                    Chi-Square: Normalized bin differences
                    Intersection: Overlap between histograms
                    Mahalanobis: Statistical distance accounting for correlations
                    Cosine: Angle-based similarity measure
                    """
                )
                
                config = {
                    'name': 'custom',
                    'r': r_bins,
                    'g': g_bins,
                    'b': b_bins,
                    'use_spatial': False,
                    'distance_metric': distance_metric
                }
            elif retrieval_method == "PCA":
                n_components = st.sidebar.selectbox(
                    "Number of Components", 
                    options=[8, 16, 32, 64, 128, 256],
                    index=2  # Default to 32
                )
                
                # Add distance metric selection for PCA
                distance_metric = st.sidebar.selectbox(
                    "Distance Metric",
                    ["Euclidean", "Manhattan", "Chi-Square", "Intersection", "Mahalanobis", "Cosine"],
                    help="""
                    Euclidean: Standard L2 distance
                    Manhattan: L1 distance, less sensitive to outliers
                    Chi-Square: Normalized differences
                    Intersection: Overlap between histograms
                    Mahalanobis: Statistical distance accounting for correlations
                    Cosine: Angle-based similarity measure
                    """
                )
                
                config = {
                    'n_components': n_components,
                    'distance_metric': distance_metric
                }
            elif retrieval_method == "Spatial Histogram":
                r_bins = st.sidebar.selectbox("R Bins", options=[2, 4, 8, 16], index=2)
                g_bins = st.sidebar.selectbox("G Bins", options=[2, 4, 8, 16], index=2)
                b_bins = st.sidebar.selectbox("B Bins", options=[2, 4, 8, 16], index=2)
                grid_size = st.sidebar.selectbox("Grid Size", options=[2, 4, 6, 8], index=1)
                
                config = {
                    'name': 'spatial',
                    'r': r_bins,
                    'g': g_bins,
                    'b': b_bins,
                    'grid_size': grid_size,
                    'use_spatial': True
                }
            elif retrieval_method == "Bag of Visual Words":
                codebook_size = st.sidebar.selectbox(
                    "Codebook Size",
                    options=[500, 1000, 2000],
                    index=1
                )
                detector = st.sidebar.selectbox(
                    "Feature Detector",
                    options=["sift", "orb", "akaze", "brisk"],
                    index=0
                )
                
                # Add distance metric selector
                distance_metric = st.sidebar.selectbox(
                    "Distance Metric",
                    options=[
                        "euclidean",
                        "manhattan",
                        "chi_square",
                        "hellinger",
                        "bhattacharyya",
                        "kl_divergence",
                        "cosine"
                    ],
                    index=0,
                    help="""
                    Euclidean: Standard L2 distance
                    Manhattan: L1 distance
                    Chi-Square: Histogram comparison
                    Hellinger: Statistical distance
                    Bhattacharyya: Similarity between distributions
                    KL Divergence: Information theory based
                    Cosine: Angle-based similarity
                    """
                )
                
                config = {
                    'codebook_size': codebook_size,
                    'detector': detector,
                    'distance_metric': distance_metric
                }
            elif retrieval_method == "CNN Classification":
                model_name = st.sidebar.selectbox(
                    "Model Architecture",
                    ["resnet18", "resnet50"],
                    index=0,
                    help="ResNet18 is faster, ResNet50 might be more accurate"
                )
                
                config = {
                    'model_name': model_name
                }
            elif retrieval_method == "SVM":
                config = {}  # SVM doesn't need additional config

            # Get query class
            query_class = None
            if selected_query:
                query_class = get_class_from_filename(os.path.basename(query_path))
            elif query_path:  # For uploaded images
                query_class = get_class_from_filename(os.path.basename(query_path))

            # Add debug prints before class lookup
            if query_class:
                st.write(f"Original query class: {query_class}")
                query_class = normalize_class_name(query_class)
                st.write(f"Normalized query class: {query_class}")
                st.write("Available classes:", list(class_to_idx.keys()))
                
                try:
                    query_idx = class_to_idx[query_class]
                    st.write(f"Found index: {query_idx}")
                except KeyError:
                    st.error(f"Class '{query_class}' not found in available classes: {list(class_to_idx.keys())}")
                    return

            if st.button("Find Similar Images"):
                with st.spinner("Processing..."):
                    results = []
                    
                    if retrieval_method == "Color Histogram":
                        results = process_histogram_query(query_img, config, query_path)
                    elif retrieval_method == "PCA":
                        results = process_pca_query(query_img, config, query_path)
                    elif retrieval_method == "Spatial Histogram":
                        results = process_spatial_histogram_query(query_img, config, query_path)
                    elif retrieval_method == "Bag of Visual Words":
                        results = process_bovw_query(query_img, config, query_path)
                    elif retrieval_method == "CNN Classification":
                        results = process_cnn_query(query_img, config, query_path)
                    elif retrieval_method == "SVM":
                        results = process_svm_query(query_img, config, query_path)

                if results:
                    # Debug information
                    # st.write(f"Total results found: {len(results)}")
                    # st.write(f"First few distances: {[x[0] for x in results[:3]]}")
                    
                    # Display results
                    st.subheader("Retrieved Images")
                    
                    # First row - top 5
                    st.write("Top 5 Similar Images:")
                    cols1 = st.columns(5)
                    for i, col in enumerate(cols1):
                        if i < min(5, len(results)):
                            dist, img_path, _ = results[i]
                            # st.write(f"Processing result {i+1}: {img_path}")  # Debug info
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                col.image(img, use_container_width=True)
                                img_class = get_class_from_filename(os.path.basename(img_path))
                                col.write(f"Class: {img_class}")
                                col.write(f"Distance: {dist:.3f}")
                                
                                # Add histogram comparison only for Color Histogram method
                                if retrieval_method == "Color Histogram":
                                    query_img_bgr = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
                                    match_img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                    
                                    query_hist = compute_global_histogram(
                                        query_img_bgr,
                                        r_bins=config['r'],
                                        g_bins=config['g'],
                                        b_bins=config['b']
                                    )
                                    match_hist = compute_global_histogram(
                                        match_img_bgr,
                                        r_bins=config['r'],
                                        g_bins=config['g'],
                                        b_bins=config['b']
                                    )
                                    
                                    fig, ax = plt.subplots(figsize=(4, 3))
                                    bins = range(len(query_hist))
                                    ax.bar(bins, query_hist, alpha=0.5, label='Query', color='blue')
                                    ax.bar(bins, match_hist, alpha=0.5, label='Match', color='red')
                                    ax.set_title(f'Distance: {dist:.3f}')
                                    ax.legend(fontsize='x-small')
                                    ax.tick_params(axis='both', which='major', labelsize='x-small')
                                    plt.tight_layout()
                                    col.pyplot(fig)
                                    plt.close()
                                
                                # Add spatial histogram comparison only for Spatial Histogram method
                                if retrieval_method == "Spatial Histogram":
                                    query_img_bgr = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
                                    match_img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                    
                                    query_hist = compute_spatial_histogram(
                                        query_img_bgr,
                                        r_bins=config['r'],
                                        g_bins=config['g'],
                                        b_bins=config['b'],
                                        grid_size=(config['grid_size'], config['grid_size'])
                                    )
                                    match_hist = compute_spatial_histogram(
                                        match_img_bgr,
                                        r_bins=config['r'],
                                        g_bins=config['g'],
                                        b_bins=config['b'],
                                        grid_size=(config['grid_size'], config['grid_size'])
                                    )
                                    
                                    fig, ax = plt.subplots(figsize=(4, 3))
                                    bins = range(len(query_hist))
                                    ax.bar(bins, query_hist, alpha=0.5, label='Query', color='blue')
                                    ax.bar(bins, match_hist, alpha=0.5, label='Match', color='red')
                                    ax.set_title(f'Distance: {dist:.3f}')
                                    ax.legend(fontsize='x-small')
                                    ax.tick_params(axis='both', which='major', labelsize='x-small')
                                    plt.tight_layout()
                                    col.pyplot(fig)
                                    plt.close()
                    
                    # Second row - next 5
                    st.write("More Similar Images:")
                    cols2 = st.columns(5)
                    for i, col in enumerate(cols2):
                        idx = i + 5
                        if idx < min(10, len(results)):
                            dist, img_path, _ = results[idx]
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                col.image(img, use_container_width=True)
                                img_class = get_class_from_filename(os.path.basename(img_path))
                                col.write(f"Class: {img_class}")
                                col.write(f"Distance: {dist:.3f}")
                                
                                # Add histogram comparison only for Color Histogram method
                                if retrieval_method == "Color Histogram":
                                    query_img_bgr = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
                                    match_img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                    
                                    query_hist = compute_global_histogram(
                                        query_img_bgr,
                                        r_bins=config['r'],
                                        g_bins=config['g'],
                                        b_bins=config['b']
                                    )
                                    match_hist = compute_global_histogram(
                                        match_img_bgr,
                                        r_bins=config['r'],
                                        g_bins=config['g'],
                                        b_bins=config['b']
                                    )
                                    
                                    fig, ax = plt.subplots(figsize=(4, 3))
                                    bins = range(len(query_hist))
                                    ax.bar(bins, query_hist, alpha=0.5, label='Query', color='blue')
                                    ax.bar(bins, match_hist, alpha=0.5, label='Match', color='red')
                                    ax.set_title(f'Distance: {dist:.3f}')
                                    ax.legend(fontsize='x-small')
                                    ax.tick_params(axis='both', which='major', labelsize='x-small')
                                    plt.tight_layout()
                                    col.pyplot(fig)
                                    plt.close()
                                
                                # Add spatial histogram comparison only for Spatial Histogram method
                                if retrieval_method == "Spatial Histogram":
                                    query_img_bgr = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
                                    match_img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                    
                                    query_hist = compute_spatial_histogram(
                                        query_img_bgr,
                                        r_bins=config['r'],
                                        g_bins=config['g'],
                                        b_bins=config['b'],
                                        grid_size=(config['grid_size'], config['grid_size'])
                                    )
                                    match_hist = compute_spatial_histogram(
                                        match_img_bgr,
                                        r_bins=config['r'],
                                        g_bins=config['g'],
                                        b_bins=config['b'],
                                        grid_size=(config['grid_size'], config['grid_size'])
                                    )
                                    
                                    fig, ax = plt.subplots(figsize=(4, 3))
                                    bins = range(len(query_hist))
                                    ax.bar(bins, query_hist, alpha=0.5, label='Query', color='blue')
                                    ax.bar(bins, match_hist, alpha=0.5, label='Match', color='red')
                                    ax.set_title(f'Distance: {dist:.3f}')
                                    ax.legend(fontsize='x-small')
                                    ax.tick_params(axis='both', which='major', labelsize='x-small')
                                    plt.tight_layout()
                                    col.pyplot(fig)
                                    plt.close()

                # Add PR Curve and Confusion Matrix section
                st.subheader("Performance Analysis")
                
                # Calculate precision and recall
                if query_class:
                    precisions = []
                    recalls = []
                    retrieved_classes = []
                    
                    total_relevant = sum(1 for f in os.listdir(IMAGE_PATH) 
                                       if query_class in get_class_from_filename(f))
                    
                    relevant_found = 0
                    for i, (_, img_path, _) in enumerate(results, 1):
                        img_class = get_class_from_filename(os.path.basename(img_path))
                        retrieved_classes.append(img_class)
                        
                        if img_class == query_class:
                            relevant_found += 1
                        
                        precision = relevant_found / i
                        recall = relevant_found / total_relevant
                        
                        precisions.append(precision)
                        recalls.append(recall)
                    
                    # Plot PR Curve
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # PR Curve
                    ax1.plot(recalls, precisions, 'b-', linewidth=2)
                    ax1.set_xlabel('Recall')
                    ax1.set_ylabel('Precision')
                    ax1.set_title('Precision-Recall Curve')
                    ax1.grid(True)
                    
                    # Confusion Matrix for top 20 results
                    classes = sorted(list(set(retrieved_classes[:20])))
                    confusion = np.zeros((len(classes), len(classes)))
                    
                    # Fill confusion matrix
                    class_to_idx = {c: i for i, c in enumerate(classes)}
                    query_class = normalize_class_name(query_class)
                    query_idx = class_to_idx[query_class]
                    
                    for retrieved_class in retrieved_classes[:20]:
                        retrieved_idx = class_to_idx[retrieved_class]
                        confusion[query_idx, retrieved_idx] += 1
                    
                    # Plot confusion matrix
                    im = ax2.imshow(confusion, cmap='Blues')
                    ax2.set_xticks(range(len(classes)))
                    ax2.set_yticks(range(len(classes)))
                    ax2.set_xticklabels(classes, rotation=45, ha='right')
                    ax2.set_yticklabels(classes)
                    ax2.set_title('Confusion Matrix (Top 20)')
                    
                    # Add text annotations to show values
                    for i in range(len(classes)):
                        for j in range(len(classes)):
                            text = ax2.text(j, i, f'{int(confusion[i, j])}',
                                          ha='center', va='center',
                                          color='white' if confusion[i, j] > confusion.max()/2 else 'black')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax2)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    # Print metrics
                    st.write("\nRetrieval Metrics:")
                    if precisions:  # Only try to access if we have precisions
                        if len(precisions) > 9:
                            st.write(f"Precision@10: {precisions[9]:.3f}")
                        if len(precisions) > 19:
                            st.write(f"Precision@20: {precisions[19]:.3f}")
                        st.write(f"Average Precision: {np.mean(precisions):.3f}")
                    else:
                        st.write("No precision metrics available for this query")

def process_histogram_query(query_img, config, query_path=None):
    """Process query using global histogram"""
    # Convert query image to BGR if it's in RGB
    query_img_bgr = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
    
    # Compute query histogram
    query_hist = compute_global_histogram(
        query_img_bgr,
        r_bins=config['r'],
        g_bins=config['g'],
        b_bins=config['b']
    )

    distances = []
    image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.bmp')]
    
    # Dictionary of distance functions
    distance_functions = {
        'Euclidean': euclidean_distance,
        'Manhattan': manhattan_distance,
        'Chi-Square': chi_square_distance,
        'Intersection': intersection_distance,
        'Mahalanobis': mahalanobis_distance,
        'Cosine': cosine_distance
    }
    
    distance_func = distance_functions[config['distance_metric']]
    
    for filename in image_files:
        img_path = os.path.join(IMAGE_PATH, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Compute histogram for database image
        hist = compute_global_histogram(
            img,
            r_bins=config['r'],
            g_bins=config['g'],
            b_bins=config['b']
        )
        
        # Calculate distance
        dist = distance_func(query_hist, hist)
        
        # Check if this is the query image
        is_query = False
        if query_path and os.path.abspath(img_path) == os.path.abspath(query_path):
            is_query = True
            dist = 0.0
            
        distances.append((dist, img_path, is_query))
    
    return sorted(distances)

def compute_spatial_histogram(img, r_bins=8, g_bins=8, b_bins=8, grid_size=(2, 2)):
    """
    Compute spatial color histogram for an image.
    """
    height, width = img.shape[:2]
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]
    
    # Initialize the final histogram
    total_bins = r_bins * g_bins * b_bins
    spatial_hist = np.zeros(total_bins * grid_size[0] * grid_size[1])
    
    # Compute histogram for each cell
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Get cell coordinates
            y_start = i * cell_height
            y_end = (i + 1) * cell_height if i < grid_size[0] - 1 else height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width if j < grid_size[1] - 1 else width
            
            # Extract cell
            cell = img[y_start:y_end, x_start:x_end]
            
            # Compute histogram for cell
            hist = cv2.calcHist(
                [cell], 
                [0, 1, 2], 
                None, 
                [b_bins, g_bins, r_bins], 
                [0, 256, 0, 256, 0, 256]
            ).flatten()
            
            # Normalize cell histogram
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist = hist / hist_sum
            
            # Store in the spatial histogram
            start_idx = (i * grid_size[1] + j) * total_bins
            spatial_hist[start_idx:start_idx + total_bins] = hist
    
    # Normalize the entire spatial histogram
    hist_sum = spatial_hist.sum()
    if hist_sum > 0:
        spatial_hist = spatial_hist / hist_sum
        
    return spatial_hist

def process_spatial_histogram_query(query_img, config, query_path=None):
    """Process query using spatial histogram"""
    try:
        # Convert query image to BGR if it's in RGB
        query_img_bgr = cv2.cvtColor(query_img, cv2.COLOR_RGB2BGR)
        
        # Compute query histogram
        query_hist = compute_spatial_histogram(
            query_img_bgr,
            r_bins=config['r'],
            g_bins=config['g'],
            b_bins=config['b'],
            grid_size=(config['grid_size'], config['grid_size'])
        )

        distances = []
        image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.bmp')]
        
        for filename in image_files:
            img_path = os.path.join(IMAGE_PATH, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Compute histogram for database image
            hist = compute_spatial_histogram(
                img,
                r_bins=config['r'],
                g_bins=config['g'],
                b_bins=config['b'],
                grid_size=(config['grid_size'], config['grid_size'])
            )
            
            # Calculate distance using chi-square distance for better results
            dist = chi_square_distance(query_hist, hist)
            
            # Check if this is the query image
            is_query = False
            if query_path and os.path.abspath(img_path) == os.path.abspath(query_path):
                is_query = True
                dist = 0.0
                
            distances.append((dist, img_path, is_query))
        
        # Sort by distance
        sorted_results = sorted(distances, key=lambda x: x[0])
        
        # Print some debug information
        st.write(f"Found {len(distances)} images to compare")
        if len(sorted_results) > 0:
            st.write(f"Top 3 distances: {[x[0] for x in sorted_results[:3]]}")
        
        return sorted_results
        
    except Exception as e:
        st.error(f"Error in spatial histogram processing: {str(e)}")
        st.write("Traceback:", traceback.format_exc())
        return []

def process_pca_query(query_img, config, query_path=None):
    """Process query using PCA"""
    try:
        # Initialize PCA retrieval
        pca_retriever = PCARetrieval(n_components=config['n_components'])
        
        # Compute features for all images
        pca_retriever.compute_features(IMAGE_PATH)
        
        # Process query and get distances
        distances = []
        if query_path:
            # Create a temporary results directory
            temp_results_dir = os.path.join("results", "temp_pca")
            os.makedirs(temp_results_dir, exist_ok=True)
            
            # Compute features and distances directly
            query_img_bgr = cv2.imread(query_path)
            query_hist = compute_global_histogram(
                query_img_bgr, 
                r_bins=pca_retriever.config['r'], 
                g_bins=pca_retriever.config['g'], 
                b_bins=pca_retriever.config['b']
            ).flatten()
            
            # Transform query to PCA space
            query_centered = query_hist - pca_retriever.mean
            query_pca = pca_retriever.pca.transform(query_centered.reshape(1, -1))[0]
            
            # Compute distances using selected metric
            for i, feat in enumerate(pca_retriever.features):
                img_path = pca_retriever.files[i]
                is_query = os.path.abspath(img_path) == os.path.abspath(query_path)
                
                if is_query:
                    dist = 0.0
                else:
                    if config['distance_metric'] == 'Euclidean':
                        dist = euclidean_distance(query_pca, feat)
                    elif config['distance_metric'] == 'Manhattan':
                        dist = manhattan_distance(query_pca, feat)
                    elif config['distance_metric'] == 'Chi-Square':
                        dist = chi_square_distance(query_pca, feat)
                    elif config['distance_metric'] == 'Intersection':
                        dist = intersection_distance(query_pca, feat)
                    elif config['distance_metric'] == 'Mahalanobis':
                        dist = mahalanobis_distance(query_pca, feat)
                    elif config['distance_metric'] == 'Cosine':
                        dist = cosine_distance(query_pca, feat)
                    else:  # Default to Euclidean if unknown metric
                        dist = euclidean_distance(query_pca, feat)
                
                distances.append((dist, img_path, is_query))
        
        # Sort distances and ensure query image is first
        sorted_distances = sorted(distances, key=lambda x: (not x[2], x[0]))
        return sorted_distances
        
    except Exception as e:
        st.error(f"Error in PCA processing: {str(e)}")
        st.write("Traceback:", traceback.format_exc())
        return []

def process_bovw_query(query_img, config, query_path=None):
    """Process query using Bag of Visual Words"""
    try:
        st.write("Debug: Starting BoVW processing")
        st.write(f"Debug: Current working directory: {os.getcwd()}")
        
        # Initialize BoVW retrieval
        bovw = BoVWRetrieval(config)
        
        # Check both possible cache locations
        cache_dir = os.path.join(os.getcwd(), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Try both naming conventions
        cache_name1 = f"{config['detector']}_{config['codebook_size']}"
        cache_name2 = str(config['codebook_size'])
        
        possible_cache_paths = [
            os.path.join(cache_dir, f'cache_bovw_codebook_{cache_name1}.npy'),
            os.path.join(cache_dir, f'cache_bovw_codebook_{cache_name2}.npy'),
            os.path.join(os.getcwd(), '..', 'cache', f'cache_bovw_codebook_{cache_name1}.npy'),
            os.path.join(os.getcwd(), '..', 'cache', f'cache_bovw_codebook_{cache_name2}.npy')
        ]
        
        st.write("Debug: Checking possible cache locations:")
        for path in possible_cache_paths:
            st.write(f"Looking for: {path}")
            if os.path.exists(path):
                st.write(f"Found cache at: {path}")
                bovw.codebook = np.load(path)
                features_path = path.replace('codebook', 'features').replace('.npy', '.npz')
                if os.path.exists(features_path):
                    cache_data = np.load(features_path)
                    bovw.features = cache_data['features']
                    bovw.files = cache_data['files']
                    st.write("Loaded both codebook and features from cache")
                    break
        else:
            st.write(f"Debug: No cache found for {config['codebook_size']} clusters, performing clustering...")
            # Get all image paths
            image_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith('.bmp')]
            image_paths = [os.path.join(IMAGE_PATH, f) for f in image_files]
            
            # Build codebook using MiniBatchKMeans
            bovw.build_codebook(image_paths, config['codebook_size'])
            bovw.compute_bovw_features(image_paths)
            
            # Save to cache
            np.save(codebook_cache, bovw.codebook)
            np.savez(features_cache, features=bovw.features, files=bovw.files)
            st.write("Debug: Saved new cache files")
        
        # Process query image
        query_descriptors = bovw.extract_features(query_img)
        if query_descriptors is None:
            st.error("No features found in query image")
            return []
                
        # Create query histogram
        assignments = []
        for desc in query_descriptors:
            distances = np.linalg.norm(bovw.codebook - desc.reshape(1, -1), axis=1)
            nearest_word = np.argmin(distances)
            assignments.append(nearest_word)
                
        query_hist, _ = np.histogram(assignments, 
                                   bins=range(len(bovw.codebook) + 1), 
                                   density=True)
        query_hist = normalize(query_hist.reshape(1, -1))[0]
        
        # Compute distances
        distances = []
        for i, feat in enumerate(bovw.features):
            img_path = bovw.files[i]
            # Check if this is the query image
            is_query = query_path and os.path.abspath(img_path) == os.path.abspath(query_path)
            
            if is_query:
                dist = 0.0
            else:
                dist = compute_bovw_distance(query_hist, feat, config['distance_metric'])
            
            distances.append((dist, img_path, is_query))
                
        # Sort distances
        distances.sort(key=lambda x: x[0])
        
        return distances
            
    except Exception as e:
        st.error(f"Error in BoVW processing: {str(e)}")
        st.write("Traceback:", traceback.format_exc())
        return []

# Add the CNN processing function
def process_cnn_query(query_img, config, query_path=None):
    """Process query using CNN features"""
    try:
        # Initialize classifier
        classifier = CNNClassifier(model_name=config['model_name'])
        
        # Find the most recent classification folder (excluding data augmentation)
        classification_folders = [
            f for f in os.listdir('results') 
            if f.startswith('classification_') and not f.startswith('classification_data_')
        ]
        
        if not classification_folders:
            st.warning("No classification results found. Please train the model first.")
            return []
            
        # Sort by timestamp (newest first)
        latest_folder = sorted(classification_folders, reverse=True)[0]
        model_folder = os.path.join('results', latest_folder, f'cnn_{config["model_name"]}')
        model_path = os.path.join(model_folder, 'best_model.pth')
        
        if not os.path.exists(model_path):
            # Look for model in other folders
            for folder in sorted(classification_folders, reverse=True):
                test_path = os.path.join('results', folder, f'cnn_{config["model_name"]}', 'best_model.pth')
                if os.path.exists(test_path):
                    model_path = test_path
                    break
            else:
                st.warning(f"No model found for {config['model_name']} in any classification folder")
                return []
        
        classifier.model.load_state_dict(torch.load(model_path, map_location=classifier.device))
        classifier.model.eval()
        
        # Process query image
        with torch.no_grad():
            x = classifier.transform(query_img).unsqueeze(0).to(classifier.device)
            outputs = classifier.model(x)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)
        
        # Display results
        st.subheader("Classification Results")
        predicted_class = pred.item() + 1
        class_name = get_class_from_filename(f"{predicted_class}_")
        
        st.write(f"**Predicted Class:** {class_name}")
        st.write(f"**Confidence:** {conf.item():.2%}")
        
        # Display top-3 predictions
        st.subheader("Top 3 Predictions")
        probs = probs.cpu().numpy()[0]
        top3_idx = np.argsort(probs)[-3:][::-1]
        
        # Create probability distribution plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot all class probabilities
        class_names = [get_class_from_filename(f"{i+1}_") for i in range(20)]
        ax1.bar(class_names, probs)
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.set_title('Probability Distribution Across All Classes')
        ax1.set_ylabel('Probability')
        
        # Plot top 3 probabilities
        top3_names = [get_class_from_filename(f"{idx+1}_") for idx in top3_idx]
        top3_probs = probs[top3_idx]
        ax2.bar(top3_names, top3_probs)
        ax2.set_title('Top 3 Predictions')
        ax2.set_ylabel('Probability')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Find and display similar images
        st.subheader("Similar Images from Predicted Class")
        
        # Get all images from the predicted class
        similar_images = []
        for img_file in os.listdir(IMAGE_PATH):
            if img_file.startswith(f"{predicted_class}_"):
                img_path = os.path.join(IMAGE_PATH, img_file)
                img = cv2.imread(img_path)
                if img is not None:  # Only add if image was successfully loaded
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    similar_images.append((img_path, img))
        
        # Display up to 10 similar images in two rows
        if similar_images:
            # First row (1-5)
            st.write("Top 5 Similar Images:")
            cols1 = st.columns(5)
            for i, col in enumerate(cols1):
                if i < min(5, len(similar_images)):
                    img_path, img = similar_images[i]
                    col.image(img, use_container_width=True)
                    img_class = get_class_from_filename(os.path.basename(img_path))
                    col.write(f"Class: {img_class}")
            
            # Second row (6-10)
            if len(similar_images) > 5:
                st.write("More Similar Images:")
                cols2 = st.columns(5)
                for i, col in enumerate(cols2):
                    idx = i + 5
                    if idx < min(10, len(similar_images)):
                        img_path, img = similar_images[idx]
                        col.image(img, use_container_width=True)
                        img_class = get_class_from_filename(os.path.basename(img_path))
                        col.write(f"Class: {img_class}")
        else:
            st.warning(f"No similar images found for class: {class_name}")
        
        # Return empty distances list to maintain compatibility with other methods
        return []
        
    except Exception as e:
        st.error(f"Error in CNN processing: {str(e)}")
        st.write("Traceback:", traceback.format_exc())
        return []

# Add the SVM processing function
def process_svm_query(query_img, config, query_path=None):
    """Process query using SVM classifier"""
    try:
        # Initialize classifier
        classifier = ImageClassifier()
        
        # Find the most recent classification folder (excluding data augmentation)
        classification_folders = [
            f for f in os.listdir('results') 
            if f.startswith('classification_') and not f.startswith('classification_data_')
        ]
        
        if not classification_folders:
            st.warning("No classification results found. Please train the model first.")
            return []
            
        # Sort by timestamp (newest first)
        latest_folder = sorted(classification_folders, reverse=True)[0]
        model_folder = os.path.join('results', latest_folder, 'svm')
        model_path = os.path.join(model_folder, 'svm_model.pkl')
        
        if not os.path.exists(model_path):
            st.warning("""
            No SVM model found. Please train the SVM model first using:
            ```
            python run_classification.py --classifier svm --feature_type bovw
            ```
            or
            ```
            python run_classification.py --classifier svm --feature_type spatial
            ```
            """)
            return []
        
        classifier.load_model(model_path)
        
        # Process query image and get prediction
        prediction = classifier.predict(query_img)
        predicted_class = prediction + 1  # Adjust for 1-based class numbering
        class_name = get_class_from_filename(f"{predicted_class}_")
        
        # Display results
        st.subheader("Classification Results")
        st.write(f"**Predicted Class:** {class_name}")
        
        # Find and display similar images
        st.subheader("Similar Images from Predicted Class")
        
        # Get all images from the predicted class
        similar_images = []
        for img_file in os.listdir(IMAGE_PATH):
            if img_file.startswith(f"{predicted_class}_"):
                img_path = os.path.join(IMAGE_PATH, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    similar_images.append((img_path, img))
        
        # Display up to 10 similar images in two rows
        if similar_images:
            # First row (1-5)
            st.write("Top 5 Similar Images:")
            cols1 = st.columns(5)
            for i, col in enumerate(cols1):
                if i < min(5, len(similar_images)):
                    img_path, img = similar_images[i]
                    col.image(img, use_container_width=True)
                    img_class = get_class_from_filename(os.path.basename(img_path))
                    col.write(f"Class: {img_class}")
            
            # Second row (6-10)
            if len(similar_images) > 5:
                st.write("More Similar Images:")
                cols2 = st.columns(5)
                for i, col in enumerate(cols2):
                    idx = i + 5
                    if idx < min(10, len(similar_images)):
                        img_path, img = similar_images[idx]
                        col.image(img, use_container_width=True)
                        img_class = get_class_from_filename(os.path.basename(img_path))
                        col.write(f"Class: {img_class}")
        else:
            st.warning(f"No similar images found for class: {class_name}")
        
        return []
        
    except Exception as e:
        st.error(f"Error in SVM processing: {str(e)}")
        st.write("Traceback:", traceback.format_exc())
        return []

def compute_bovw_distance(query_hist, db_hist, distance_type='euclidean'):
    """
    Compute distance between two BoVW histograms using different metrics.
    
    Args:
        query_hist: Query image histogram
        db_hist: Database image histogram
        distance_type: Type of distance metric to use
    Returns:
        float: Distance value
    """
    if distance_type == 'euclidean':
        return np.linalg.norm(query_hist - db_hist)
    elif distance_type == 'manhattan':
        return np.sum(np.abs(query_hist - db_hist))
    elif distance_type == 'chi_square':
        eps = 1e-10  # To avoid division by zero
        return np.sum((query_hist - db_hist)**2 / (query_hist + db_hist + eps))
    elif distance_type == 'hellinger':
        return np.sqrt(np.sum((np.sqrt(query_hist) - np.sqrt(db_hist))**2)) / np.sqrt(2)
    elif distance_type == 'bhattacharyya':
        return -np.log(np.sum(np.sqrt(query_hist * db_hist)))
    elif distance_type == 'kl_divergence':
        eps = 1e-10
        return np.sum(query_hist * np.log((query_hist + eps) / (db_hist + eps)))
    elif distance_type == 'cosine':
        return 1 - np.dot(query_hist, db_hist) / \
               (np.linalg.norm(query_hist) * np.linalg.norm(db_hist))
    else:
        return np.linalg.norm(query_hist - db_hist)  # Default to euclidean

def fast_kmeans_clustering(features, n_clusters):
    """
    Perform fast k-means clustering using MiniBatchKMeans.
    
    Args:
        features: Feature descriptors
        n_clusters: Number of clusters (codebook size)
    Returns:
        cluster_centers: Codebook vocabulary
    """
    # Initialize MiniBatchKMeans
    batch_size = min(1000, len(features))  # Adjust batch size based on data
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=42,
        max_iter=100,
        n_init='auto'
    )
    
    # Fit the model
    kmeans.fit(features)
    
    return kmeans.cluster_centers_

if __name__ == "__main__":
    main()