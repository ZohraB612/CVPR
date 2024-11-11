"""
Visualization module for image retrieval results.
Handles plotting and saving of results, including PR curves and match visualizations.

Author: Zohra Bouchamaoui
Student ID: 6848526
Module: EEE3032 Computer Vision and Pattern Recognition
"""

import os
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.utils import get_image_class
from src.histogram import compute_global_histogram
from config.settings import DPI, CONFIGS
import numpy as np
from sklearn.preprocessing import LabelEncoder

def plot_histogram_comparison(query_hist, match_hist, config, ax):
    """Plot histogram comparison."""
    feature_dim = len(query_hist)
    x = np.arange(feature_dim)
    
    # Determine if using spatial grid features
    is_spatial = 'grid' in config and config['grid'] != (1, 1)
    
    if is_spatial:
        # For spatial features, plot as multiple subplots
        grid_size = config['grid']
        features_per_cell = feature_dim // (grid_size[0] * grid_size[1])
        
        # Clear current axis and create grid of subplots
        ax.clear()
        fig = ax.figure
        fig.clear()
        
        # Create subplots for each grid cell
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                cell_idx = i * grid_size[1] + j
                cell_start = cell_idx * features_per_cell
                cell_end = (cell_idx + 1) * features_per_cell
                
                # Create subplot
                subplot_idx = i * grid_size[1] + j + 1
                ax_cell = fig.add_subplot(grid_size[0], grid_size[1], subplot_idx)
                
                # Plot cell features
                ax_cell.bar(x[:features_per_cell], 
                          query_hist[cell_start:cell_end],
                          alpha=0.6, color='blue', label='Query' if cell_idx == 0 else "")
                ax_cell.bar(x[:features_per_cell],
                          match_hist[cell_start:cell_end],
                          alpha=0.6, color='red', label='Match' if cell_idx == 0 else "")
                
                ax_cell.set_title(f'Cell ({i},{j})')
                
                if cell_idx == 0:
                    ax_cell.legend()
                
        plt.tight_layout()
        
    else:
        # Original histogram plotting for non-spatial features
        ax.bar(x - 0.2, query_hist, 0.4, label='Query', alpha=0.6, color='blue')
        ax.bar(x + 0.2, match_hist, 0.4, label='Match', alpha=0.6, color='red')
        ax.set_xlabel('Bin')
        ax.set_ylabel('Normalized Count')
        ax.legend()

def save_match_visualization(query_path, matches, config, save_dir):
    """Create and save visualization of query and top matches with histograms."""
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 25))
    gs = plt.GridSpec(11, 2, height_ratios=[1] + [2]*10)
    
    # Query image and histogram
    query_img = cv2.imread(query_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    query_hist = compute_global_histogram(query_img, config['r'], config['g'], config['b'])
    
    # Plot query image
    ax_query = fig.add_subplot(gs[0, 0])
    ax_query.imshow(query_img)
    ax_query.set_title(f'Query Image\nClass: {get_image_class(query_path)}')
    ax_query.axis('off')
    
    # Plot query histogram
    ax_query_hist = fig.add_subplot(gs[0, 1])
    plot_histogram_comparison(query_hist, query_hist, config, ax_query_hist)
    ax_query_hist.set_title('Query Histogram')
    
    # Plot top matches with histograms
    for i, (dist, match_path) in enumerate(matches[:10], 1):
        match_img = cv2.imread(match_path)
        match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
        match_hist = compute_global_histogram(match_img, config['r'], config['g'], config['b'])
        
        # Plot match image
        ax_match = fig.add_subplot(gs[i, 0])
        ax_match.imshow(match_img)
        match_class = get_image_class(match_path)
        correct = '✓' if match_class == get_image_class(query_path) else '✗'
        ax_match.set_title(f'Match {i} ({correct})\nClass: {match_class}\nDist: {dist:.3f}')
        ax_match.axis('off')
        
        # Plot histogram comparison
        ax_hist = fig.add_subplot(gs[i, 1])
        plot_histogram_comparison(query_hist, match_hist, config, ax_hist)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'search_results.png'), dpi=DPI, bbox_inches='tight')
    plt.close()

def save_pr_curve(pr_data, config, save_dir):
    """Save precision-recall curve visualization."""
    recalls, precisions = pr_data
    plt.figure(figsize=(8, 6))
    
    # Plot PR curve with filled area
    plt.plot(recalls, precisions, 'b-', linewidth=2)
    plt.fill_between(recalls, precisions, alpha=0.2)
    
    # Calculate and display Average Precision
    ap = np.trapz(precisions, recalls)
    plt.text(0.5, 0.95, f'AP: {ap:.3f}', transform=plt.gca().transAxes)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve\nConfiguration: R{config["r"]}G{config["g"]}B{config["b"]}')
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return ap

def create_confusion_matrix(distances, query_class, num_display=20):
    """
    Create confusion matrix for retrieval results.
    
    Args:
        distances: List of (distance, path) tuples
        query_class: Class of the query image
        num_display: Number of top matches to consider
    
    Returns:
        tuple: (results_distribution, class_names)
    """
    # Get all unique classes
    class_counts = {}
    
    # Count occurrences of each class in top N results
    for _, path in distances[:num_display]:
        predicted_class = get_image_class(path)
        class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
    
    # Create a single-row matrix showing distribution of retrieved results
    class_names = sorted(class_counts.keys())
    distribution = np.zeros((1, len(class_names)), dtype=np.float32)
    
    for i, class_name in enumerate(class_names):
        distribution[0, i] = float(class_counts.get(class_name, 0))
    
    return distribution, class_names

def save_confusion_matrix(distances, query_class, save_dir):
    """Save retrieval results distribution visualization."""
    distribution, classes = create_confusion_matrix(distances, query_class)
    
    plt.figure(figsize=(15, 4))
    
    # Create heatmap showing distribution of retrieved classes
    sns.heatmap(distribution, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=classes,
                yticklabels=['Retrieved Results'])
    
    plt.title(f'Distribution of Retrieved Classes (Top 20) for Query: {query_class}')
    plt.xlabel('Class')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'retrieval_distribution.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return distribution, classes

def get_class_name(class_id):
    """
    Convert numeric class ID to human-readable name.
    Add your class mapping here based on the dataset.
    """
    class_names = {
        '1': 'building',
        '2': 'grass',
        '3': 'tree',
        '4': 'cow',
        '5': 'sheep',
        '6': 'sky',
        '7': 'airplane',
        '8': 'water',
        '9': 'face',
        '10': 'car',
        '11': 'bicycle',
        '12': 'flower',
        '13': 'sign',
        '14': 'bird',
        '15': 'book',
        '16': 'chair',
        '17': 'road',
        '18': 'cat',
        '19': 'dog',
        '20': 'body'
    }
    return class_names.get(str(class_id), str(class_id))

def calculate_ap(pr_data):
    """
    Calculate Average Precision from precision-recall data.
    
    Args:
        pr_data: Tuple of (recalls, precisions)
        
    Returns:
        float: Average Precision value
    """
    recalls, precisions = pr_data
    # Calculate area under PR curve using trapezoidal rule
    return np.trapz(precisions, recalls)

def save_experiment_results(query_path, distances, pr_data, config, results_dir):
    """Save all experiment results."""
    # Create experiment directory
    experiment_dir = os.path.join(
        results_dir,
        f"query_{os.path.splitext(os.path.basename(query_path))[0]}"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Get query class and prepare matches
    query_class = get_image_class(query_path)
    matches = [(get_image_class(path), get_image_class(path)) for _, path in distances]
    
    # Calculate metrics
    ap = calculate_ap(pr_data)
    
    # Save visualizations
    save_match_visualization(query_path, distances, config, experiment_dir)
    save_pr_curve(pr_data, config, experiment_dir)
    confusion_data = save_confusion_matrix(matches, query_class, experiment_dir)
    
    # Save analysis report
    save_analysis_report(query_path, distances, config, ap, confusion_data, experiment_dir)

def save_analysis_report(query_path, distances, config, ap, confusion_data, save_dir):
    """Save detailed analysis report including retrieval distribution insights."""
    query_class = get_image_class(query_path)
    distribution, classes = confusion_data
    
    with open(os.path.join(save_dir, 'analysis.txt'), 'w') as f:
        f.write(f"Analysis Report for Query Image: {os.path.basename(query_path)}\n")
        f.write(f"Query Class: {query_class}\n\n")
        
        # Configuration details
        f.write("Color Quantization Configuration:\n")
        f.write(f"R: {config['r']} bins\n")
        f.write(f"G: {config['g']} bins\n")
        f.write(f"B: {config['b']} bins\n")
        f.write(f"Total bins: {config['r'] * config['g'] * config['b']}\n\n")
        
        # Performance metrics
        f.write("Retrieval Performance:\n")
        f.write(f"Average Precision: {ap:.4f}\n")
        
        # Top-N accuracy
        top_10_correct = sum(1 for _, path in distances[:10] 
                           if get_image_class(path) == query_class)
        top_20_correct = sum(1 for _, path in distances[:20] 
                           if get_image_class(path) == query_class)
        
        f.write(f"Precision@10: {top_10_correct/10:.3f}\n")
        f.write(f"Precision@20: {top_20_correct/20:.3f}\n\n")
        
        # Distribution analysis
        f.write("Retrieval Distribution Analysis (Top 20):\n")
        
        # Find query class in distribution
        try:
            query_idx = classes.index(query_class)
            correct_retrievals = int(distribution[0, query_idx])
            f.write(f"Correct retrievals (same class): {correct_retrievals}\n")
        except ValueError:
            f.write("Query class not found in top 20 retrievals\n")
        
        # Find most retrieved classes
        class_counts = [(cls, int(distribution[0, i])) for i, cls in enumerate(classes)]
        class_counts.sort(key=lambda x: x[1], reverse=True)
        
        f.write("\nMost retrieved classes:\n")
        for cls, count in class_counts[:3]:  # Top 3 most retrieved
            if count > 0:  # Only show classes that were actually retrieved
                f.write(f"- {cls}: {count} instances\n")
        
        # Distance analysis
        f.write("\nDistance Analysis:\n")
        f.write(f"Minimum distance: {distances[0][0]:.4f}\n")
        f.write(f"Maximum distance: {distances[-1][0]:.4f}\n")
        
        # Top 10 matches details
        f.write("\nTop 10 Retrieved Images:\n")
        for i, (dist, path) in enumerate(distances[:10], 1):
            match_class = get_image_class(path)
            correct = "✓" if match_class == query_class else "✗"
            f.write(f"{i}. {os.path.basename(path)} - {match_class} {correct} (distance: {dist:.4f})\n")