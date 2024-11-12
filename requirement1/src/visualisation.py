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
from src.class_mapping import CLASS_MAPPING

# Add this after the imports
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
    '17': 'street',
    '18': 'nature',
    '19': 'people',
    '20': 'boats',
}

def get_class_from_filename(filename):
    """Extract class number from filename and return corresponding class name"""
    class_num = filename.split('_')[0]
    # class_mapping = {
    #     '1': 'farm animals',
    #     '2': 'trees',
    #     '3': 'buildings',
    #     '4': 'airplanes',
    #     '5': 'cows',
    #     '6': 'faces',
    #     '7': 'cars',
    #     '8': 'bicycles',
    #     '9': 'sheep',
    #     '10': 'flowers',
    #     '11': 'signs',
    #     '12': 'birds',
    #     '13': 'books',
    #     '14': 'chairs',
    #     '15': 'cats',
    #     '16': 'dogs',
    #     '17': 'street',
    #     '18': 'nature',
    #     '19': 'people',
    #     '20': 'boats',
    # }
    return class_mapping.get(class_num, 'unknown')

def plot_histogram_comparison(query_hist, match_hist, config, ax):
    """
    Plot histogram comparison with support for multiple feature types.
    
    Args:
        query_hist: Query image histogram
        match_hist: Matching image histogram
        config: Configuration dictionary
        ax: Matplotlib axis object
    """
    # Calculate total number of bins
    total_bins = config['r'] * config['g'] * config['b']
    x = np.arange(total_bins)
    
    # Clear existing plot
    ax.clear()
    
    # Thick bars
    width = 0.8
    
    # Simple bar plot comparison
    ax.bar(x - width/2, query_hist, width, label='Query', alpha=0.8, color='blue')
    ax.bar(x + width/2, match_hist, width, label='Match', alpha=0.8, color='red')
    
    # Customize plot with larger font sizes
    ax.set_xlabel('Bin', fontsize=14)
    ax.set_ylabel('Normalized Count', fontsize=14)
    ax.legend(fontsize=14)
    
    # Set title based on configuration
    title = f'Histogram (R{config["r"]}G{config["g"]}B{config["b"]})'
    ax.set_title(title, fontsize=16, pad=20)
    
    # Adjust x-axis for readability
    if total_bins > 20:
        ax.set_xticks([])
    else:
        ax.set_xticks(x)
        ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Set y-axis limits
    ax.set_ylim([0, max(np.max(query_hist), np.max(match_hist)) * 1.1])
    ax.tick_params(axis='y', labelsize=12)

def save_match_visualization(query_path, distances, config, save_dir):
    """Create and save visualization of query and top matches with histograms."""
    # Create larger figure
    fig = plt.figure(figsize=(25, 35))
    
    # Adjust grid to make images larger relative to histograms
    gs = plt.GridSpec(11, 2, height_ratios=[1.5] + [2.5]*10)
    
    # Query image and histogram
    query_img = cv2.imread(query_path)
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    query_hist = compute_global_histogram(query_img, config['r'], config['g'], config['b'])
    
    # Plot query image
    ax_query = fig.add_subplot(gs[0, 0])
    ax_query.imshow(query_img)
    ax_query.set_title(f'Query Image\nClass: {get_image_class(query_path)}', 
                      fontsize=16, pad=20)
    ax_query.axis('off')
    
    # Plot query histogram
    ax_query_hist = fig.add_subplot(gs[0, 1])
    plot_histogram_comparison(query_hist, query_hist, config, ax_query_hist)
    ax_query_hist.set_title('Query Histogram', fontsize=16, pad=20)
    
    # Plot top matches with histograms
    for i, (dist, match_path) in enumerate(distances[:10], 1):
        match_img = cv2.imread(match_path)
        match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)
        match_hist = compute_global_histogram(match_img, config['r'], config['g'], config['b'])
        
        # Plot match image
        ax_match = fig.add_subplot(gs[i, 0])
        ax_match.imshow(match_img)
        match_class = get_image_class(match_path)
        correct = '✓' if match_class == get_image_class(query_path) else '✗'
        ax_match.set_title(f'Match {i} ({correct})\nClass: {match_class}\nDist: {dist:.3f}', 
                          fontsize=16, pad=20)
        ax_match.axis('off')
        
        # Plot histogram comparison
        ax_hist = fig.add_subplot(gs[i, 1])
        plot_histogram_comparison(query_hist, match_hist, config, ax_hist)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'search_results.png'), 
                dpi=300, 
                bbox_inches='tight')
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
    plt.title(f'Precision-Recall Curve for\nConfiguration: R{config["r"]}G{config["g"]}B{config["b"]}')
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return ap

def create_confusion_matrix(distances, query_class, num_display=20):
    """
    Create confusion matrix for retrieval results using raw counts.
    
    Args:
        distances (list): List of (distance, path) tuples
        query_class (str): Class of query image
        num_display (int): Number of top matches to consider
    
    Returns:
        tuple: (confusion_matrix, class_names)
    """
    # Get all possible classes in order
    all_classes = [class_mapping[str(i)] for i in range(1, 21)]
    
    # Initialize confusion matrix
    n_classes = len(all_classes)
    conf_matrix = np.zeros((n_classes, n_classes))
    
    try:
        # Get predictions for top N results
        predictions = []
        for _, path in distances[:num_display]:
            filename = os.path.basename(path)
            class_num = filename.split('_')[0]
            pred_class = class_mapping.get(class_num, 'unknown')
            predictions.append(pred_class)
        
        # For each predicted class, update confusion matrix with raw counts
        query_idx = all_classes.index(query_class)
        for pred_class in predictions:
            try:
                pred_idx = all_classes.index(pred_class)
                conf_matrix[query_idx, pred_idx] += 1  # Raw count increment
            except ValueError:
                print(f"Warning: Prediction class {pred_class} not found in class list")
                continue
                
    except Exception as e:
        print(f"Error creating confusion matrix: {str(e)}")
        print(f"Query class: {query_class}")
        print(f"Available classes: {all_classes}")
        return np.zeros((n_classes, n_classes)), all_classes
    
    return conf_matrix, all_classes

def save_confusion_matrix(distances, query_class, save_dir):
    """
    Save confusion matrix visualization with raw counts, showing only the query class row.
    
    Args:
        distances (list): List of (distance, path) tuples
        query_class (str): Class of query image
        save_dir (str): Directory to save visualization
    """
    # Create confusion matrix
    conf_matrix, classes = create_confusion_matrix(distances, query_class)
    
    # Get only the row corresponding to the query class
    query_idx = classes.index(query_class)
    conf_matrix_row = conf_matrix[query_idx:query_idx+1, :]
    
    # Convert matrix to integers
    conf_matrix_row = conf_matrix_row.astype(int)
    
    plt.figure(figsize=(15, 3))  # Adjusted figure size for single row
    
    # Create heatmap with raw counts
    sns.heatmap(conf_matrix_row, 
                annot=True, 
                fmt='d',  # Use integer format for raw counts
                cmap='Blues',
                xticklabels=classes,
                yticklabels=[query_class],
                cbar=False)  # Remove colorbar as it's less relevant for single row
    
    plt.title(f'Distribution of Retrieved Results for {query_class}', fontsize=14, pad=20)
    plt.xlabel('Retrieved Class', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return conf_matrix_row, classes

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
        pr_data: Tuple of (recalls, precisions) as numpy arrays
        
    Returns:
        float: Average Precision value
    """
    recalls, precisions = pr_data
    
    # Ensure inputs are numpy arrays and not empty
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0
        
    # Convert to numpy arrays if they aren't already
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    
    # Calculate area under PR curve using trapezoidal rule
    return np.trapz(precisions, recalls)

def save_experiment_results(query_path, distances, pr_data, config, results_dir):
    """
    Save all experiment results.
    
    Args:
        query_path (str): Path to query image
        distances (list): List of (distance, path) tuples
        pr_data: Precision-recall data
        config (dict): Configuration dictionary
        results_dir (str): Directory to save results
    """
    # Create experiment directory
    experiment_dir = os.path.join(
        results_dir,
        f"query_{os.path.splitext(os.path.basename(query_path))[0]}"
    )
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Get query class and prepare matches
    query_class = get_image_class(query_path)
    # Modified this line to handle (distance, path) tuples
    matches = [(get_image_class(path), distance) for distance, path in distances]
    
    # Calculate metrics
    ap = calculate_ap(pr_data)
    
    # Save visualizations
    save_match_visualization(query_path, distances, config, experiment_dir)
    save_pr_curve(pr_data, config, experiment_dir)
    conf_matrix, classes = save_confusion_matrix(distances, query_class, experiment_dir)
    
    # Save analysis report
    save_analysis_report(query_path, distances, config, ap, (conf_matrix, classes), experiment_dir)

def save_analysis_report(query_path, distances, config, ap, confusion_data, save_dir):
    """
    Save detailed analysis report including retrieval distribution insights.
    
    Args:
        query_path (str): Path to query image
        distances (list): List of (distance, path) tuples
        config (dict): Configuration dictionary
        ap (float): Average precision
        confusion_data (tuple): (confusion_matrix_row, class_names)
        save_dir (str): Directory to save report
    """
    query_class = get_image_class(query_path)
    conf_matrix_row, classes = confusion_data
    
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
            correct_retrievals = int(conf_matrix_row[0, query_idx])  # Updated to use row index 0
            f.write(f"Correct retrievals (same class): {correct_retrievals}\n")
        except ValueError:
            f.write("Query class not found in confusion matrix classes\n")
        
        # Find most retrieved classes
        class_counts = []
        for i, cls in enumerate(classes):
            count = int(conf_matrix_row[0, i])  # Updated to use row index 0
            if count > 0:
                class_counts.append((cls, count))
        
        class_counts.sort(key=lambda x: x[1], reverse=True)
        
        f.write("\nMost retrieved classes:\n")
        for cls, count in class_counts[:3]:  # Top 3 most retrieved
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

def plot_pr_curve(pr_data, config, title="Precision-Recall Curve"):
    """
    Plot precision-recall curve that can be displayed both in files and Streamlit.
    
    Args:
        pr_data: Tuple of (recalls, precisions)
        config: Configuration dictionary
        title: Optional title for the plot
    
    Returns:
        fig: matplotlib figure object that can be used with st.pyplot()
    """
    recalls, precisions = pr_data
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot PR curve with filled area
    ax.plot(recalls, precisions, 'b-', linewidth=2)
    ax.fill_between(recalls, precisions, alpha=0.2)
    
    # Calculate and display Average Precision
    ap = np.trapz(precisions, recalls)
    ax.text(0.5, 0.95, f'AP: {ap:.3f}', transform=ax.transAxes)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{title}\nConfiguration: R{config["r"]}G{config["g"]}B{config["b"]}')
    ax.grid(True)
    
    plt.tight_layout()
    return fig

def plot_confusion_matrix(distances, query_class):
    """
    Create confusion matrix visualization for Streamlit.
    
    Args:
        distances: List of (distance, path) tuples
        query_class: Class of the query image
    
    Returns:
        fig: matplotlib figure object that can be used with st.pyplot()
    """
    conf_matrix, classes = create_confusion_matrix(distances, query_class)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap with normalized values
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='Blues',
                xticklabels=classes,
                yticklabels=classes,
                vmin=0,
                vmax=1,
                ax=ax)
    
    ax.set_title('Normalized Confusion Matrix for Retrieved Results')
    ax.set_xlabel('Retrieved Class')
    ax.set_ylabel('Query Class')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    return fig

def compute_average_pr_curve(all_pr_data):
    """
    Compute average precision-recall curve from multiple queries.
    
    Args:
        all_pr_data: List of (recalls, precisions) tuples from different queries
    
    Returns:
        tuple: (mean_recalls, mean_precisions, std_precisions)
    """
    # Interpolate all PR curves to a fixed set of recall points
    recall_points = np.linspace(0, 1, 100)
    interpolated_precisions = []
    
    for recalls, precisions in all_pr_data:
        if len(recalls) > 1:  # Only process valid PR curves
            # Interpolate precision values for fixed recall points
            interp_precision = np.interp(recall_points, recalls, precisions)
            interpolated_precisions.append(interp_precision)
    
    if not interpolated_precisions:
        return None, None, None
        
    # Convert to numpy array for calculations
    interpolated_precisions = np.array(interpolated_precisions)
    
    # Calculate mean and standard deviation
    mean_precisions = np.mean(interpolated_precisions, axis=0)
    std_precisions = np.std(interpolated_precisions, axis=0)
    
    return recall_points, mean_precisions, std_precisions

def plot_average_pr_curve(all_pr_data, config, title="Average Precision-Recall Curve"):
    """
    Plot average precision-recall curve with confidence intervals.
    
    Args:
        all_pr_data: List of (recalls, precisions) tuples from different queries
        config: Configuration dictionary
        title: Plot title
    
    Returns:
        fig: matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute average PR curve
    recalls, mean_precisions, std_precisions = compute_average_pr_curve(all_pr_data)
    
    if recalls is None:
        ax.text(0.5, 0.5, "No valid PR curves available", 
                ha='center', va='center')
        return fig
    
    # Plot mean PR curve
    ax.plot(recalls, mean_precisions, 'b-', linewidth=2, label='Mean PR Curve')
    
    # Plot confidence interval
    ax.fill_between(recalls, 
                   np.maximum(0, mean_precisions - std_precisions),
                   np.minimum(1, mean_precisions + std_precisions),
                   alpha=0.2, color='b', label='±1 std dev')
    
    # Calculate mean Average Precision
    mean_ap = np.trapz(mean_precisions, recalls)
    
    # Add mAP to plot
    ax.text(0.05, 0.95, f'mAP: {mean_ap:.3f}', 
            transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Customize plot
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{title}\nConfiguration: R{config["r"]}G{config["g"]}B{config["b"]}')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_class_wise_pr_curves(all_pr_data, class_names, config):
    """
    Plot PR curves grouped by class.
    
    Args:
        all_pr_data: Dictionary mapping class names to lists of PR curves
        class_names: List of class names
        config: Configuration dictionary
    
    Returns:
        fig: matplotlib figure object
    """
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))  # 4x5 grid for 20 classes
    axes = axes.flatten()
    
    for idx, class_name in enumerate(class_names):
        ax = axes[idx]
        class_pr_data = all_pr_data.get(class_name, [])
        
        if class_pr_data:
            recalls, mean_precisions, std_precisions = compute_average_pr_curve(class_pr_data)
            
            if recalls is not None:
                ax.plot(recalls, mean_precisions, 'b-', linewidth=2)
                ax.fill_between(recalls, 
                              np.maximum(0, mean_precisions - std_precisions),
                              np.minimum(1, mean_precisions + std_precisions),
                              alpha=0.2, color='b')
                
                # Calculate AP for this class
                ap = np.trapz(mean_precisions, recalls)
                ax.text(0.05, 0.95, f'AP: {ap:.3f}', 
                       transform=ax.transAxes,
                       bbox=dict(facecolor='white', alpha=0.8))
        
        ax.set_title(class_name)
        ax.grid(True)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Only add labels for bottom row and leftmost column
        if idx >= 15:  # Bottom row
            ax.set_xlabel('Recall')
        if idx % 5 == 0:  # Leftmost column
            ax.set_ylabel('Precision')
    
    plt.suptitle(f'Class-wise PR Curves\nConfiguration: R{config["r"]}G{config["g"]}B{config["b"]}')
    plt.tight_layout()
    return fig