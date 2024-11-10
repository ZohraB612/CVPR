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

def plot_histogram_comparison(query_hist, match_hist, config, ax=None):
    """Plot histogram comparison between query and match."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    
    total_bins = config['r'] * config['g'] * config['b']
    x = np.arange(total_bins)
    
    # Plot both histograms
    ax.bar(x - 0.2, query_hist, 0.4, label='Query', alpha=0.6, color='blue')
    ax.bar(x + 0.2, match_hist, 0.4, label='Match', alpha=0.6, color='red')
    
    ax.set_xlabel('Bin Index')
    ax.set_ylabel('Normalized Count')
    ax.legend()
    ax.set_title(f'Histogram Comparison (R{config["r"]}G{config["g"]}B{config["b"]})')

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

def create_confusion_matrix(matches, query_class, num_classes=20):
    """Create confusion matrix for the retrieval results."""
    true_labels = [query_class] * 10
    pred_labels = [get_image_class(path) for _, path in matches[:10]]
    
    return confusion_matrix(
        true_labels, 
        pred_labels, 
        labels=range(num_classes)
    )

def save_confusion_matrix(matches, query_class, save_dir):
    """Save confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    
    cm = create_confusion_matrix(matches, query_class)
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=range(cm.shape[1]),
        yticklabels=range(cm.shape[0])
    )
    
    plt.title(f'Confusion Matrix for Query Class {query_class}\n(Top 10 Matches)')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=DPI, bbox_inches='tight')
    plt.close()
    
    return cm

def save_experiment_results(query_path, matches, pr_data, config, results_dir):
    """Save comprehensive results for report documentation."""
    query_class = get_image_class(query_path)
    experiment_dir = os.path.join(results_dir, f"query_{query_class}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save all visualizations
    save_match_visualization(query_path, matches, config, experiment_dir)
    ap = save_pr_curve(pr_data, config, experiment_dir)
    cm = save_confusion_matrix(matches, query_class, experiment_dir)
    
    # Save analysis report
    save_analysis_report(query_path, matches, config, ap, cm, experiment_dir)

def save_analysis_report(query_path, matches, config, ap, confusion_matrix, save_dir):
    """Save detailed analysis report in text format."""
    query_class = get_image_class(query_path)
    
    with open(os.path.join(save_dir, 'analysis.txt'), 'w') as f:
        # Header
        f.write(f"Analysis for Query Class: {query_class}\n")
        f.write(f"Configuration: R={config['r']}, G={config['g']}, B={config['b']}\n\n")
        
        # Quantization details
        f.write("Quantization Details:\n")
        f.write(f"- Red channel bins: {config['r']}\n")
        f.write(f"- Green channel bins: {config['g']}\n")
        f.write(f"- Blue channel bins: {config['b']}\n")
        f.write(f"- Total bins: {config['r'] * config['g'] * config['b']}\n\n")
        
        # Performance metrics
        correct_matches = sum(1 for _, path in matches[:10] 
                            if get_image_class(path) == query_class)
        
        f.write("Performance Metrics:\n")
        f.write(f"- Average Precision: {ap:.3f}\n")
        f.write(f"- Precision@10: {correct_matches/10:.3f}\n")
        f.write(f"- Correct matches in top 10: {correct_matches}\n\n")
        
        # Confusion matrix analysis
        f.write("Confusion Matrix Analysis:\n")
        f.write(f"- True Positives: {confusion_matrix[query_class, query_class]}\n")
        f.write(f"- False Positives: {confusion_matrix.sum(axis=0)[query_class] - confusion_matrix[query_class, query_class]}\n")
        f.write(f"- Most confused with: Class {confusion_matrix[query_class].argmax()} ")
        f.write(f"({confusion_matrix[query_class].max()} instances)\n\n")
        
        # Detailed matches analysis
        f.write("Top 10 Matches Analysis:\n")
        for i, (dist, path) in enumerate(matches[:10], 1):
            match_class = get_image_class(path)
            correct = "Correct" if match_class == query_class else "Incorrect"
            f.write(f"{i}. {path}\n")
            f.write(f"   Class: {match_class} ({correct})\n")
            f.write(f"   Distance: {dist:.4f}\n")