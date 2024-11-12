import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def create_combined_plots(base_path, query_folder, plot_type):
    pca_folders = ['pca_16', 'pca_32', 'pca_64', 'pca_128']
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f'Comparison of {plot_type} for {query_folder}\nacross PCA configurations', fontsize=16)
    
    for idx, pca in enumerate(pca_folders):
        row = idx // 2
        col = idx % 2
        
        img_path = os.path.join(base_path, pca, query_folder, f'{plot_type}.png')
        print(f"Checking path: {img_path}")
        
        if os.path.exists(img_path):
            try:
                img = plt.imread(img_path)
                axes[row, col].imshow(img)
                axes[row, col].set_title(f'PCA {pca.split("_")[1]} components')
                axes[row, col].axis('off')
                print(f"Successfully loaded image from {img_path}")
            except Exception as e:
                print(f"Error loading image {img_path}: {str(e)}")
        else:
            print(f"File not found: {img_path}")
    
    plt.tight_layout()
    # Create a directory for combined plots if it doesn't exist
    combined_dir = os.path.join(base_path, 'combined_plots')
    os.makedirs(combined_dir, exist_ok=True)
    
    # Save with query folder name in filename
    save_path = os.path.join(combined_dir, f'combined_{plot_type}_{query_folder}.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved combined plot to: {save_path}")
    plt.close()

def get_query_folders(base_path):
    # Get list of query folders from pca_16 directory (assuming all PCA folders have same query folders)
    pca_16_path = os.path.join(base_path, 'pca_16')
    if os.path.exists(pca_16_path):
        return [d for d in os.listdir(pca_16_path) if d.startswith('query_')]
    return []

# Base path to your results directory
base_path = '/home/zohrab/CVPR/requirement1/results'

# Get all query folders
query_folders = get_query_folders(base_path)
print(f"Found query folders: {query_folders}")

# Create combined plots for each query folder
for query_folder in query_folders:
    print(f"\nProcessing {query_folder}")
    create_combined_plots(base_path, query_folder, 'pr_curve')
    create_combined_plots(base_path, query_folder, 'retrieval_distribution')