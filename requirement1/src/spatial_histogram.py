import cv2
import numpy as np

def compute_spatial_histogram(img, r_bins=8, g_bins=8, b_bins=4, grid_size=(2, 2)):
    """
    Compute spatial grid histogram combining color and texture features.
    
    Args:
        img: Input image (BGR format)
        r_bins, g_bins, b_bins: Color quantization bins
        grid_size: Tuple of (rows, cols) for spatial grid
    
    Returns:
        np.ndarray: Concatenated features from all grid cells
    """
    height, width = img.shape[:2]
    cell_height = height // grid_size[0]
    cell_width = width // grid_size[1]
    
    # Initialize list to store features from each cell
    cell_features_list = []
    
    # Create Gabor filters for texture
    gabor_filters = []
    num_orientations = 8  # Angular quantization for texture
    for theta in np.arange(0, np.pi, np.pi / num_orientations):
        kernel = cv2.getGaborKernel(
            ksize=(16, 16),
            sigma=4.0,
            theta=theta,
            lambd=10.0,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )
        kernel /= 1.5 * kernel.sum()
        gabor_filters.append(kernel)
    
    # Process each grid cell
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Extract cell
            y1 = i * cell_height
            y2 = (i + 1) * cell_height
            x1 = j * cell_width
            x2 = (j + 1) * cell_width
            cell = img[y1:y2, x1:x2]
            
            # 1. Compute color histogram for this cell
            color_hist = cv2.calcHist(
                [cell], 
                [0, 1, 2], 
                None, 
                [b_bins, g_bins, r_bins], 
                [0, 256, 0, 256, 0, 256]
            )
            color_hist = color_hist.flatten()
            # Normalize color histogram
            color_hist = color_hist / (np.sum(color_hist) + 1e-7)  # Add small epsilon to avoid division by zero
            
            # 2. Compute texture features for this cell
            gray_cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            texture_features = []
            
            for gabor_kernel in gabor_filters:
                filtered = cv2.filter2D(gray_cell, cv2.CV_8UC3, gabor_kernel)
                # Compute mean and standard deviation of filter response
                texture_features.extend([
                    np.mean(filtered),
                    np.std(filtered)
                ])
            
            # Normalize texture features
            texture_features = np.array(texture_features)
            texture_features = texture_features / (np.sum(texture_features) + 1e-7)
            
            # 3. Combine color and texture features for this cell
            combined_cell_features = np.concatenate([color_hist, texture_features])
            
            # 4. Add this cell's features to our list
            cell_features_list.append(combined_cell_features)
    
    # 5. Concatenate features from all cells into final feature vector
    final_features = np.concatenate(cell_features_list)
    
    # 6. Final normalization of complete feature vector
    final_features = final_features / (np.sum(final_features) + 1e-7)
    
    return final_features

def compute_texture_features(gray_cell, gabor_filters):
    """Helper function to compute texture features for a cell."""
    texture_features = []
    for gabor_kernel in gabor_filters:
        filtered = cv2.filter2D(gray_cell, cv2.CV_8UC3, gabor_kernel)
        texture_features.extend([
            np.mean(filtered),
            np.std(filtered)
        ])
    return np.array(texture_features)