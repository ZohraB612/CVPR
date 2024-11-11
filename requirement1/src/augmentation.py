"""
Image augmentation utilities for improving classification performance.

Author: Zohra Bouchamaoui
Student ID: 6848526
"""

import cv2
import numpy as np
from tqdm import tqdm

def augment_image(img):
    """
    Create augmented versions of an input image.
    
    Args:
        img: Input image in BGR format
        
    Returns:
        list: Augmented versions of the image
    """
    augmented = []
    
    # 1. Rotation augmentations
    for angle in [90, 180, 270]:
        matrix = cv2.getRotationMatrix2D(
            (img.shape[1] / 2, img.shape[0] / 2), 
            angle, 
            1.0
        )
        rotated = cv2.warpAffine(
            img, 
            matrix, 
            (img.shape[1], img.shape[0])
        )
        augmented.append(rotated)
    
    # 2. Flip augmentations
    augmented.append(cv2.flip(img, 1))  # horizontal flip
    augmented.append(cv2.flip(img, 0))  # vertical flip
    
    # 3. Brightness adjustments
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
    augmented.extend([bright, dark])
    
    # 4. Add noise
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    augmented.append(noisy)
    
    # 5. Blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    augmented.append(blurred)
    
    return augmented

def augment_underrepresented_classes(image_paths, class_threshold=7):
    """
    Augment images from classes with fewer samples than threshold.
    
    Args:
        image_paths: List of image paths
        class_threshold: Minimum number of samples per class
        
    Returns:
        list: Updated image paths including augmented images
    """
    # Count samples per class
    class_counts = {}
    for path in image_paths:
        class_name = path.split('_')[0].split('/')[-1]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Identify classes needing augmentation
    augmented_paths = image_paths.copy()
    
    print("Augmenting underrepresented classes...")
    for class_name, count in tqdm(class_counts.items()):
        if count < class_threshold:
            # Find all images of this class
            class_images = [p for p in image_paths if class_name in p.split('_')[0].split('/')[-1]]
            
            # Calculate how many augmentations needed
            num_augmentations_needed = class_threshold - count
            
            # Augment random images from this class
            for i in range(num_augmentations_needed):
                # Select random image from class
                img_path = np.random.choice(class_images)
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # Get augmented versions
                augmented_images = augment_image(img)
                
                # Select one random augmentation
                selected_aug = np.random.choice(augmented_images)
                
                # Save augmented image
                aug_path = img_path.replace(
                    '.bmp', 
                    f'_aug_{i}.bmp'
                )
                cv2.imwrite(aug_path, selected_aug)
                augmented_paths.append(aug_path)
    
    return augmented_paths 