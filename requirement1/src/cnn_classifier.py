"""
CNN-based image classifier using transfer learning with pre-trained models.

Author: Zohra Bouchamaoui
Student ID: 6848526
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
import random
from config.settings import TEST_QUERIES
import traceback

def get_image_class(image_path):
    """Extract class from image filename."""
    return os.path.basename(image_path).split('_')[0]

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        # Extract class labels from filenames
        self.labels = [int(get_image_class(path)) for path in image_paths]
        self.classes = sorted(list(set(self.labels)))
        # Map labels to consecutive integers
        self.label_map = {label: idx for idx, label in enumerate(self.classes)}
        self.labels = [self.label_map[label] for label in self.labels]
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.labels[idx]

class CNNClassifier:
    def __init__(self, model_name='resnet18', results_dir='results'):
        """Initialize CNN classifier."""
        self.model_name = model_name
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        # Modify final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 20)  # 20 classes
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize optimizer and criterion
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                 step_size=7, 
                                                 gamma=0.1,
                                                 verbose=True)

    def train(self, train_paths, val_paths, batch_size=32, num_epochs=20):
        """Train the CNN model."""
        # Create datasets and dataloaders
        train_dataset = ImageDataset(train_paths, transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = ImageDataset(val_paths, transform=self.transform)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        train_losses = []
        val_accuracies = []
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0
            
            # Training loop
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{num_epochs}')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss/len(train_loader)})
            
            # Validation
            val_acc = self.evaluate(val_loader)
            print(f"Epoch {epoch}: Val Accuracy = {val_acc:.4f}")
            
            train_losses.append(total_loss/len(train_loader))
            val_accuracies.append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 
                         os.path.join(self.results_dir, 'best_model.pth'))
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
            
            self.scheduler.step()
        
        # Load best model
        self.model.load_state_dict(torch.load(os.path.join(self.results_dir, 
                                                          'best_model.pth'),
                                             weights_only=True))
        
        print("\nGenerating visualizations...")
        try:
            # Create validation loader for visualizations
            val_dataset = ImageDataset(val_paths, transform=self.transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Generate and save visualizations
            self._save_training_curves(train_losses, val_accuracies)
            self._save_confusion_matrix(val_loader)
            self._evaluate_test_queries()
            
        except Exception as e:
            print(f"Error in visualization generation: {str(e)}")
            print("Traceback:", traceback.format_exc())

    def evaluate(self, val_loader):
        """Evaluate the model on validation data."""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Generate classification report
        report = classification_report(all_labels, all_preds)
        with open(os.path.join(self.results_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        
        return correct / total
    
    def _save_training_curves(self, train_losses, val_accuracies):
        """Plot and save training curves."""
        plt.figure(figsize=(15, 5))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'b-', label='Training Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, 'r-', label='Validation Accuracy')
        plt.title('Validation Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_curves.png'))
        plt.close()
    
    def _save_confusion_matrix(self, val_loader):
        """Generate and save confusion matrix visualization."""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close()
        
        return cm
    
    def _evaluate_test_queries(self):
        """Evaluate model on standard test queries."""
        self.model.eval()
        results = {}
        
        # Create figure for test queries
        num_queries = len(TEST_QUERIES)
        fig = plt.figure(figsize=(15, 5*num_queries))
        gs = plt.GridSpec(num_queries, 2)
        
        for idx, (query_name, query_path) in enumerate(TEST_QUERIES.items()):
            # Load and process image
            img = cv2.imread(query_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get prediction
            with torch.no_grad():
                x = self.transform(img).unsqueeze(0).to(self.device)
                outputs = self.model(x)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            
            # Get true label
            true_label = get_image_class(query_path)
            
            # Plot image
            ax_img = fig.add_subplot(gs[idx, 0])
            ax_img.imshow(img)
            correct = '✓' if str(pred.item()+1) == true_label else '✗'
            ax_img.set_title(f'Query: {query_name} ({correct})\n'
                            f'True: {true_label}, Pred: {pred.item()+1}\n'
                            f'Confidence: {conf.item():.3f}')
            ax_img.axis('off')
            
            # Plot probabilities
            ax_prob = fig.add_subplot(gs[idx, 1])
            probs = probs.cpu().numpy()[0]
            ax_prob.bar(range(1, len(probs)+1), probs)
            ax_prob.set_title(f'Class Probabilities')
            ax_prob.set_xlabel('Class')
            ax_prob.set_ylabel('Probability')
            
            # Store results
            results[query_name] = {
                'true_label': true_label,
                'predicted_label': str(pred.item()+1),
                'confidence': conf.item(),
                'probabilities': probs
            }
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'test_queries.png'))
        plt.close()
        
        # Save analysis report
        self._save_test_queries_analysis(results)
        
        return results

    def _save_test_queries_analysis(self, results):
        """Save detailed analysis of test queries."""
        with open(os.path.join(self.results_dir, 'test_queries_analysis.txt'), 'w') as f:
            f.write("Test Queries Analysis\n")
            f.write("====================\n\n")
            
            correct_count = 0
            total_confidence = 0
            
            for query_name, result in results.items():
                f.write(f"\nQuery: {query_name}\n")
                f.write(f"True Label: {result['true_label']}\n")
                f.write(f"Predicted: {result['predicted_label']}\n")
                f.write(f"Confidence: {result['confidence']:.4f}\n")
                
                # Top-3 predictions
                top3_idx = np.argsort(result['probabilities'])[-3:][::-1]
                f.write("\nTop-3 Predictions:\n")
                for idx in top3_idx:
                    f.write(f"Class {idx+1}: {result['probabilities'][idx]:.4f}\n")
                
                if result['true_label'] == result['predicted_label']:
                    correct_count += 1
                    total_confidence += result['confidence']
            
            # Overall statistics
            f.write("\nOverall Statistics\n")
            f.write("=================\n")
            f.write(f"Accuracy on test queries: {correct_count/len(results):.3f}\n")
            if correct_count > 0:
                f.write(f"Average confidence for correct predictions: {total_confidence/correct_count:.3f}\n")

    def _plot_results(self):
        """Plot all results after training."""
        try:
            # Training curves
            plt.figure(figsize=(15, 5))
            
            # Loss curve
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses, 'b-', label='Training Loss')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            # Accuracy curve
            plt.subplot(1, 2, 2)
            plt.plot(self.val_accuracies, 'r-', label='Validation Accuracy')
            plt.title('Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'training_curves.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error plotting results: {str(e)}")