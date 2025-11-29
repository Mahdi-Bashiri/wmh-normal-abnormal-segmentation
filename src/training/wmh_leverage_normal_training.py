"""
Enhanced WMH Segmentation with U-Net - Journal Paper Implementation
Three-class segmentation: Background vs Normal WMH vs Abnormal WMH
Professional results saving and visualization for publication

This relates to our article:
"Incorporating Normal Periventricular Changes for Enhanced Pathological
White Matter Hyperintensity Segmentation: On Multi-Class Deep Learning Approaches"

Authors:
"Mahdi Bashiri Bawil, Mousa Shamsi, Ali Fahmi Jafargholkhanloo, Abolhassan Shakeri Bavil"

Developer:
"Mahdi Bashiri Bawil"
"""

###################### Libraries ######################

# General Utilities
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import cv2 as cv
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
import pickle
from pathlib import Path
from skimage.morphology import remove_small_objects, binary_opening, disk
from skimage.measure import label

# Deep Learning
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras import backend as K
from tensorflow.keras import layers, optimizers, callbacks
from keras.utils import to_categorical

# Analysis and Statistics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Models
from unet_model import build_unet_3class
from attn_unet_model import build_attention_unet_3class
from trans_unet_model import build_trans_unet_3class
from dlv3_unet_model import build_deeplabv3_unet_3class

# Loss Functions
from loss_functions import *

# Metrics Functions
from metrics_functions import *

# Check for GPU assistance
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU Available: ", tf.test.is_gpu_available())
print("Built with CUDA: ", tf.test.is_built_with_cuda())
print("Physical devices: ", tf.config.list_physical_devices())

# Force GPU if available
if tf.config.list_physical_devices('GPU'):
    print("\n\n\t\t\tUsing GPU\n\n")
else:
    print("\n\n\t\t\tUsing CPU\n\n")

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Set publication-ready matplotlib settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

###################### Configuration and Setup ######################

class Config:
    """Configuration class for the experiment"""
    def __init__(self):
        # Model Name
        self.model_name = 'attn_unet'    # 'unet', 'attn_unet', 'trans_unet', 'deepl3_unet'
        # Paths
        self.train_dir = "Leverage_Article_Data/train_3L_wmh_local_public/"
        self.test_dir = "Leverage_Article_Data/test_3L_wmh_local_public/"
        self.intended_study_dir = "leverage_results_20251124_160949_attn_unet"  # for inference intentions
        
        # Model parameters
        self.input_shape = (256, 256, 1)
        self.target_size = (256, 256)
        self.num_classes_3 = 3
        self.num_classes_binary = 1
        
        # Training parameters
        self.mode = 'training'
        self.epochs = 50  # Increased for better convergence
        
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.validation_split = 0.1
        self.random_state = 42
        
        # Loss function options
        self.loss_options = {
            'scenario1': 'weighted_bce',  # weighted_bce, focal, combined, dice
            'scenario2': 'weighted_categorical'  # weighted_categorical, multiclass_dice, categorical
        }

        # Choose a model to train or inference
        if self.model_name == 'unet':
            self.build_unet_variant = build_unet_3class
        elif self.model_name == 'attn_unet':
            self.build_unet_variant = build_attention_unet_3class
        elif self.model_name == 'trans_unet':
            self.build_unet_variant = build_trans_unet_3class
        elif self.model_name == 'deepl3_unet':
            self.build_unet_variant = build_deeplabv3_unet_3class

        # Create results directory with timestamp
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.mode != 'training':
            self.results_dir = Path(f"leverage_results_{self.timestamp}_{self.model_name}_no_training")
        else:
            self.results_dir = Path(f"leverage_results_{self.timestamp}_{self.model_name}")
        self.create_directory_structure()
        
    def create_directory_structure(self):
        """Create professional directory structure for results"""
        subdirs = [
            'models',
            'figures',
            'tables',
            'statistics',
            'predictions',
            'logs',
            'config'
        ]
        
        self.results_dir.mkdir(exist_ok=True)
        for subdir in subdirs:
            (self.results_dir / subdir).mkdir(exist_ok=True)
            
        # Save experiment configuration
        config_dict = {
            'timestamp': self.timestamp,
            'input_shape': self.input_shape,
            'target_size': self.target_size,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'validation_split': self.validation_split,
            'random_state': self.random_state,
            'loss_options': self.loss_options
        }
        
        with open(self.results_dir / 'config' / 'experiment_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)

config = Config()

###################### Data Loading Functions ######################

def extract_number(filename):
    """Extract patient ID and slice number for proper sorting"""
    return int(''.join(filter(str.isdigit, filename.split('_')[0])))

def load_wmh_dataset(data_dir, target_size=(256, 256), save_info=True):
    """
    Load dataset with specific format: 256x512 images (FLAIR + GT mask concatenated)
    """
    images, masks_3class, masks_binary = [], [], []
    # image_files = sorted(os.listdir(data_dir), key=extract_number)
    image_files = [f for f in os.listdir(data_dir)]
 
    dataset_info = {
        'total_files': len(image_files),
        'loaded_files': 0,
        'skipped_files': [],
        'image_shapes': [],
        'class_distributions': {'background': [], 'normal_wmh': [], 'abnormal_wmh': []}
    }
    
    for img_name in tqdm(image_files, desc=f"Loading from {os.path.basename(data_dir)}"):
        # Load concatenated image
        full_img = cv.imread(os.path.join(data_dir, img_name), cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE).astype(np.float32)

        if full_img is None or full_img.shape[1] != 512:
            dataset_info['skipped_files'].append(img_name)
            continue

        # Split into FLAIR and GT
        flair_img = full_img[:, :256]
        gt_mask = full_img[:, 256:]
        
        # Resize if needed
        if target_size != (256, 256):
            flair_img = cv.resize(flair_img, target_size)
            gt_mask = cv.resize(gt_mask, target_size)
        
        dataset_info['image_shapes'].append(flair_img.shape)
        
        # Normalize FLAIR image
        flair_img = flair_img.astype(np.float32)
        flair_img = (flair_img - np.mean(flair_img)) / (np.std(flair_img) + 1e-7)
        flair_img = np.expand_dims(flair_img, axis=-1)
        
        # Process ground truth masks
        gt_mask = gt_mask.astype(np.float32)
        
        # Create 3-class mask
        mask_3class = np.zeros_like(gt_mask, dtype=np.uint8)
        threshold_1 = 32767 // 2
        threshold_2 = 32767 + 1000
        threshold_3 = 65535 - 32767 // 2
        mask_3class[gt_mask < threshold_1] = 0
        mask_3class[(gt_mask >= threshold_1) & (gt_mask < threshold_2)] = 1
        mask_3class[gt_mask >= threshold_3] = 2
        
        # Create binary mask
        mask_binary = np.zeros_like(gt_mask, dtype=np.uint8)
        mask_binary[gt_mask >= threshold_3] = 1
        
        # Record class distributions
        unique, counts = np.unique(mask_3class, return_counts=True)
        class_dist = dict(zip(unique, counts))
        dataset_info['class_distributions']['background'].append(class_dist.get(0, 0))
        dataset_info['class_distributions']['normal_wmh'].append(class_dist.get(1, 0))
        dataset_info['class_distributions']['abnormal_wmh'].append(class_dist.get(2, 0))
        
        images.append(flair_img)
        masks_3class.append(mask_3class)
        masks_binary.append(mask_binary)
        dataset_info['loaded_files'] += 1
    
    # Save dataset information
    if save_info:
        dataset_info['class_distributions'] = {k: np.array(v) for k, v in dataset_info['class_distributions'].items()}
        with open(config.results_dir / 'logs' / f'dataset_info_{os.path.basename(data_dir)}.pkl', 'wb') as f:
            pickle.dump(dataset_info, f)
    
    return np.array(images), np.array(masks_3class), np.array(masks_binary), dataset_info

###################### U-Net Architecture ######################
# callable from related functions saved in the main directory

###################### Loss Functions ######################
# callable from the related function saved in the main directory

###################### Metrics and Evaluation ######################
# callable from related functions saved in the main directory

###################### Post Processing ######################

def post_process_predictions(predictions, min_object_size=5, apply_opening=True, kernel_size=3):
    """
    Post-process binary predictions to remove small objects and apply morphological operations
    
    Args:
        predictions: Binary prediction masks (numpy array)
        min_object_size: Minimum object size in pixels (objects smaller than this are removed)
        apply_opening: Whether to apply morphological opening operation
        kernel_size: Size of morphological kernel for opening operation
    
    Returns:
        post_processed_predictions: Cleaned binary masks
    """
    from skimage.morphology import remove_small_objects, binary_opening, disk
    from skimage.measure import label
    
    post_processed = np.zeros_like(predictions, dtype=np.uint8)
    
    for i in range(predictions.shape[0]):
        mask = predictions[i].astype(bool)
        
        # Remove small objects
        if min_object_size > 0:
            mask = remove_small_objects(mask, min_size=min_object_size)
        
        # Apply morphological opening
        if apply_opening:
            kernel = disk(kernel_size)
            mask = binary_opening(mask, kernel)
            # Remove small objects
            if min_object_size > 0:
                mask = remove_small_objects(mask, min_size=min_object_size)
        
        
        post_processed[i] = mask.astype(np.uint8)
    
    return post_processed

###################### Professional Visualization Functions ######################

class PublicationPlotter:
    """Professional plotting class for publication-quality figures"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.figures_dir = self.results_dir / 'figures'
 
    def plot_training_curves(self, history_s1, history_s2, save_name='training_curves'):
        """Plot publication-quality training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Handle both History objects (from training) and dicts (from loading)
        if hasattr(history_s1, 'history'):
            hist_s1 = history_s1.history  # From training
        else:
            hist_s1 = history_s1  # From loading (already a dict)
        
        if hasattr(history_s2, 'history'):
            hist_s2 = history_s2.history  # From training
        else:
            hist_s2 = history_s2  # From loading (already a dict)
        
        # Scenario 1
        axes[0, 0].plot(hist_s1['loss'], 'b-', linewidth=2, label='Training')
        axes[0, 0].plot(hist_s1['val_loss'], 'r-', linewidth=2, label='Validation')
        axes[0, 0].set_title('(a) Binary Classification Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(hist_s1['accuracy'], 'b-', linewidth=2, label='Training')
        axes[0, 1].plot(hist_s1['val_accuracy'], 'r-', linewidth=2, label='Validation')
        axes[0, 1].set_title('(b) Binary Classification Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scenario 2
        axes[1, 0].plot(hist_s2['loss'], 'g-', linewidth=2, label='Training')
        axes[1, 0].plot(hist_s2['val_loss'], 'orange', linewidth=2, label='Validation')
        axes[1, 0].set_title('(c) Three-class Classification Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(hist_s2['accuracy'], 'g-', linewidth=2, label='Training')
        axes[1, 1].plot(hist_s2['val_accuracy'], 'orange', linewidth=2, label='Validation')
        axes[1, 1].set_title('(d) Three-class Classification Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'{save_name}.png')
        plt.savefig(self.figures_dir / f'{save_name}.pdf')  # For LaTeX
        # plt.show()
    
    def plot_comparison_visualization(self, images, gt_3class, gt_binary, pred_s1, pred_s2, 
                                indices=None, save_name='comparison_visualization'):
        """Create publication-quality comparison visualization in single column format"""
        # Use random selection:
        if indices is None:
            indices = np.random.choice(len(images), 3, replace=False)
        # or Use manual selection:
        indices = np.array([50, 51, 62, 74])  # our chosen indices
        # indices = np.array([44])  # our chosen indices
            
        # Create single column layout: 6 rows, 1 column per sample
        n_samples = len(indices)
        n_plots = 6  # Number of different visualizations
        
        # Adjust figure size for single column format
        fig, axes = plt.subplots(n_plots * n_samples, 1, figsize=(8, 3 * n_plots * n_samples))
        
        # If only one sample, ensure axes is iterable
        if n_samples == 1:
            axes = np.array(axes).reshape(-1)
        
        titles = ['FLAIR Image', 'GT (3-class)', 'GT (Abnormal)', 
                'Scenario 1 Performance', 'Scenario 2 Performance', 'Legend']
        
        for sample_idx, idx in enumerate(indices):
            base_row = sample_idx * n_plots
            
            # Add sample identifier if multiple samples
            if n_samples > 1:
                sample_title = f" - Sample {sample_idx + 1}"
            else:
                sample_title = ""
            
            # FLAIR Image
            axes[base_row + 0].imshow(images[idx].squeeze(), cmap='gray')
            axes[base_row + 0].set_title(titles[0] + sample_title, fontsize=12, pad=10)
            axes[base_row + 0].axis('off')
            
            # GT 3-class - Using grayscale
            axes[base_row + 1].imshow(gt_3class[idx], cmap='gray')
            axes[base_row + 1].set_title(titles[1] + sample_title, fontsize=12, pad=10)
            axes[base_row + 1].axis('off')
            
            # GT Binary (Abnormal only) - Black and white binary
            axes[base_row + 2].imshow(gt_binary[idx], cmap='gray', vmin=0, vmax=1)
            axes[base_row + 2].set_title(titles[2] + sample_title, fontsize=12, pad=10)
            axes[base_row + 2].axis('off')
            
            # Create RGB version of FLAIR image for overlays
            flair_rgb = np.stack([images[idx].squeeze()] * 3, axis=-1)
            # Normalize to 0-1 range if needed
            flair_rgb = (flair_rgb - flair_rgb.min()) / (flair_rgb.max() - flair_rgb.min())
            
            # Scenario 1 Performance Analysis
            # Convert prediction to binary (assuming abnormal class)
            pred_s1_binary = (pred_s1[idx] > 0).astype(np.uint8)
            
            # Calculate TP, FP, FN
            tp_s1 = (gt_binary[idx] == 1) & (pred_s1_binary == 1)
            fp_s1 = (gt_binary[idx] == 0) & (pred_s1_binary == 1)
            fn_s1 = (gt_binary[idx] == 1) & (pred_s1_binary == 0)
            
            # Create overlay image for Scenario 1
            overlay_s1 = flair_rgb.copy()
            overlay_s1[tp_s1, :] = [0, 1, 0]  # Green for TP
            overlay_s1[fp_s1, :] = [1, 0, 0]  # Red for FP
            overlay_s1[fn_s1, :] = [1, 1, 0]  # Yellow for FN
            
            axes[base_row + 3].imshow(overlay_s1)
            axes[base_row + 3].set_title(titles[3] + sample_title, fontsize=12, pad=10)
            axes[base_row + 3].axis('off')
            
            # Scenario 2 Performance Analysis
            # Convert prediction to binary (abnormal only)
            if np.max(pred_s2) == 2:
                pred_s2_binary = (pred_s2[idx] == 2).astype(np.uint8)
            else:
                pred_s2_binary = (pred_s2[idx] > 0).astype(np.uint8)

            
            # Calculate TP, FP, FN
            tp_s2 = (gt_binary[idx] == 1) & (pred_s2_binary == 1)
            fp_s2 = (gt_binary[idx] == 0) & (pred_s2_binary == 1)
            fn_s2 = (gt_binary[idx] == 1) & (pred_s2_binary == 0)
            
            # Create overlay image for Scenario 2
            overlay_s2 = flair_rgb.copy()
            overlay_s2[tp_s2, :] = [0, 1, 0]  # Green for TP
            overlay_s2[fp_s2, :] = [1, 0, 0]  # Red for FP
            overlay_s2[fn_s2, :] = [0, 0, 1]  # Blue for FN
            
            axes[base_row + 4].imshow(overlay_s2)
            axes[base_row + 4].set_title(titles[4] + sample_title, fontsize=12, pad=10)
            axes[base_row + 4].axis('off')
            
            # Legend plot (replacing the overlay comparison)
            axes[base_row + 5].axis('off')
            from matplotlib.patches import Rectangle
            from matplotlib.lines import Line2D
            
            # Create legend elements
            legend_elements = [
                Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
                       markersize=15, label='True Positive (TP)'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
                       markersize=15, label='False Positive (FP)'),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', 
                       markersize=15, label='False Negative (FN)')
            ]
            
            legend = axes[base_row + 5].legend(handles=legend_elements, 
                                             loc='center', fontsize=12, 
                                             title='Performance Metrics', 
                                             title_fontsize=14)
            legend.get_title().set_fontweight('bold')
        
        # Adjust layout with more spacing for better readability
        plt.tight_layout(pad=2.0)
        
        # Save with high DPI for publication quality
        plt.savefig(self.figures_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / f'{save_name}.pdf', bbox_inches='tight')
        # plt.show()

    def plot_metrics_comparison(self, metrics_s1, metrics_s2, save_name='metrics_comparison'):
        """Create professional metrics comparison plot including IoU"""
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'Dice', 'IoU']
        s1_values = [metrics_s1[metric] for metric in metrics_to_plot]
        s2_values = [metrics_s2[metric] for metric in metrics_to_plot]
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, s1_values, width, label='Binary Classification', 
                    color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, s2_values, width, label='Three-class Classification', 
                    color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Performance Comparison: Binary vs Three-class Classification')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'{save_name}.png')
        plt.savefig(self.figures_dir / f'{save_name}.pdf')
        
    def plot_dice_distribution(self, dice_s1, dice_s2, save_name='dice_distribution'):
        """Plot Dice coefficient distributions"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot comparison
        ax1.boxplot([dice_s1, dice_s2], labels=['Binary\nClassification', 'Three-class\nClassification'])
        ax1.set_ylabel('Dice Coefficient')
        ax1.set_title('(a) Dice Coefficient Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Histogram overlay
        ax2.hist(dice_s1, alpha=0.6, bins=20, label='Binary Classification', color='skyblue')
        ax2.hist(dice_s2, alpha=0.6, bins=20, label='Three-class Classification', color='lightcoral')
        ax2.set_xlabel('Dice Coefficient')
        ax2.set_ylabel('Frequency')
        ax2.set_title('(b) Dice Coefficient Histogram')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'{save_name}.png')
        plt.savefig(self.figures_dir / f'{save_name}.pdf')
        # plt.show()

    def plot_dice_iou_distribution(self, dice_s1, dice_s2, iou_s1, iou_s2, save_name='dice_iou_distribution'):
        """Plot both Dice and IoU coefficient distributions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Dice box plots
        axes[0,0].boxplot([dice_s1, dice_s2], labels=['Binary\nClassification', 'Three-class\nClassification'])
        axes[0,0].set_ylabel('Dice Coefficient')
        axes[0,0].set_title('(a) Dice Coefficient Distribution')
        axes[0,0].grid(True, alpha=0.3)
        
        # Dice histograms
        axes[0,1].hist(dice_s1, alpha=0.6, bins=20, label='Binary Classification', color='skyblue')
        axes[0,1].hist(dice_s2, alpha=0.6, bins=20, label='Three-class Classification', color='lightcoral')
        axes[0,1].set_xlabel('Dice Coefficient')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('(b) Dice Coefficient Histogram')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # IoU box plots
        axes[1,0].boxplot([iou_s1, iou_s2], labels=['Binary\nClassification', 'Three-class\nClassification'])
        axes[1,0].set_ylabel('IoU Coefficient')
        axes[1,0].set_title('(c) IoU Coefficient Distribution')
        axes[1,0].grid(True, alpha=0.3)
        
        # IoU histograms
        axes[1,1].hist(iou_s1, alpha=0.6, bins=20, label='Binary Classification', color='skyblue')
        axes[1,1].hist(iou_s2, alpha=0.6, bins=20, label='Three-class Classification', color='lightcoral')
        axes[1,1].set_xlabel('IoU Coefficient')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('(d) IoU Coefficient Histogram')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / f'{save_name}.png')
        plt.savefig(self.figures_dir / f'{save_name}.pdf')
        
###################### Results Saving Functions ######################

class ResultsSaver:
    """Professional results saving and documentation"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        
    def save_models(self, model_s1, model_s2, history_s1, history_s2):
        """Save trained models and training histories"""
        # Save models
        model_s1.save(self.results_dir / 'models' / 'scenario1_binary_model.h5')
        model_s2.save(self.results_dir / 'models' / 'scenario2_multiclass_model.h5')
        
        # Save training histories
        with open(self.results_dir / 'models' / 'training_history_s1.pkl', 'wb') as f:
            pickle.dump(history_s1.history, f)
        with open(self.results_dir / 'models' / 'training_history_s2.pkl', 'wb') as f:
            pickle.dump(history_s2.history, f)
    
    def load_models(self, study_dir, loss_s1_func, loss_s2_func):
        """Load saved models and training histories with proper custom loss functions
        
        Args:
            study_dir: Directory containing the saved models
        """
        try:
            # Convert study_dir to Path object if it's a string
            study_dir = Path(study_dir)
            
            # Load models with their respective custom objects
            model_s1 = keras.models.load_model(
                study_dir / 'models' / 'scenario1_binary_model.h5',
                compile=False
                # custom_objects=loss_s1_func
            )
            model_s2 = keras.models.load_model(
                study_dir / 'models' / 'scenario2_multiclass_model.h5',
                compile=False
                # custom_objects=loss_s2_func
            )
            
            # Load training histories
            with open(study_dir / 'models' / 'training_history_s1.pkl', 'rb') as f:
                history_s1 = pickle.load(f)
            with open(study_dir / 'models' / 'training_history_s2.pkl', 'rb') as f:
                history_s2 = pickle.load(f)
            
            print("Models and histories loaded successfully!")
            return model_s1, model_s2, history_s1, history_s2
            
        except FileNotFoundError as e:
            print(f"Error: Could not find saved models. {e}")
            print("Make sure you have saved models using save_models() first.")
            print(f"Looking in directory: {Path(study_dir) / 'models'}")
            return None, None, None, None
        except Exception as e:
            print(f"Error loading models: {e}")
            return None, None, None, None
        
    def save_predictions(self, test_images, test_masks_3class, test_masks_binary, 
                        pred_s1, pred_s2, dataset_info):
        """Save predictions and test data"""
        predictions_dir = self.results_dir / 'predictions'
        
        # Save raw predictions
        np.save(predictions_dir / 'test_images.npy', test_images)
        np.save(predictions_dir / 'test_masks_3class.npy', test_masks_3class)
        np.save(predictions_dir / 'test_masks_binary.npy', test_masks_binary)
        np.save(predictions_dir / 'predictions_scenario1.npy', pred_s1)
        np.save(predictions_dir / 'predictions_scenario2.npy', pred_s2)
        
        # Save dataset information
        with open(predictions_dir / 'dataset_info.pkl', 'wb') as f:
            pickle.dump(dataset_info, f)
    
    def save_metrics_table(self, metrics_s1, metrics_s2, dice_stats):
        """Save comprehensive metrics table including HD95 and ASSD"""
        # Create comprehensive results table
        results_table = pd.DataFrame([metrics_s1, metrics_s2])
        
        # Add statistical information - updated for all metrics including surface-based
        stats_row = {
            'Scenario': 'Statistical Analysis',
            'Accuracy': f"Dice p={dice_stats['dice_p_value']:.4f}",
            'Precision': f"Dice t={dice_stats['dice_t_statistic']:.4f}",
            'Recall': f"Dice Δ={dice_stats['dice_improvement']:.4f}",
            'Specificity': f"Dice ES={dice_stats['dice_effect_size']:.4f}",
            'Dice': f"IoU p={dice_stats['iou_p_value']:.4f}",
            'IoU': f"IoU Δ={dice_stats['iou_improvement']:.4f}"
        }
        
        results_table = pd.concat([results_table, pd.DataFrame([stats_row])], ignore_index=True)
        
        # Save as CSV and Excel
        results_table.to_csv(self.results_dir / 'tables' / 'comprehensive_results.csv', index=False)
        results_table.to_excel(self.results_dir / 'tables' / 'comprehensive_results.xlsx', index=False)
        
        # Create separate surface metrics table
        surface_metrics_data = {
            'Scenario': ['Binary (S1)', 'Three-class (S2)', 'Statistical Analysis'],
            'HD95_Mean': [
                dice_stats['hd95_scenario1_mean'],
                dice_stats['hd95_scenario2_mean'],
                None
            ],
            'HD95_Std': [
                dice_stats['hd95_scenario1_std'],
                dice_stats['hd95_scenario2_std'],
                None
            ],
            'HD95_Median': [
                dice_stats['hd95_scenario1_median'],
                dice_stats['hd95_scenario2_median'],
                None
            ],
            'ASSD_Mean': [
                dice_stats['assd_scenario1_mean'],
                dice_stats['assd_scenario2_mean'],
                None
            ],
            'ASSD_Std': [
                dice_stats['assd_scenario1_std'],
                dice_stats['assd_scenario2_std'],
                None
            ],
            'ASSD_Median': [
                dice_stats['assd_scenario1_median'],
                dice_stats['assd_scenario2_median'],
                None
            ],
            'HD95_Stats': [
                None,
                None,
                f"Δ={dice_stats['hd95_improvement']:.4f}px, p={dice_stats['hd95_p_value']:.4f}"
            ],
            'ASSD_Stats': [
                None,
                None,
                f"Δ={dice_stats['assd_improvement']:.4f}px, p={dice_stats['assd_p_value']:.4f}"
            ]
        }
        
        surface_table = pd.DataFrame(surface_metrics_data)
        surface_table.to_csv(self.results_dir / 'tables' / 'surface_metrics.csv', index=False)
        surface_table.to_excel(self.results_dir / 'tables' / 'surface_metrics.xlsx', index=False)
        
        # Create comprehensive LaTeX table with all metrics
        latex_table = results_table.iloc[:-1].to_latex(
            index=False, 
            float_format="%.4f",
            caption="Performance comparison between binary and three-class segmentation approaches",
            label="tab:performance_comparison"
        )
        
        with open(self.results_dir / 'tables' / 'latex_table.tex', 'w') as f:
            f.write(latex_table)
        
        # Create LaTeX table for surface metrics
        latex_surface_table = surface_table.iloc[:-1].to_latex(
            index=False,
            float_format="%.4f",
            caption="Surface-based metrics (HD95 and ASSD) comparison in pixels",
            label="tab:surface_metrics"
        )
        
        with open(self.results_dir / 'tables' / 'latex_surface_table.tex', 'w') as f:
            f.write(latex_surface_table)
            
        return results_table, surface_table

    def save_statistical_analysis(self, dice_s1, dice_s2, iou_s1, iou_s2, 
                                hd95_s1, hd95_s2, assd_s1, assd_s2,
                                metrics_s1, metrics_s2):
        """Comprehensive statistical analysis for multiple metrics including surface-based metrics"""
        from scipy.stats import ttest_rel, wilcoxon, normaltest, levene
        import scipy.stats as stats
        
        def analyze_metric(metric1, metric2, metric_name, lower_is_better=False):
            """Analyze a single metric pair"""
            # Normality tests
            _, p_normal_1 = normaltest(metric1)
            _, p_normal_2 = normaltest(metric2)
            
            # Paired t-test
            # For lower_is_better metrics (HD95, ASSD), we reverse the comparison
            if lower_is_better:
                t_stat, p_ttest = ttest_rel(metric1, metric2)  # Test if metric1 > metric2
            else:
                t_stat, p_ttest = ttest_rel(metric2, metric1)  # Test if metric2 > metric1
            
            # Wilcoxon signed-rank test (non-parametric alternative)
            if lower_is_better:
                w_stat, p_wilcoxon = wilcoxon(metric1, metric2, alternative='two-sided')
            else:
                w_stat, p_wilcoxon = wilcoxon(metric2, metric1, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(metric1, ddof=1) + np.var(metric2, ddof=1)) / 2)
            if lower_is_better:
                cohens_d = (np.mean(metric1) - np.mean(metric2)) / pooled_std if pooled_std > 0 else 0
            else:
                cohens_d = (np.mean(metric2) - np.mean(metric1)) / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval for difference
            diff = metric2 - metric1 if not lower_is_better else metric1 - metric2
            ci_lower, ci_upper = stats.t.interval(0.95, len(diff)-1, 
                                                loc=np.mean(diff), 
                                                scale=stats.sem(diff))
            
            # Calculate improvement
            if lower_is_better:
                improvement = np.mean(metric1) - np.mean(metric2)  # Reduction is good
                improvement_percent = (improvement / np.mean(metric1)) * 100 if np.mean(metric1) > 0 else 0
            else:
                improvement = np.mean(metric2) - np.mean(metric1)  # Increase is good
                improvement_percent = (improvement / np.mean(metric1)) * 100 if np.mean(metric1) > 0 else 0
            
            return {
                f'{metric_name}_scenario1_mean': np.mean(metric1),
                f'{metric_name}_scenario1_std': np.std(metric1),
                f'{metric_name}_scenario1_median': np.median(metric1),
                f'{metric_name}_scenario2_mean': np.mean(metric2),
                f'{metric_name}_scenario2_std': np.std(metric2),
                f'{metric_name}_scenario2_median': np.median(metric2),
                f'{metric_name}_improvement': improvement,
                f'{metric_name}_improvement_percent': improvement_percent,
                f'{metric_name}_t_statistic': t_stat,
                f'{metric_name}_p_value': p_ttest,
                f'{metric_name}_wilcoxon_statistic': w_stat,
                f'{metric_name}_wilcoxon_p_value': p_wilcoxon,
                f'{metric_name}_effect_size': cohens_d,
                f'{metric_name}_ci_lower': ci_lower,
                f'{metric_name}_ci_upper': ci_upper,
                f'{metric_name}_normality_s1_p': p_normal_1,
                f'{metric_name}_normality_s2_p': p_normal_2,
                f'{metric_name}_significant': p_ttest < 0.05
            }
        
        # Analyze all metrics
        dice_results = analyze_metric(dice_s1, dice_s2, 'dice')
        iou_results = analyze_metric(iou_s1, iou_s2, 'iou')
        hd95_results = analyze_metric(hd95_s1, hd95_s2, 'hd95', lower_is_better=True)
        assd_results = analyze_metric(assd_s1, assd_s2, 'assd', lower_is_better=True)
        
        # Combine all results
        statistical_results = {
            'sample_size': len(dice_s1),
            'sample_size_hd95': len(hd95_s1),  # May be different due to filtering
            'sample_size_assd': len(assd_s1),
            **dice_results,
            **iou_results,
            **hd95_results,
            **assd_results
        }
        
        # Save statistical results
        with open(self.results_dir / 'statistics' / 'statistical_analysis.json', 'w') as f:
            json.dump(statistical_results, f, indent=2, default=str)
            
        # Create comprehensive statistical report
        report = f"""
    COMPREHENSIVE STATISTICAL ANALYSIS REPORT
    ==========================================

    Sample Sizes: 
    - Dice/IoU: {len(dice_s1)} test images
    - HD95: {len(hd95_s1)} test images (after filtering invalid cases)
    - ASSD: {len(assd_s1)} test images (after filtering invalid cases)

    DICE COEFFICIENT ANALYSIS:
    --------------------------
    Scenario 1 (Binary):     {dice_results['dice_scenario1_mean']:.4f} ± {dice_results['dice_scenario1_std']:.4f} (median: {dice_results['dice_scenario1_median']:.4f})
    Scenario 2 (Three-class): {dice_results['dice_scenario2_mean']:.4f} ± {dice_results['dice_scenario2_std']:.4f} (median: {dice_results['dice_scenario2_median']:.4f})
    Improvement:              {dice_results['dice_improvement']:.4f} ({dice_results['dice_improvement_percent']:.2f}%)

    Paired t-test:           t = {dice_results['dice_t_statistic']:.4f}, p = {dice_results['dice_p_value']:.4f}
    Wilcoxon signed-rank:    W = {dice_results['dice_wilcoxon_statistic']:.4f}, p = {dice_results['dice_wilcoxon_p_value']:.4f}
    Effect Size (Cohen's d): {dice_results['dice_effect_size']:.4f}
    95% CI for difference:   [{dice_results['dice_ci_lower']:.4f}, {dice_results['dice_ci_upper']:.4f}]
    Result: {'SIGNIFICANT' if dice_results['dice_significant'] else 'NOT SIGNIFICANT'}

    IoU COEFFICIENT ANALYSIS:
    -------------------------
    Scenario 1 (Binary):     {iou_results['iou_scenario1_mean']:.4f} ± {iou_results['iou_scenario1_std']:.4f} (median: {iou_results['iou_scenario1_median']:.4f})
    Scenario 2 (Three-class): {iou_results['iou_scenario2_mean']:.4f} ± {iou_results['iou_scenario2_std']:.4f} (median: {iou_results['iou_scenario2_median']:.4f})
    Improvement:              {iou_results['iou_improvement']:.4f} ({iou_results['iou_improvement_percent']:.2f}%)

    Paired t-test:           t = {iou_results['iou_t_statistic']:.4f}, p = {iou_results['iou_p_value']:.4f}
    Wilcoxon signed-rank:    W = {iou_results['iou_wilcoxon_statistic']:.4f}, p = {iou_results['iou_wilcoxon_p_value']:.4f}
    Effect Size (Cohen's d): {iou_results['iou_effect_size']:.4f}
    95% CI for difference:   [{iou_results['iou_ci_lower']:.4f}, {iou_results['iou_ci_upper']:.4f}]
    Result: {'SIGNIFICANT' if iou_results['iou_significant'] else 'NOT SIGNIFICANT'}

    HD95 (95th Percentile Hausdorff Distance) ANALYSIS:
    --------------------------------------------------
    Scenario 1 (Binary):     {hd95_results['hd95_scenario1_mean']:.4f} ± {hd95_results['hd95_scenario1_std']:.4f} pixels (median: {hd95_results['hd95_scenario1_median']:.4f})
    Scenario 2 (Three-class): {hd95_results['hd95_scenario2_mean']:.4f} ± {hd95_results['hd95_scenario2_std']:.4f} pixels (median: {hd95_results['hd95_scenario2_median']:.4f})
    Improvement:              {hd95_results['hd95_improvement']:.4f} pixels ({hd95_results['hd95_improvement_percent']:.2f}% reduction)

    Paired t-test:           t = {hd95_results['hd95_t_statistic']:.4f}, p = {hd95_results['hd95_p_value']:.4f}
    Wilcoxon signed-rank:    W = {hd95_results['hd95_wilcoxon_statistic']:.4f}, p = {hd95_results['hd95_wilcoxon_p_value']:.4f}
    Effect Size (Cohen's d): {hd95_results['hd95_effect_size']:.4f}
    95% CI for difference:   [{hd95_results['hd95_ci_lower']:.4f}, {hd95_results['hd95_ci_upper']:.4f}]
    Result: {'SIGNIFICANT' if hd95_results['hd95_significant'] else 'NOT SIGNIFICANT'}

    ASSD (Average Symmetric Surface Distance) ANALYSIS:
    --------------------------------------------------
    Scenario 1 (Binary):     {assd_results['assd_scenario1_mean']:.4f} ± {assd_results['assd_scenario1_std']:.4f} pixels (median: {assd_results['assd_scenario1_median']:.4f})
    Scenario 2 (Three-class): {assd_results['assd_scenario2_mean']:.4f} ± {assd_results['assd_scenario2_std']:.4f} pixels (median: {assd_results['assd_scenario2_median']:.4f})
    Improvement:              {assd_results['assd_improvement']:.4f} pixels ({assd_results['assd_improvement_percent']:.2f}% reduction)

    Paired t-test:           t = {assd_results['assd_t_statistic']:.4f}, p = {assd_results['assd_p_value']:.4f}
    Wilcoxon signed-rank:    W = {assd_results['assd_wilcoxon_statistic']:.4f}, p = {assd_results['assd_wilcoxon_p_value']:.4f}
    Effect Size (Cohen's d): {assd_results['assd_effect_size']:.4f}
    95% CI for difference:   [{assd_results['assd_ci_lower']:.4f}, {assd_results['assd_ci_upper']:.4f}]
    Result: {'SIGNIFICANT' if assd_results['assd_significant'] else 'NOT SIGNIFICANT'}

    NORMALITY TESTS:
    ----------------
    Dice - Scenario 1 p-value:     {dice_results['dice_normality_s1_p']:.4f} {'(Normal)' if dice_results['dice_normality_s1_p'] > 0.05 else '(Non-normal)'}
    Dice - Scenario 2 p-value:     {dice_results['dice_normality_s2_p']:.4f} {'(Normal)' if dice_results['dice_normality_s2_p'] > 0.05 else '(Non-normal)'}
    IoU - Scenario 1 p-value:      {iou_results['iou_normality_s1_p']:.4f} {'(Normal)' if iou_results['iou_normality_s1_p'] > 0.05 else '(Non-normal)'}
    IoU - Scenario 2 p-value:      {iou_results['iou_normality_s2_p']:.4f} {'(Normal)' if iou_results['iou_normality_s2_p'] > 0.05 else '(Non-normal)'}
    HD95 - Scenario 1 p-value:     {hd95_results['hd95_normality_s1_p']:.4f} {'(Normal)' if hd95_results['hd95_normality_s1_p'] > 0.05 else '(Non-normal)'}
    HD95 - Scenario 2 p-value:     {hd95_results['hd95_normality_s2_p']:.4f} {'(Normal)' if hd95_results['hd95_normality_s2_p'] > 0.05 else '(Non-normal)'}
    ASSD - Scenario 1 p-value:     {assd_results['assd_normality_s1_p']:.4f} {'(Normal)' if assd_results['assd_normality_s1_p'] > 0.05 else '(Non-normal)'}
    ASSD - Scenario 2 p-value:     {assd_results['assd_normality_s2_p']:.4f} {'(Normal)' if assd_results['assd_normality_s2_p'] > 0.05 else '(Non-normal)'}

    OVERALL CONCLUSIONS:
    -------------------
    Dice Improvement: {'STATISTICALLY SIGNIFICANT' if dice_results['dice_significant'] else 'NOT SIGNIFICANT'} 
    IoU Improvement:  {'STATISTICALLY SIGNIFICANT' if iou_results['iou_significant'] else 'NOT SIGNIFICANT'}
    HD95 Improvement: {'STATISTICALLY SIGNIFICANT' if hd95_results['hd95_significant'] else 'NOT SIGNIFICANT'}
    ASSD Improvement: {'STATISTICALLY SIGNIFICANT' if assd_results['assd_significant'] else 'NOT SIGNIFICANT'}

    Note: For HD95 and ASSD, lower values indicate better boundary accuracy.
    """
        
        with open(self.results_dir / 'statistics' / 'statistical_report.txt', 'w') as f:
            f.write(report)
            
        return statistical_results
    
    def generate_leverage_summary(self, config, dataset_info_train, dataset_info_test, 
                                metrics_s1, metrics_s2, statistical_results):
        """Generate comprehensive leverage paper summary with all metrics"""
        
        summary = f"""
    LEVERAGE PAPER RESULTS SUMMARY
    ================================
    Experiment Timestamp: {config.timestamp}
    Model Architecture: {config.model_name.upper()}
    WMH Segmentation: Binary vs Three-class Classification Comparison

    DATASET INFORMATION:
    --------------------
    Training Images: {dataset_info_train['loaded_files']} 
    Test Images: {dataset_info_test['loaded_files']}
    Image Size: {config.target_size}
    Classes: Background (0), Normal WMH (1), Abnormal WMH (2)

    METHODOLOGY:
    ------------
    Architecture: {config.model_name.upper()}
    Loss Functions: 
    - Scenario 1: {config.loss_options['scenario1']}
    - Scenario 2: {config.loss_options['scenario2']}
    Training Epochs: {config.epochs}
    Batch Size: {config.batch_size}
    Learning Rate: {config.learning_rate}

    PERFORMANCE RESULTS:
    --------------------
    OVERLAP-BASED METRICS:
                            | Scenario 1 (Binary) | Scenario 2 (3-class) | Improvement
    --------------------|---------------------|----------------------|------------
    Accuracy            | {metrics_s1['Accuracy']:.4f}            | {metrics_s2['Accuracy']:.4f}             | {metrics_s2['Accuracy'] - metrics_s1['Accuracy']:+.4f}
    Precision           | {metrics_s1['Precision']:.4f}           | {metrics_s2['Precision']:.4f}            | {metrics_s2['Precision'] - metrics_s1['Precision']:+.4f}
    Recall              | {metrics_s1['Recall']:.4f}              | {metrics_s2['Recall']:.4f}               | {metrics_s2['Recall'] - metrics_s1['Recall']:+.4f}
    Specificity         | {metrics_s1['Specificity']:.4f}         | {metrics_s2['Specificity']:.4f}          | {metrics_s2['Specificity'] - metrics_s1['Specificity']:+.4f}
    Dice Coefficient    | {metrics_s1['Dice']:.4f}                | {metrics_s2['Dice']:.4f}                 | {metrics_s2['Dice'] - metrics_s1['Dice']:+.4f}
    IoU Coefficient     | {metrics_s1['IoU']:.4f}                 | {metrics_s2['IoU']:.4f}                  | {metrics_s2['IoU'] - metrics_s1['IoU']:+.4f}

    SURFACE-BASED METRICS (lower is better):
                            | Scenario 1 (Binary) | Scenario 2 (3-class) | Improvement
    --------------------|---------------------|----------------------|------------
    HD95 (pixels)       | {statistical_results['hd95_scenario1_mean']:.4f} ± {statistical_results['hd95_scenario1_std']:.4f} | {statistical_results['hd95_scenario2_mean']:.4f} ± {statistical_results['hd95_scenario2_std']:.4f} | {statistical_results['hd95_improvement']:+.4f}
    ASSD (pixels)       | {statistical_results['assd_scenario1_mean']:.4f} ± {statistical_results['assd_scenario1_std']:.4f} | {statistical_results['assd_scenario2_mean']:.4f} ± {statistical_results['assd_scenario2_std']:.4f} | {statistical_results['assd_improvement']:+.4f}

    Note: For HD95 and ASSD, positive improvement means reduction (better boundary accuracy)
    Valid samples: HD95={statistical_results['sample_size_hd95']}/{statistical_results['sample_size']}, ASSD={statistical_results['sample_size_assd']}/{statistical_results['sample_size']}

    STATISTICAL SIGNIFICANCE:
    -------------------------
    DICE COEFFICIENT:
    Test: Paired t-test
    t-statistic: {statistical_results['dice_t_statistic']:.4f}
    p-value: {statistical_results['dice_p_value']:.4f}
    Effect Size (Cohen's d): {statistical_results['dice_effect_size']:.4f}
    95% Confidence Interval: [{statistical_results['dice_ci_lower']:.4f}, {statistical_results['dice_ci_upper']:.4f}]
    Result: {'SIGNIFICANT' if statistical_results['dice_significant'] else 'NOT SIGNIFICANT'} improvement

    IoU COEFFICIENT:
    Test: Paired t-test
    t-statistic: {statistical_results['iou_t_statistic']:.4f}
    p-value: {statistical_results['iou_p_value']:.4f}
    Effect Size (Cohen's d): {statistical_results['iou_effect_size']:.4f}
    95% Confidence Interval: [{statistical_results['iou_ci_lower']:.4f}, {statistical_results['iou_ci_upper']:.4f}]
    Result: {'SIGNIFICANT' if statistical_results['iou_significant'] else 'NOT SIGNIFICANT'} improvement

    HD95 (95th Percentile Hausdorff Distance):
    Test: Paired t-test
    t-statistic: {statistical_results['hd95_t_statistic']:.4f}
    p-value: {statistical_results['hd95_p_value']:.4f}
    Effect Size (Cohen's d): {statistical_results['hd95_effect_size']:.4f}
    95% Confidence Interval: [{statistical_results['hd95_ci_lower']:.4f}, {statistical_results['hd95_ci_upper']:.4f}] pixels
    Result: {'SIGNIFICANT' if statistical_results['hd95_significant'] else 'NOT SIGNIFICANT'} improvement

    ASSD (Average Symmetric Surface Distance):
    Test: Paired t-test
    t-statistic: {statistical_results['assd_t_statistic']:.4f}
    p-value: {statistical_results['assd_p_value']:.4f}
    Effect Size (Cohen's d): {statistical_results['assd_effect_size']:.4f}
    95% Confidence Interval: [{statistical_results['assd_ci_lower']:.4f}, {statistical_results['assd_ci_upper']:.4f}] pixels
    Result: {'SIGNIFICANT' if statistical_results['assd_significant'] else 'NOT SIGNIFICANT'} improvement

    KEY FINDINGS:
    -------------
    OVERLAP-BASED METRICS:
    1. Three-class segmentation shows {statistical_results['dice_improvement_percent']:.2f}% improvement in Dice coefficient
    2. Three-class segmentation shows {statistical_results['iou_improvement_percent']:.2f}% improvement in IoU coefficient
    3. Dice improvement is {'statistically significant (p<0.05)' if statistical_results['dice_significant'] else 'not statistically significant'}
    4. IoU improvement is {'statistically significant (p<0.05)' if statistical_results['iou_significant'] else 'not statistically significant'}

    SURFACE-BASED METRICS:
    5. HD95 shows {abs(statistical_results['hd95_improvement_percent']):.2f}% {'reduction' if statistical_results['hd95_improvement'] > 0 else 'increase'} (lower is better)
    6. ASSD shows {abs(statistical_results['assd_improvement_percent']):.2f}% {'reduction' if statistical_results['assd_improvement'] > 0 else 'increase'} (lower is better)
    7. HD95 improvement is {'statistically significant (p<0.05)' if statistical_results['hd95_significant'] else 'not statistically significant'}
    8. ASSD improvement is {'statistically significant (p<0.05)' if statistical_results['assd_significant'] else 'not statistically significant'}

    OVERALL ASSESSMENT:
    9. Post-processing provided substantial improvements in both scenarios
    10. Three-class approach shows {'consistent advantages' if statistical_results['dice_significant'] and statistical_results['iou_significant'] else 'mixed results'} across multiple metrics
    11. Boundary accuracy (HD95/ASSD) {'improved significantly' if statistical_results['hd95_significant'] or statistical_results['assd_significant'] else 'showed no significant improvement'}

    FILES GENERATED:
    ----------------
    - Models: scenario1_binary_model.h5, scenario2_multiclass_model.h5
    - Figures: training_curves.png/.pdf, comparison_visualization.png/.pdf, metrics_comparison.png/.pdf
    - Tables: comprehensive_results.csv/.xlsx, surface_metrics.csv/.xlsx, latex_table.tex, latex_surface_table.tex
    - Statistics: statistical_analysis.json, statistical_report.txt
    - Predictions: All test predictions and ground truth data saved

    PUBLICATION READINESS:
    ----------------------
    ✓ High-resolution figures (300 DPI, PNG/PDF)
    ✓ LaTeX-formatted tables (overlap and surface metrics)
    ✓ Comprehensive statistical analysis (Dice, IoU, HD95, ASSD)
    ✓ Post-processing impact analysis
    ✓ Reproducible results with saved models
    ✓ Professional documentation
    ✓ Surface-based metrics for boundary accuracy assessment
    """
        
        with open(self.results_dir / 'leverage_summary.txt', 'w') as f:
            f.write(summary)
            
        print("="*80)
        print("LEVERAGE RESULTS SUMMARY GENERATED")
        print("="*80)
        print(summary)
        
###################### Main Experiment Function ######################

def run_leverage_experiment():
    """Main function to run the complete leverage experiment"""
    
    print("="*80)
    print("STARTING LEVERAGE PAPER EXPERIMENT")
    print("="*80)
    
    # Initialize components
    plotter = PublicationPlotter(config.results_dir)
    saver = ResultsSaver(config.results_dir)
    
    # Load datasets
    print("\nLoading datasets...")
    train_images, train_masks_3class, train_masks_binary, dataset_info_train = load_wmh_dataset(
        config.train_dir, config.target_size
    )
    
    test_images, test_masks_3class, test_masks_binary, dataset_info_test = load_wmh_dataset(
        config.test_dir, config.target_size
    )

    # Split training data
    x_train, x_val, y_train_3class, y_val_3class, y_train_binary, y_val_binary = train_test_split(
        train_images, train_masks_3class, train_masks_binary, 
        test_size=config.validation_split, random_state=config.random_state
    )
    
    print(f"Training: {x_train.shape[0]}, Validation: {x_val.shape[0]}, Test: {test_images.shape[0]}")
    
    # Calculate class weights
    binary_weights = calculate_class_weights(y_train_binary, 2)
    multiclass_weights = calculate_class_weights(y_train_3class, 3)
    
    print(f"Binary class weights: {binary_weights}")
    print(f"Multi-class weights: {multiclass_weights}")
    
    # Scenario 1: Binary Classification
    print("\n" + "="*60)
    print("TRAINING SCENARIO 1: BINARY CLASSIFICATION")
    print("="*60)
    
    model_s1 = config.build_unet_variant(config.input_shape, num_classes=1)
    print(f"Scenario 1 Model Parameters: {model_s1.count_params():,}")
    model_s1.summary()  # Optional: for detailed architecture view
    
    # Configure loss function
    if config.loss_options['scenario1'] == 'weighted_bce':
        loss_s1 = weighted_binary_crossentropy(pos_weight=binary_weights[1])
    elif config.loss_options['scenario1'] == 'focal':
        loss_s1 = focal_loss(alpha=0.75, gamma=2.0)
    elif config.loss_options['scenario1'] == 'combined':
        loss_s1 = combined_loss(alpha=0.5, pos_weight=binary_weights[1])
    else:
        loss_s1 = 'binary_crossentropy'
    
    model_s1.compile(
        optimizer=optimizers.legacy.Adam(config.learning_rate),
        loss=loss_s1,
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks_s1 = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7)
    ]
    
    y_train_binary_expanded = np.expand_dims(y_train_binary, axis=-1)
    y_val_binary_expanded = np.expand_dims(y_val_binary, axis=-1)
    
    if config.mode == 'training':
        history_s1 = model_s1.fit(
            x_train, y_train_binary_expanded,
            validation_data=(x_val, y_val_binary_expanded),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks_s1,
            verbose=1
        )
    
    # Scenario 2: Three-class Classification
    print("\n" + "="*60)
    print("TRAINING SCENARIO 2: THREE-CLASS CLASSIFICATION")
    print("="*60)
    
    model_s2 = config.build_unet_variant(config.input_shape, num_classes=3)
    print(f"Scenario 2 Model Parameters: {model_s2.count_params():,}")
    model_s2.summary()  # Optional: for detailed architecture view
    
    # Configure loss function
    if config.loss_options['scenario2'] == 'weighted_categorical':
        loss_s2 = weighted_categorical_crossentropy(multiclass_weights)
    elif config.loss_options['scenario2'] == 'multiclass_dice':
        loss_s2 = multiclass_dice_loss(num_classes=3)
    else:
        loss_s2 = 'categorical_crossentropy'
    
    model_s2.compile(
        optimizer=optimizers.legacy.Adam(config.learning_rate),
        loss=loss_s2,
        metrics=['accuracy']
    )
    
    callbacks_s2 = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-7)
    ]
    
    y_train_3class_categorical = to_categorical(y_train_3class, num_classes=3)
    y_val_3class_categorical = to_categorical(y_val_3class, num_classes=3)
    
    if config.mode == 'training':
        history_s2 = model_s2.fit(
            x_train, y_train_3class_categorical,
            validation_data=(x_val, y_val_3class_categorical),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks_s2,
            verbose=1
        )

    # Save models
    if config.mode == 'training':
        saver.save_models(model_s1, model_s2, history_s1, history_s2)
    else:
        # or Load models
        model_s1, model_s2, history_s1, history_s2 = saver.load_models(config.intended_study_dir, loss_s1_func=loss_s1, loss_s2_func=loss_s2)
        # Check if models loaded successfully before using them
        if model_s1 is None or model_s2 is None:
            print("Failed to load models. Cannot proceed with predictions.")
            exit(1)

    # Generate predictions
    print("\n" + "="*60)
    print("GENERATING PREDICTIONS AND EVALUATION")
    print("="*60)

    test_pred_s1 = model_s1.predict(test_images, batch_size=config.batch_size)
    test_pred_s1_binary = (test_pred_s1.squeeze() > 0.5).astype(np.uint8)

    test_pred_s2 = model_s2.predict(test_images, batch_size=config.batch_size)
    test_pred_s2_classes = np.argmax(test_pred_s2, axis=-1)

    # Post-process predictions
    print("Applying post-processing...")
    test_pred_s1_binary_processed = post_process_predictions(
        test_pred_s1_binary, 
        min_object_size=5, 
        apply_opening=True, 
        kernel_size=2
    )

    abnormal_pred_from_3class = (test_pred_s2_classes == 2).astype(np.uint8)
    abnormal_pred_from_3class_processed = post_process_predictions(
        abnormal_pred_from_3class,
        min_object_size=5,
        apply_opening=True,
        kernel_size=2
    )

    # Calculate metrics for raw predictions (without surface metrics for aggregate)
    print("Calculating metrics for raw predictions...")
    metrics_s1_raw = calculate_comprehensive_metrics(
        test_masks_binary.flatten(), 
        test_pred_s1_binary.flatten(), 
        "Binary Classification (Raw)"
    )

    abnormal_true_from_3class = (test_masks_3class == 2).astype(np.uint8)
    metrics_s2_raw = calculate_comprehensive_metrics(
        abnormal_true_from_3class.flatten(),
        abnormal_pred_from_3class.flatten(),
        "Three-class Classification (Raw)"
    )

    # Calculate metrics for post-processed predictions (without surface metrics for aggregate)
    print("Calculating metrics for post-processed predictions...")
    metrics_s1 = calculate_comprehensive_metrics(
        test_masks_binary.flatten(), 
        test_pred_s1_binary_processed.flatten(), 
        "Binary Classification (Processed)"
    )

    metrics_s2 = calculate_comprehensive_metrics(
        abnormal_true_from_3class.flatten(),
        abnormal_pred_from_3class_processed.flatten(),
        "Three-class Classification (Processed)"
    )

    # Calculate per-image metrics for statistical analysis (including HD95 and ASSD)
    dice_scores_s1_raw, dice_scores_s2_raw = [], []
    dice_scores_s1, dice_scores_s2 = [], []
    iou_scores_s1_raw, iou_scores_s2_raw = [], []
    iou_scores_s1, iou_scores_s2 = [], []
    hd95_scores_s1_raw, hd95_scores_s2_raw = [], []
    hd95_scores_s1, hd95_scores_s2 = [], []
    assd_scores_s1_raw, assd_scores_s2_raw = [], []
    assd_scores_s1, assd_scores_s2 = [], []

    for i in range(test_images.shape[0]):
        # Raw predictions
        dice_s1_raw = dice_coefficient_multiclass(test_masks_binary[i].flatten(), test_pred_s1_binary[i].flatten(), 1)
        dice_s2_raw = dice_coefficient_multiclass(abnormal_true_from_3class[i].flatten(), abnormal_pred_from_3class[i].flatten(), 1)
        iou_s1_raw = iou_coefficient_multiclass(test_masks_binary[i].flatten(), test_pred_s1_binary[i].flatten(), 1)
        iou_s2_raw = iou_coefficient_multiclass(abnormal_true_from_3class[i].flatten(), abnormal_pred_from_3class[i].flatten(), 1)
        hd95_s1_raw = hausdorff_distance_95(test_masks_binary[i], test_pred_s1_binary[i])
        hd95_s2_raw = hausdorff_distance_95(abnormal_true_from_3class[i], abnormal_pred_from_3class[i])
        assd_s1_raw = average_symmetric_surface_distance(test_masks_binary[i], test_pred_s1_binary[i])
        assd_s2_raw = average_symmetric_surface_distance(abnormal_true_from_3class[i], abnormal_pred_from_3class[i])
        
        # Post-processed predictions
        dice_s1 = dice_coefficient_multiclass(test_masks_binary[i].flatten(), test_pred_s1_binary_processed[i].flatten(), 1)
        dice_s2 = dice_coefficient_multiclass(abnormal_true_from_3class[i].flatten(), abnormal_pred_from_3class_processed[i].flatten(), 1)
        iou_s1 = iou_coefficient_multiclass(test_masks_binary[i].flatten(), test_pred_s1_binary_processed[i].flatten(), 1)
        iou_s2 = iou_coefficient_multiclass(abnormal_true_from_3class[i].flatten(), abnormal_pred_from_3class_processed[i].flatten(), 1)
        hd95_s1 = hausdorff_distance_95(test_masks_binary[i], test_pred_s1_binary_processed[i])
        hd95_s2 = hausdorff_distance_95(abnormal_true_from_3class[i], abnormal_pred_from_3class_processed[i])
        assd_s1 = average_symmetric_surface_distance(test_masks_binary[i], test_pred_s1_binary_processed[i])
        assd_s2 = average_symmetric_surface_distance(abnormal_true_from_3class[i], abnormal_pred_from_3class_processed[i])
        
        dice_scores_s1_raw.append(dice_s1_raw)
        dice_scores_s2_raw.append(dice_s2_raw)
        dice_scores_s1.append(dice_s1)
        dice_scores_s2.append(dice_s2)
        iou_scores_s1_raw.append(iou_s1_raw)
        iou_scores_s2_raw.append(iou_s2_raw)
        iou_scores_s1.append(iou_s1)
        iou_scores_s2.append(iou_s2)
        hd95_scores_s1_raw.append(hd95_s1_raw)
        hd95_scores_s2_raw.append(hd95_s2_raw)
        hd95_scores_s1.append(hd95_s1)
        hd95_scores_s2.append(hd95_s2)
        assd_scores_s1_raw.append(assd_s1_raw)
        assd_scores_s2_raw.append(assd_s2_raw)
        assd_scores_s1.append(assd_s1)
        assd_scores_s2.append(assd_s2)

    dice_scores_s1_raw = np.array(dice_scores_s1_raw)
    dice_scores_s2_raw = np.array(dice_scores_s2_raw)
    dice_scores_s1 = np.array(dice_scores_s1)
    dice_scores_s2 = np.array(dice_scores_s2)
    iou_scores_s1_raw = np.array(iou_scores_s1_raw)
    iou_scores_s2_raw = np.array(iou_scores_s2_raw)
    iou_scores_s1 = np.array(iou_scores_s1)
    iou_scores_s2 = np.array(iou_scores_s2)
    hd95_scores_s1_raw = np.array(hd95_scores_s1_raw)
    hd95_scores_s2_raw = np.array(hd95_scores_s2_raw)
    hd95_scores_s1 = np.array(hd95_scores_s1)
    hd95_scores_s2 = np.array(hd95_scores_s2)
    assd_scores_s1_raw = np.array(assd_scores_s1_raw)
    assd_scores_s2_raw = np.array(assd_scores_s2_raw)
    assd_scores_s1 = np.array(assd_scores_s1)
    assd_scores_s2 = np.array(assd_scores_s2)

    # Filter out inf values for HD95 and ASSD while maintaining pairing
    # Create masks for valid (finite) values in both scenarios
    hd95_valid_mask = np.isfinite(hd95_scores_s1) & np.isfinite(hd95_scores_s2)
    assd_valid_mask = np.isfinite(assd_scores_s1) & np.isfinite(assd_scores_s2)
    hd95_valid_mask_raw = np.isfinite(hd95_scores_s1_raw) & np.isfinite(hd95_scores_s2_raw)
    assd_valid_mask_raw = np.isfinite(assd_scores_s1_raw) & np.isfinite(assd_scores_s2_raw)

    # Apply masks to both scenarios to keep them paired
    hd95_scores_s1 = hd95_scores_s1[hd95_valid_mask]
    hd95_scores_s2 = hd95_scores_s2[hd95_valid_mask]
    assd_scores_s1 = assd_scores_s1[assd_valid_mask]
    assd_scores_s2 = assd_scores_s2[assd_valid_mask]
    hd95_scores_s1_raw = hd95_scores_s1_raw[hd95_valid_mask_raw]
    hd95_scores_s2_raw = hd95_scores_s2_raw[hd95_valid_mask_raw]
    assd_scores_s1_raw = assd_scores_s1_raw[assd_valid_mask_raw]
    assd_scores_s2_raw = assd_scores_s2_raw[assd_valid_mask_raw]

    print(f"\nValid samples after filtering infinite values:")
    print(f"HD95: {len(hd95_scores_s1)} / {len(dice_scores_s1)} images")
    print(f"ASSD: {len(assd_scores_s1)} / {len(dice_scores_s1)} images")

    # Print comparison of raw vs processed results
    print(f"\nPost-processing Impact:")
    print(f"Scenario 1 - Dice improvement: {np.mean(dice_scores_s1) - np.mean(dice_scores_s1_raw):+.4f}")
    print(f"Scenario 1 - IoU improvement: {np.mean(iou_scores_s1) - np.mean(iou_scores_s1_raw):+.4f}")
    print(f"Scenario 1 - HD95 improvement: {np.mean(hd95_scores_s1_raw) - np.mean(hd95_scores_s1):+.4f} pixels (lower is better)")
    print(f"Scenario 1 - ASSD improvement: {np.mean(assd_scores_s1_raw) - np.mean(assd_scores_s1):+.4f} pixels (lower is better)")
    print(f"Scenario 2 - Dice improvement: {np.mean(dice_scores_s2) - np.mean(dice_scores_s2_raw):+.4f}")
    print(f"Scenario 2 - IoU improvement: {np.mean(iou_scores_s2) - np.mean(iou_scores_s2_raw):+.4f}")
    print(f"Scenario 2 - HD95 improvement: {np.mean(hd95_scores_s2_raw) - np.mean(hd95_scores_s2):+.4f} pixels (lower is better)")
    print(f"Scenario 2 - ASSD improvement: {np.mean(assd_scores_s2_raw) - np.mean(assd_scores_s2):+.4f} pixels (lower is better)") 
    
    # Statistical analysis
    print("\nPerforming statistical analysis...")
    statistical_results = saver.save_statistical_analysis(
        dice_scores_s1, dice_scores_s2, iou_scores_s1, iou_scores_s2,
        hd95_scores_s1, hd95_scores_s2, assd_scores_s1, assd_scores_s2,
        metrics_s1, metrics_s2
    )
    
    # Save all results
    print("\nSaving results...")
    saver.save_predictions(
        test_images, test_masks_3class, test_masks_binary,
        test_pred_s1_binary, test_pred_s2_classes, 
        {'train': dataset_info_train, 'test': dataset_info_test}
    )
    
    results_table, surface_table = saver.save_metrics_table(metrics_s1, metrics_s2, statistical_results)

    # Generate visualizations
    print("\nGenerating publication-quality figures...")
    plotter.plot_training_curves(history_s1, history_s2)
    plotter.plot_comparison_visualization(
        test_images, test_masks_3class, test_masks_binary,
        test_pred_s1_binary_processed, abnormal_pred_from_3class_processed
    )
    plotter.plot_metrics_comparison(metrics_s1, metrics_s2)
    plotter.plot_dice_distribution(dice_scores_s1, dice_scores_s2)
    plotter.plot_dice_iou_distribution(dice_scores_s1, dice_scores_s2, iou_scores_s1, iou_scores_s2)

    
    # Generate final summary
    saver.generate_leverage_summary(
        config, dataset_info_train, dataset_info_test,
        metrics_s1, metrics_s2, statistical_results
    )
    
    return {
        'config': config,
        'models': {'scenario1': model_s1, 'scenario2': model_s2},
        'histories': {'scenario1': history_s1, 'scenario2': history_s2},
        'metrics': {'scenario1': metrics_s1, 'scenario2': metrics_s2},
        'statistical_results': statistical_results,
        'results_table': results_table,
        'surface_table': surface_table
    }


def create_comparative_analysis(all_results, models_to_test):
    """Create a comparative analysis across all models"""
    import pandas as pd
    from pathlib import Path
    
    # Create a comparative results directory
    comparative_dir = Path(f"leverage_comparative_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    comparative_dir.mkdir(exist_ok=True)
    
    # Collect data for comparison
    comparison_data = []
    
    for model_name in models_to_test:
        if all_results[model_name] is not None:
            result = all_results[model_name]
            metrics_s1 = result['metrics']['scenario1']
            metrics_s2 = result['metrics']['scenario2']
            stats = result['statistical_results']
            
            # Scenario 1
            comparison_data.append({
                'Model': model_name,
                'Scenario': 'Binary (S1)',
                'Accuracy': metrics_s1['Accuracy'],
                'Precision': metrics_s1['Precision'],
                'Recall': metrics_s1['Recall'],
                'Dice': metrics_s1['Dice'],
                'IoU': metrics_s1['IoU']
            })
            
            # Scenario 2
            comparison_data.append({
                'Model': model_name,
                'Scenario': 'Three-class (S2)',
                'Accuracy': metrics_s2['Accuracy'],
                'Precision': metrics_s2['Precision'],
                'Recall': metrics_s2['Recall'],
                'Dice': metrics_s2['Dice'],
                'IoU': metrics_s2['IoU']
            })
    
    # Create DataFrame
    df_comparison = pd.DataFrame(comparison_data)
    
    # Save to CSV and Excel
    df_comparison.to_csv(comparative_dir / 'model_comparison.csv', index=False)
    df_comparison.to_excel(comparative_dir / 'model_comparison.xlsx', index=False)
    
    # Create improvement summary
    improvement_data = []
    for model_name in models_to_test:
        if all_results[model_name] is not None:
            stats = all_results[model_name]['statistical_results']
            improvement_data.append({
                'Model': model_name,
                'Dice_Improvement': stats['dice_improvement'],
                'Dice_p_value': stats['dice_p_value'],
                'Dice_Significant': stats['dice_significant'],
                'IoU_Improvement': stats['iou_improvement'],
                'IoU_p_value': stats['iou_p_value'],
                'IoU_Significant': stats['iou_significant'],
                'HD95_Improvement': stats['hd95_improvement'],
                'HD95_p_value': stats['hd95_p_value'],
                'ASSD_Improvement': stats['assd_improvement'],
                'ASSD_p_value': stats['assd_p_value']
            })
    
    df_improvement = pd.DataFrame(improvement_data)
    df_improvement.to_csv(comparative_dir / 'improvement_comparison.csv', index=False)
    df_improvement.to_excel(comparative_dir / 'improvement_comparison.xlsx', index=False)
    
    # Create a summary report
    summary_report = f"""
COMPARATIVE ANALYSIS ACROSS ALL MODELS
========================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

MODELS TESTED:
{', '.join([m.upper() for m in models_to_test if all_results[m] is not None])}

PERFORMANCE COMPARISON:
-----------------------
{df_comparison.to_string(index=False)}

IMPROVEMENT ANALYSIS:
---------------------
{df_improvement.to_string(index=False)}

BEST PERFORMING MODEL:
----------------------
Best Dice (S2): {df_comparison[df_comparison['Scenario'] == 'Three-class (S2)'].nlargest(1, 'Dice')['Model'].values[0].upper()}
Best IoU (S2): {df_comparison[df_comparison['Scenario'] == 'Three-class (S2)'].nlargest(1, 'IoU')['Model'].values[0].upper()}
Largest Dice Improvement: {df_improvement.nlargest(1, 'Dice_Improvement')['Model'].values[0].upper()}
Largest IoU Improvement: {df_improvement.nlargest(1, 'IoU_Improvement')['Model'].values[0].upper()}

Files saved in: {comparative_dir}
"""
    
    with open(comparative_dir / 'comparative_summary.txt', 'w') as f:
        f.write(summary_report)
    
    print(f"\n\nComparative analysis saved in: {comparative_dir}")
    print(summary_report)

###################### Execute Experiment ######################

if __name__ == "__main__":
    # List of all models to test
    models_to_test = ['unet', 'attn_unet', 'trans_unet', 'deepl3_unet']
    
    # Dictionary to store results from all models
    all_results = {}
    
    print("\n" + "="*80)
    print("STARTING MULTI-MODEL LEVERAGE PAPER EXPERIMENT")
    print("="*80)
    print(f"Models to test: {', '.join(models_to_test)}")
    print("="*80)
    
    # Run experiment for each model
    for model_idx, model_name in enumerate(models_to_test, 1):
        print("\n" + "="*80)
        print(f"RUNNING EXPERIMENT {model_idx}/{len(models_to_test)}: {model_name.upper()}")
        print("="*80)
        
        # Create new config for this model
        config = Config()
        config.model_name = model_name
        
        # Update build_unet_variant based on model name
        if config.model_name == 'unet':
            config.build_unet_variant = build_unet_3class
        elif config.model_name == 'attn_unet':
            config.build_unet_variant = build_attention_unet_3class
        elif config.model_name == 'trans_unet':
            config.build_unet_variant = build_trans_unet_3class
        elif config.model_name == 'deepl3_unet':
            config.build_unet_variant = build_deeplabv3_unet_3class
        
        # Recreate results directory with correct model name
        config.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config.mode != 'training':
            config.results_dir = Path(f"leverage_results_{config.timestamp}_{config.model_name}_no_training")
        else:
            config.results_dir = Path(f"leverage_results_{config.timestamp}_{config.model_name}")
        config.create_directory_structure()

        # if we work on non-training mode, load (use) the previously trained models
        if config.model_name == 'unet':
            config.intended_study_dir = "leverage_results_20251124_152044_unet"
        elif config.model_name == 'attn_unet':
            config.intended_study_dir = "leverage_results_20251125_133300_attn_unet"
        elif config.model_name == 'trans_unet':
            config.intended_study_dir = "leverage_results_20251124_171430_trans_unet"
        elif config.model_name == 'deepl3_unet':
            config.intended_study_dir = "leverage_results_20251124_180934_deepl3_unet"
        
        # Set seeds for reproducibility for each model
        np.random.seed(config.random_state)
        tf.random.set_seed(config.random_state)
        
        try:
            # Run the complete experiment for this model
            results = run_leverage_experiment()
            all_results[model_name] = results
            
            print("\n" + "="*80)
            print(f"EXPERIMENT FOR {model_name.upper()} COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Results saved in: {config.results_dir}")
            print("="*80)
            
        except Exception as e:
            print("\n" + "="*80)
            print(f"ERROR: Experiment for {model_name.upper()} failed!")
            print("="*80)
            print(f"Error message: {str(e)}")
            print("Continuing with next model...")
            print("="*80)
            all_results[model_name] = None
            continue
    
    # Generate comparative summary across all models
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print("\nSUMMARY OF ALL MODELS:")
    print("-" * 80)
    
    for model_name in models_to_test:
        if all_results[model_name] is not None:
            result = all_results[model_name]
            metrics_s1 = result['metrics']['scenario1']
            metrics_s2 = result['metrics']['scenario2']
            stats = result['statistical_results']
            
            print(f"\n{model_name.upper()}:")
            print(f"  Results directory: {result['config'].results_dir}")
            print(f"  Scenario 1 - Dice: {metrics_s1['Dice']:.4f}, IoU: {metrics_s1['IoU']:.4f}")
            print(f"  Scenario 2 - Dice: {metrics_s2['Dice']:.4f}, IoU: {metrics_s2['IoU']:.4f}")
            print(f"  Dice Improvement: {stats['dice_improvement']:.4f} (p={stats['dice_p_value']:.4f})")
            print(f"  IoU Improvement: {stats['iou_improvement']:.4f} (p={stats['iou_p_value']:.4f})")
        else:
            print(f"\n{model_name.upper()}: FAILED")
    
    print("\n" + "="*80)
    print("All files are ready for leverage paper submission!")
    print("="*80)
    
    # Optionally: Create a comparative analysis file
    create_comparative_analysis(all_results, models_to_test)
