###################### Metrics and Evaluation ######################

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt

def calculate_class_weights(masks, num_classes):
    """Calculate class weights inversely proportional to class frequency"""
    flattened = masks.flatten()
    class_counts = np.bincount(flattened, minlength=num_classes)
    total_pixels = len(flattened)
    class_weights = total_pixels / (num_classes * class_counts)
    class_weights = class_weights / class_weights[0]
    return class_weights

def dice_coefficient_multiclass(y_true, y_pred, class_id):
    """Calculate Dice coefficient for specific class"""
    y_true_class = (y_true == class_id).astype(np.float32)
    y_pred_class = (y_pred == class_id).astype(np.float32)
    
    smooth = 1e-6
    intersection = np.sum(y_true_class * y_pred_class)
    return (2. * intersection + smooth) / (np.sum(y_true_class) + np.sum(y_pred_class) + smooth)

def iou_coefficient_multiclass(y_true, y_pred, class_id):
    """Calculate IoU (Intersection over Union) coefficient for specific class"""
    y_true_class = (y_true == class_id).astype(np.float32)
    y_pred_class = (y_pred == class_id).astype(np.float32)
    
    intersection = np.sum(y_true_class * y_pred_class)
    union = np.sum(y_true_class) + np.sum(y_pred_class) - intersection
    
    # Handle edge case where both masks are empty
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union

def compute_surface_distances(mask_gt, mask_pred):
    """
    Compute surface distances between ground truth and prediction masks.
    Returns distances from pred surface to GT surface and vice versa.
    """
    # Get surfaces (boundaries) of the masks
    # Use distance transform to find distances from each point to nearest surface
    
    # If either mask is empty, return None
    if np.sum(mask_gt) == 0 or np.sum(mask_pred) == 0:
        return None, None
    
    # Compute distance transforms
    # Distance from each point to the nearest background point
    dt_gt = distance_transform_edt(~mask_gt.astype(bool))
    dt_pred = distance_transform_edt(~mask_pred.astype(bool))
    
    # Get surface points (where distance transform is small, i.e., near boundary)
    surface_gt = (dt_gt <= 1) & mask_gt.astype(bool)
    surface_pred = (dt_pred <= 1) & mask_pred.astype(bool)
    
    # If no surface points found, return None
    if np.sum(surface_gt) == 0 or np.sum(surface_pred) == 0:
        return None, None
    
    # Compute distances from pred surface to GT
    distances_pred_to_gt = dt_gt[surface_pred]
    
    # Compute distances from GT surface to pred
    distances_gt_to_pred = dt_pred[surface_gt]
    
    return distances_pred_to_gt, distances_gt_to_pred

def hausdorff_distance_95(y_true, y_pred):
    """
    Calculate 95th percentile Hausdorff Distance (HD95).
    
    HD95 measures the 95th percentile of surface distances, making it robust to outliers.
    Returns distance in pixels.
    
    Args:
        y_true: Ground truth binary mask (2D array)
        y_pred: Predicted binary mask (2D array)
    
    Returns:
        hd95: 95th percentile Hausdorff distance in pixels, or np.inf if masks are empty
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    # Handle empty masks
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return 0.0  # Perfect match (both empty)
    if np.sum(y_true) == 0 or np.sum(y_pred) == 0:
        return np.inf  # One empty, one not - worst case
    
    # Get surface distances
    distances_pred_to_gt, distances_gt_to_pred = compute_surface_distances(y_true, y_pred)
    
    if distances_pred_to_gt is None or distances_gt_to_pred is None:
        return np.inf
    
    # Combine all distances
    all_distances = np.concatenate([distances_pred_to_gt, distances_gt_to_pred])
    
    # Calculate 95th percentile
    hd95 = np.percentile(all_distances, 95)
    
    return hd95

def average_symmetric_surface_distance(y_true, y_pred):
    """
    Calculate Average Symmetric Surface Distance (ASSD).
    
    ASSD measures the average distance between surfaces of prediction and ground truth,
    providing an evaluation of the segmentation's boundary accuracy.
    Returns distance in pixels.
    
    Args:
        y_true: Ground truth binary mask (2D array)
        y_pred: Predicted binary mask (2D array)
    
    Returns:
        assd: Average symmetric surface distance in pixels, or np.inf if masks are empty
    """
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    # Handle empty masks
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return 0.0  # Perfect match (both empty)
    if np.sum(y_true) == 0 or np.sum(y_pred) == 0:
        return np.inf  # One empty, one not - worst case
    
    # Get surface distances
    distances_pred_to_gt, distances_gt_to_pred = compute_surface_distances(y_true, y_pred)
    
    if distances_pred_to_gt is None or distances_gt_to_pred is None:
        return np.inf
    
    # Calculate average of all surface distances (symmetric)
    assd = (np.mean(distances_pred_to_gt) + np.mean(distances_gt_to_pred)) / 2.0
    
    return assd

def calculate_comprehensive_metrics(y_true, y_pred, scenario_name):
    """Calculate comprehensive segmentation metrics including HD95 and ASSD"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    dice = dice_coefficient_multiclass(y_true, y_pred, 1)
    iou = iou_coefficient_multiclass(y_true.reshape(y_pred.shape), y_pred, 1)
    
    # Additional metrics
    specificity = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    return {
        'Scenario': scenario_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'Specificity': specificity,
        'Dice': dice,
        'IoU': iou
    }

def calculate_comprehensive_metrics_with_surface(y_true_2d, y_pred_2d, scenario_name):
    """
    Calculate comprehensive segmentation metrics including surface-based metrics.
    
    Args:
        y_true_2d: Ground truth binary mask (2D array, single image)
        y_pred_2d: Predicted binary mask (2D array, single image)
        scenario_name: Name of the scenario for identification
    
    Returns:
        Dictionary with all metrics including HD95 and ASSD
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    # Flatten for standard metrics
    y_true_flat = y_true_2d.flatten()
    y_pred_flat = y_pred_2d.flatten()
    
    acc = accuracy_score(y_true_flat, y_pred_flat)
    prec = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    rec = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    dice = dice_coefficient_multiclass(y_true_flat, y_pred_flat, 1)
    iou = iou_coefficient_multiclass(y_true_flat, y_pred_flat, 1)
    
    # Additional metrics
    specificity = precision_score(y_true_flat, y_pred_flat, pos_label=0, zero_division=0)
    
    # Surface-based metrics (using 2D arrays)
    hd95 = hausdorff_distance_95(y_true_2d, y_pred_2d)
    assd = average_symmetric_surface_distance(y_true_2d, y_pred_2d)
    
    return {
        'Scenario': scenario_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'Specificity': specificity,
        'Dice': dice,
        'IoU': iou,
        'HD95': hd95,
        'ASSD': assd
    }