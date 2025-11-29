###################### Libraries ######################
from keras import backend as K
import tensorflow as tf

###################### Loss Functions ######################

def weighted_binary_crossentropy(pos_weight=1.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_true = tf.cast(y_true, tf.float32)  
        
        # Calculate weighted cross-entropy
        loss_pos = pos_weight * y_true * K.log(y_pred)
        loss_neg = (1 - y_true) * K.log(1 - y_pred)
        
        return -K.mean(loss_pos + loss_neg)
    return loss

def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_true = tf.cast(y_true, tf.float32)  # Add this line
        pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_t = tf.where(K.equal(y_true, 1), alpha, 1 - alpha)
        focal_loss = -alpha_t * K.pow(1 - pt, gamma) * K.log(pt)
        return K.mean(focal_loss)
    return loss

def dice_loss():
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, tf.float32)  # Add this line
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return 1 - dice_coef
    return loss

def dice_binary():
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, tf.float32)  # Add this line
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        return dice_coef
    return loss

def combined_loss(alpha=0.5, pos_weight=100.0):
    def loss(y_true, y_pred):
        bce = weighted_binary_crossentropy(pos_weight)(y_true, y_pred)
        dice = dice_loss()(y_true, y_pred)
        return alpha * bce + (1 - alpha) * dice
    return loss

def weighted_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        y_true = tf.cast(y_true, tf.float32)  # Add this line
        weights = tf.constant(class_weights, dtype=tf.float32)
        class_weights_tensor = tf.gather(weights, K.argmax(y_true, axis=-1))
        cross_entropy = -K.sum(y_true * K.log(y_pred), axis=-1)
        weighted_loss = cross_entropy * class_weights_tensor
        return K.mean(weighted_loss)
    return loss

def multiclass_dice_loss(num_classes=3):
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, tf.float32)  # Add this line
        dice_scores = []
        for class_idx in range(num_classes):
            y_true_class = y_true[..., class_idx]
            y_pred_class = y_pred[..., class_idx]
            y_true_f = K.flatten(y_true_class)
            y_pred_f = K.flatten(y_pred_class)
            intersection = K.sum(y_true_f * y_pred_f)
            dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
            dice_scores.append(dice_coef)
        mean_dice = K.mean(K.stack(dice_scores))
        return 1 - mean_dice
    return loss

def multiclass_dice():
    def loss(y_true, y_pred):
        smooth = 1e-6
        y_true = tf.cast(y_true, tf.float32)  # Add this line
        dice_scores = []
        num_classes = tf.shape(y_true)[-1]  # Get number of classes dynamically
        for class_idx in range(num_classes):
            y_true_class = y_true[..., class_idx]
            y_pred_class = y_pred[..., class_idx]
            y_true_f = K.flatten(y_true_class)
            y_pred_f = K.flatten(y_pred_class)
            intersection = K.sum(y_true_f * y_pred_f)
            dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
            dice_scores.append(dice_coef)
        mean_dice = K.mean(K.stack(dice_scores))
        return mean_dice
    return loss
