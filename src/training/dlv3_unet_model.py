###################### Libraries ######################
# Deep Learning
import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras import backend as K
from tensorflow.keras import layers, optimizers, callbacks
from keras.utils import to_categorical

def build_deeplabv3_unet_3class(input_shape=(256, 256, 1), num_classes=3):
    """
    Standard DeepLabV3+ implementation with ResNet-50 backbone
    Following the original paper: "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
    """
    
    def conv_block(x, filters, kernel_size=3, strides=1, dilation_rate=1, use_bias=False, name=None):
        """Standard convolution block with BN and ReLU"""
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same', 
                         dilation_rate=dilation_rate, use_bias=use_bias, name=name)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    def bottleneck_residual_block(x, filters, strides=1, dilation_rate=1, projection_shortcut=False, name_prefix=""):
        """ResNet-50 bottleneck block with optional atrous convolution"""
        shortcut = x
        
        # Projection shortcut if needed
        if projection_shortcut:
            shortcut = layers.Conv2D(filters * 4, 1, strides=strides, use_bias=False, 
                                   name=f"{name_prefix}_0_conv")(shortcut)
            shortcut = layers.BatchNormalization(name=f"{name_prefix}_0_bn")(shortcut)
        
        # Bottleneck layers
        x = layers.Conv2D(filters, 1, use_bias=False, name=f"{name_prefix}_1_conv")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_1_bn")(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, 3, strides=strides, padding='same', 
                         dilation_rate=dilation_rate, use_bias=False, name=f"{name_prefix}_2_conv")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_2_bn")(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters * 4, 1, use_bias=False, name=f"{name_prefix}_3_conv")(x)
        x = layers.BatchNormalization(name=f"{name_prefix}_3_bn")(x)
        
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        
        return x
    
    def aspp_block(x, filters=256):
        """Atrous Spatial Pyramid Pooling with proper implementation"""
        
        # ASPP branches
        # 1x1 convolution
        b1 = layers.Conv2D(filters, 1, use_bias=False, name='aspp_1x1')(x)
        b1 = layers.BatchNormalization(name='aspp_1x1_bn')(b1)
        b1 = layers.Activation('relu')(b1)
        
        # 3x3 convolution with rate = 6
        b2 = layers.Conv2D(filters, 3, padding='same', dilation_rate=6, use_bias=False, name='aspp_3x3_6')(x)
        b2 = layers.BatchNormalization(name='aspp_3x3_6_bn')(b2)
        b2 = layers.Activation('relu')(b2)
        
        # 3x3 convolution with rate = 12
        b3 = layers.Conv2D(filters, 3, padding='same', dilation_rate=12, use_bias=False, name='aspp_3x3_12')(x)
        b3 = layers.BatchNormalization(name='aspp_3x3_12_bn')(b3)
        b3 = layers.Activation('relu')(b3)
        
        # 3x3 convolution with rate = 18
        b4 = layers.Conv2D(filters, 3, padding='same', dilation_rate=18, use_bias=False, name='aspp_3x3_18')(x)
        b4 = layers.BatchNormalization(name='aspp_3x3_18_bn')(b4)
        b4 = layers.Activation('relu')(b4)
        
        # Image-level features (Global Average Pooling) - Simplified approach
        # Get input spatial dimensions
        input_shape = tf.shape(x)
        h, w = input_shape[1], input_shape[2]
        
        b5 = layers.GlobalAveragePooling2D(name='aspp_gap')(x)
        b5 = layers.Reshape((1, 1, -1))(b5)
        b5 = layers.Conv2D(filters, 1, use_bias=False, name='aspp_gap_conv')(b5)
        b5 = layers.BatchNormalization(name='aspp_gap_bn')(b5)
        b5 = layers.Activation('relu')(b5)
        
        # Use a resize function that handles KerasTensors properly
        def resize_to_input_shape(args):
            features, spatial_shape = args
            return tf.image.resize(features, spatial_shape, method='bilinear')
        
        b5 = layers.Lambda(resize_to_input_shape, name='aspp_gap_resize')([b5, [h, w]])
        
        # Concatenate all branches
        concat_features = layers.Concatenate(name='aspp_concat')([b1, b2, b3, b4, b5])
        
        # Final 1x1 convolution
        output = layers.Conv2D(filters, 1, use_bias=False, name='aspp_final_conv')(concat_features)
        output = layers.BatchNormalization(name='aspp_final_bn')(output)
        output = layers.Activation('relu')(output)
        output = layers.Dropout(0.1, name='aspp_dropout')(output)
        
        return output
    
    # Input layer
    inputs = layers.Input(input_shape, name='input')
    
    # ==================== ENCODER (ResNet-50 Backbone) ====================
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False, name='conv1')(inputs)
    x = layers.BatchNormalization(name='conv1_bn')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same', name='pool1')(x)
    
    # Stage 1 (conv2_x) - Low-level features for decoder
    x = bottleneck_residual_block(x, 64, strides=1, projection_shortcut=True, name_prefix='conv2_block1')
    x = bottleneck_residual_block(x, 64, name_prefix='conv2_block2')
    low_level_features = bottleneck_residual_block(x, 64, name_prefix='conv2_block3')
    
    # Stage 2 (conv3_x)
    x = bottleneck_residual_block(low_level_features, 128, strides=2, projection_shortcut=True, name_prefix='conv3_block1')
    x = bottleneck_residual_block(x, 128, name_prefix='conv3_block2')
    x = bottleneck_residual_block(x, 128, name_prefix='conv3_block3')
    x = bottleneck_residual_block(x, 128, name_prefix='conv3_block4')
    
    # Stage 3 (conv4_x) - With atrous convolution
    x = bottleneck_residual_block(x, 256, strides=1, dilation_rate=2, projection_shortcut=True, name_prefix='conv4_block1')
    x = bottleneck_residual_block(x, 256, dilation_rate=2, name_prefix='conv4_block2')
    x = bottleneck_residual_block(x, 256, dilation_rate=2, name_prefix='conv4_block3')
    x = bottleneck_residual_block(x, 256, dilation_rate=2, name_prefix='conv4_block4')
    x = bottleneck_residual_block(x, 256, dilation_rate=2, name_prefix='conv4_block5')
    x = bottleneck_residual_block(x, 256, dilation_rate=2, name_prefix='conv4_block6')
    
    # Stage 4 (conv5_x) - With higher atrous rate
    x = bottleneck_residual_block(x, 512, strides=1, dilation_rate=4, projection_shortcut=True, name_prefix='conv5_block1')
    x = bottleneck_residual_block(x, 512, dilation_rate=4, name_prefix='conv5_block2')
    x = bottleneck_residual_block(x, 512, dilation_rate=4, name_prefix='conv5_block3')
    
    # ==================== ASPP MODULE ====================
    x = aspp_block(x, filters=256)
    
    # ==================== DECODER ====================
    
    # Use fixed upsampling - the spatial relationship should be predictable
    # ASPP output is at 1/16 resolution, low_level_features at 1/4 resolution
    # So we need 4x upsampling to match
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear', name='decoder_upsample1')(x)
    
    # Process low-level features
    low_level_features = layers.Conv2D(48, 1, use_bias=False, name='decoder_low_level_conv')(low_level_features)
    low_level_features = layers.BatchNormalization(name='decoder_low_level_bn')(low_level_features)
    low_level_features = layers.Activation('relu')(low_level_features)
    
    # If there's still a size mismatch, crop or pad to match
    def match_spatial_dims(tensors):
        high_level, low_level = tensors
        # Get shapes
        high_shape = tf.shape(high_level)
        low_shape = tf.shape(low_level)
        
        # Crop high_level to match low_level if it's larger
        high_level_matched = high_level[:, :low_shape[1], :low_shape[2], :]
        return high_level_matched, low_level
    
    x_matched, low_level_matched = layers.Lambda(match_spatial_dims, name='match_dims')([x, low_level_features])
    
    # Concatenate high-level and low-level features
    x = layers.Concatenate(name='decoder_concat')([x_matched, low_level_matched])
    
    # Refine features
    x = layers.Conv2D(256, 3, padding='same', use_bias=False, name='decoder_conv1')(x)
    x = layers.BatchNormalization(name='decoder_conv1_bn')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1, name='decoder_dropout1')(x)  # Light regularization
    
    x = layers.Conv2D(256, 3, padding='same', use_bias=False, name='decoder_conv2')(x)
    x = layers.BatchNormalization(name='decoder_conv2_bn')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.1, name='decoder_dropout2')(x)
    
    # Final upsampling to original resolution (4x upsampling)
    x = layers.UpSampling2D(size=(4, 4), interpolation='bilinear', name='decoder_upsample2')(x)
    
    # ==================== OUTPUT ====================
    
    # Output layer - preserving your original conditional logic
    if num_classes == 1:
        outputs = layers.Conv2D(1, 1, activation='sigmoid', name='output')(x)
    else:
        outputs = layers.Conv2D(num_classes, 1, activation='softmax', name='output')(x)
    
    # Create model
    model = keras.Model(inputs, outputs, name='DeepLabV3Plus_ResNet50')
    
    return model
