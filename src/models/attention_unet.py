###################### Libraries ######################
# Deep Learning
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate

def build_attention_unet_3class(input_shape=(256, 256, 1), num_classes=3):
    """Enhanced Attention U-Net architecture with dropout"""
    
    def attention_block(F_g, F_l, F_int):
        """Attention gate implementation"""
        W_g = Conv2D(F_int, 1, padding='same')(F_g)
        W_x = Conv2D(F_int, 1, padding='same')(F_l)
        psi = keras.layers.Add()([W_g, W_x])
        psi = keras.layers.Activation('relu')(psi)
        psi = Conv2D(1, 1, padding='same')(psi)
        psi = keras.layers.Activation('sigmoid')(psi)
        return keras.layers.Multiply()([F_l, psi])
    
    inputs = Input(input_shape)
    
    # Encoder with dropout (matching your original dropout pattern)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(2)(c1)
    p1 = keras.layers.Dropout(0.1)(p1)
    
    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(2)(c2)
    p2 = keras.layers.Dropout(0.1)(p2)
    
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D(2)(c3)
    p3 = keras.layers.Dropout(0.2)(p3)
    
    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(2)(c4)
    p4 = keras.layers.Dropout(0.2)(p4)
    
    # Bridge
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(c5)
    c5 = keras.layers.Dropout(0.3)(c5)
    
    # Decoder with attention gates (using Conv2DTranspose - more standard)
    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    att6 = attention_block(u6, c4, 256)
    u6 = concatenate([u6, att6])
    u6 = keras.layers.Dropout(0.2)(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(c6)
    
    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    att7 = attention_block(u7, c3, 128)
    u7 = concatenate([u7, att7])
    u7 = keras.layers.Dropout(0.2)(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(c7)
    
    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    att8 = attention_block(u8, c2, 64)
    u8 = concatenate([u8, att8])
    u8 = keras.layers.Dropout(0.1)(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(c8)
    
    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    att9 = attention_block(u9, c1, 32)
    u9 = concatenate([u9, att9])
    u9 = keras.layers.Dropout(0.1)(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(c9)
    
    # Output layer - preserving your original conditional logic
    if num_classes == 1:
        outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    else:
        outputs = Conv2D(num_classes, 1, activation='softmax')(c9)
    
    return Model(inputs, outputs)

