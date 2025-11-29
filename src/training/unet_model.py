###################### Libraries ######################
# Deep Learning
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate

def build_unet_3class(input_shape=(256, 256, 1), num_classes=3):
    """Enhanced U-Net architecture with batch normalization and dropout"""
    inputs = Input(input_shape)

    # Encoder with batch normalization
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    # c1 = keras.layers.BatchNormalization()(c1)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    # c1 = keras.layers.BatchNormalization()(c1)
    p1 = MaxPooling2D()(c1)
    p1 = keras.layers.Dropout(0.1)(p1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    # c2 = keras.layers.BatchNormalization()(c2)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    # c2 = keras.layers.BatchNormalization()(c2)
    p2 = MaxPooling2D()(c2)
    p2 = keras.layers.Dropout(0.1)(p2)

    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    # c3 = keras.layers.BatchNormalization()(c3)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    # c3 = keras.layers.BatchNormalization()(c3)
    p3 = MaxPooling2D()(c3)
    p3 = keras.layers.Dropout(0.2)(p3)

    c4 = Conv2D(512, 3, activation='relu', padding='same')(p3)
    # c4 = keras.layers.BatchNormalization()(c4)
    c4 = Conv2D(512, 3, activation='relu', padding='same')(c4)
    # c4 = keras.layers.BatchNormalization()(c4)
    p4 = MaxPooling2D()(c4)
    p4 = keras.layers.Dropout(0.2)(p4)

    # Bottleneck
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(p4)
    # c5 = keras.layers.BatchNormalization()(c5)
    c5 = Conv2D(1024, 3, activation='relu', padding='same')(c5)
    # c5 = keras.layers.BatchNormalization()(c5)
    c5 = keras.layers.Dropout(0.3)(c5)

    # Decoder
    u6 = Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = keras.layers.Dropout(0.2)(u6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(u6)
    # c6 = keras.layers.BatchNormalization()(c6)
    c6 = Conv2D(512, 3, activation='relu', padding='same')(c6)
    # c6 = keras.layers.BatchNormalization()(c6)

    u7 = Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = keras.layers.Dropout(0.2)(u7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(u7)
    # c7 = keras.layers.BatchNormalization()(c7)
    c7 = Conv2D(256, 3, activation='relu', padding='same')(c7)
    # c7 = keras.layers.BatchNormalization()(c7)

    u8 = Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = keras.layers.Dropout(0.1)(u8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(u8)
    # c8 = keras.layers.BatchNormalization()(c8)
    c8 = Conv2D(128, 3, activation='relu', padding='same')(c8)
    # c8 = keras.layers.BatchNormalization()(c8)

    u9 = Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = keras.layers.Dropout(0.1)(u9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(u9)
    # c9 = keras.layers.BatchNormalization()(c9)
    c9 = Conv2D(64, 3, activation='relu', padding='same')(c9)
    # c9 = keras.layers.BatchNormalization()(c9)

    # Output layer
    if num_classes == 1:
        outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    else:
        outputs = Conv2D(num_classes, 1, activation='softmax')(c9)
    
    return Model(inputs, outputs, name='UNet')
