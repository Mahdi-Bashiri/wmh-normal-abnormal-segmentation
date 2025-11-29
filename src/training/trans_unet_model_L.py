def build_trans_unet_3class_lightweight(input_shape=(256, 256, 1), num_classes=3):
    """
    Lightweight TransUNet variant with reduced transformer complexity
    Better suited for smaller datasets and faster training
    """
    inputs = layers.Input(input_shape)
    
    # CNN Encoder (same as above but with fewer channels)
    conv1 = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    conv1 = layers.Conv2D(64, 3, padding='same', activation='relu')(conv1)
    conv1 = layers.Dropout(0.1)(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = layers.Conv2D(64, 3, padding='same', activation='relu')(pool1)
    conv2 = layers.Conv2D(128, 3, padding='same', activation='relu')(conv2)
    conv2 = layers.Dropout(0.1)(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = layers.Conv2D(128, 3, padding='same', activation='relu')(pool2)
    conv3 = layers.Conv2D(256, 3, padding='same', activation='relu')(conv3)
    conv3 = layers.Dropout(0.2)(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = layers.Conv2D(256, 3, padding='same', activation='relu')(pool3)
    conv4 = layers.Conv2D(512, 3, padding='same', activation='relu')(conv4)
    conv4 = layers.Dropout(0.2)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Simplified Transformer Bottleneck
    bottleneck = layers.Conv2D(512, 3, padding='same', activation='relu')(pool4)
    bottleneck = layers.Dropout(0.3)(bottleneck)
    
    # Lightweight transformer with fewer parameters
    h, w, d_model = 16, 16, 512
    transformer_input = layers.Reshape((h * w, d_model))(bottleneck)
    
    # Single transformer layer with reduced complexity
    attention_output = layers.MultiHeadAttention(
        num_heads=4, key_dim=d_model // 4, dropout=0.1
    )(transformer_input, transformer_input)
    attention_output = layers.Dropout(0.1)(attention_output)
    transformer_output = layers.LayerNormalization()(transformer_input + attention_output)
    
    # Feed forward
    ffn = layers.Dense(d_model, activation='relu')(transformer_output)
    ffn = layers.Dropout(0.1)(ffn)
    transformer_final = layers.LayerNormalization()(transformer_output + ffn)
    
    # Reshape back
    bottleneck_enhanced = layers.Reshape((h, w, d_model))(transformer_final)
    bottleneck_enhanced = layers.Dropout(0.3)(bottleneck_enhanced)
    
    # CNN Decoder
    up1 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(bottleneck_enhanced)
    concat1 = layers.Concatenate()([up1, conv4])
    concat1 = layers.Dropout(0.2)(concat1)
    conv_up1 = layers.Conv2D(256, 3, padding='same', activation='relu')(concat1)
    conv_up1 = layers.Conv2D(256, 3, padding='same', activation='relu')(conv_up1)
    
    up2 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(conv_up1)
    concat2 = layers.Concatenate()([up2, conv3])
    concat2 = layers.Dropout(0.2)(concat2)
    conv_up2 = layers.Conv2D(128, 3, padding='same', activation='relu')(concat2)
    conv_up2 = layers.Conv2D(128, 3, padding='same', activation='relu')(conv_up2)
    
    up3 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(conv_up2)
    concat3 = layers.Concatenate()([up3, conv2])
    concat3 = layers.Dropout(0.1)(concat3)
    conv_up3 = layers.Conv2D(64, 3, padding='same', activation='relu')(concat3)
    conv_up3 = layers.Conv2D(64, 3, padding='same', activation='relu')(conv_up3)
    
    up4 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(conv_up3)
    concat4 = layers.Concatenate()([up4, conv1])
    concat4 = layers.Dropout(0.1)(concat4)
    conv_up4 = layers.Conv2D(32, 3, padding='same', activation='relu')(concat4)
    conv_up4 = layers.Conv2D(32, 3, padding='same', activation='relu')(conv_up4)
    
    # Output
    if num_classes == 1:
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(conv_up4)
    else:
        outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv_up4)
    
    model = tf.keras.Model(inputs, outputs, name='TransUNet_Lightweight')
    return model
