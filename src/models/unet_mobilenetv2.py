import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, SeparableConv2D, BatchNormalization, ReLU, Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def get_unet_mobilenetv2(input_shape=(288, 512, 3), max_lanes=4):
    inputs = Input(shape=input_shape)
    
    # Load MobileNetV2 pretrained on ImageNet
    encoder = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Extract skip connections
    # MobileNetV2 layers:
    # block_3_expand_relu (stride 4)
    # block_6_expand_relu (stride 8)
    # block_13_expand_relu (stride 16)
    # out_relu (stride 32)
    
    skips = [
        encoder.get_layer('block_3_expand_relu').output,
        encoder.get_layer('block_6_expand_relu').output,
        encoder.get_layer('block_13_expand_relu').output,
        encoder.get_layer('out_relu').output
    ]
    
    extractor = Model(inputs=encoder.input, outputs=skips)
    # Set to False initially for warm-up
    extractor.trainable = False 
    
    s1, s2, s3, s4 = extractor(inputs)
    
    # Auxiliary head for lane existence
    # gap = GlobalAveragePooling2D()(s4)
    # aux_out = Dense(max_lanes, activation='sigmoid', name='aux_out')(gap)
    
    # Decoder
    def decoder_block(x, skip, filters):
        x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = Concatenate()([x, skip])
        
        x = SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        x = SeparableConv2D(filters, 3, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x
        
    x = decoder_block(s4, s3, 256)
    x = decoder_block(x, s2, 128)
    x = decoder_block(x, s1, 64)
    
    # Upsample to stride 2
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = SeparableConv2D(32, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Upsample to stride 1
    x = UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
    x = SeparableConv2D(16, 3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    mask_out = Conv2D(1, 1, activation='sigmoid', name='mask_out')(x)
    
    model = Model(inputs, mask_out, name="MobileNetV2_UNet")
    return model
