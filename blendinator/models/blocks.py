import tensorflow as tf
import tensorflow.keras as tfk


def conv2d_normal_reg(input, filters, kernel_size, activation, name):
    """Custom 2D convolution layer with normal initialization and L2 regularization"""
    return tfk.layers.Conv2D(
            filters, kernel_size,
            padding='same', 
            kernel_initializer=tfk.initializers.he_normal(),
            bias_initializer=tfk.initializers.TruncatedNormal(stddev=0.001),
            kernel_regularizer=tfk.regularizers.l2(l=1), 
            bias_regularizer=tfk.regularizers.l2(l=1), 
            name=name,
            activation=activation)(input)


def downblock(features,
              output_channels,
              block_size,
              kernel_size=3,
              activation='relu',
              downsample=True,
              name='down_block'):
    """A function which make a cycle of num_conv convolution/regularization/activation/downsampling"""

    if downsample:
        features = tfk.layers.AveragePooling2D(padding='same')(features)

    for j in range(block_size):
        features = conv2d_normal_reg(
            features, output_channels, kernel_size, activation, name)
    return features


def upblock(lower_res_inputs,
            same_res_inputs,
            output_channels,
            block_size,
            name,
            kernel_size=3,
            activation='relu'):
    """Upsampling block for the UNet architecture"""

    upsampled_inputs = tfk.backend.resize_images(
        lower_res_inputs, data_format='channels_last',
        height_factor=2, width_factor=2,
        interpolation='bilinear')
        
    features = tf.concat([upsampled_inputs, same_res_inputs], axis=-1)

    for j in range(block_size):
        features = conv2d_normal_reg(
            features, output_channels, kernel_size, activation, name)

    return features

