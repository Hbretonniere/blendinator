import tensorflow as tf
import tensorflow.keras as tfk


def conv2d_normal_reg(input, filters, kernel_size, activation, name):
    """Custom 2D convolution layer with normal initialization and L2 regularization"""
    return tfk.layers.Conv2D(
            filters, 
            kernel_size,
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
        features = tfk.layers.AveragePooling2D(padding='same', name=name+'_Downsample')(features)

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

#     upsampled_inputs = tfk.backend.resize_images(
#         lower_res_inputs, data_format='channels_last',
#         height_factor=2, width_factor=2,
#         interpolation='bilinear')
    upsampled_inputs = tf.image.resize(lower_res_inputs, 
                    [lower_res_inputs.shape[1]*2, lower_res_inputs.shape[2]*2],
                     method='bilinear',
                     name=name+'_Upsample')

    features = tfk.layers.concatenate([upsampled_inputs, same_res_inputs], axis=-1, name=name+'_encoder_concat')

    for j in range(block_size):
        features = conv2d_normal_reg(
            features, output_channels, kernel_size, activation, name)

    return features

