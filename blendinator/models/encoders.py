import tensorflow.keras as tfk
import tensorflow as tf
from blendinator.models.blocks import downblock, conv2d_normal_reg


def image_encoder(input_shape, list_num_channels, block_size, latent_dim):
    """
    The gaussian encoder is a submodel of the probabilistic Unet.
    It encodes the image to a n-dimensional gaussian
    It uses only downblocks

    Input : 
    - Tensor((None, 128, 128, 1), dtype=float32))

    Output :
     Vector[2*latent_dim], which will be interpreted as [mu, log_sigma]. I think it's better for now since it's a bit complicated to pass a distribution in input of a submodel. The sampling is made on the combinatory model.
     """

    encoder_input = tfk.layers.Input(shape=input_shape, dtype='float32', name='Image')  
    features = encoder_input
    for i, n_channels in enumerate(list_num_channels):
        if i == 0:
            downsample = False
        else :
            downsample = True

        features = downblock(features,
                            n_channels,
                            block_size=block_size,
                            downsample=downsample,
                            name=f'img_encoder_down_block_{i}')
    
    features = tf.reduce_mean(features, axis=[1, 2], keepdims=True)
    mu_log_sigma = conv2d_normal_reg(features, 2*latent_dim, 1, 'relu', 'img_encoder_gaussian_params')
    mu_log_sigma = tf.squeeze(mu_log_sigma, axis=[1, 2])
    encoder_output = mu_log_sigma 

    return tfk.Model(encoder_input, encoder_output, name='image_encoder')


def image_with_label_encoder(input_shape, list_num_channels, block_size, latent_dim):
    """
    The gaussian encoder is a submodel of the probabilistic Unet.
    It encodes a concatenation of the image and the labels into a n-dimensional gaussian
    It uses only downblocks

    Inputs : 
    - [Tensor((None, 128, 128,1), dtype=float32)), Tensor((None, 128, 128,1), dtype=float32))]

    Output :
     Vector[2*latent_dim], which will be interpreted as [mu, log_sigma]. I think it's better for now since it's a bit complicated to pass a distribution in input of a submodel. The sampling is made on the combinatory model.
     """

    label_input = tfk.layers.Input(shape=input_shape, name='Ground_Truth')
    img_input = tfk.layers.Input(shape=input_shape, name='Image')
    spatial_shape = img_input.get_shape()[-3:-1]

    features = tf.concat([img_input, label_input], axis=-1)

    for i, n_channels in enumerate(list_num_channels):
        downsample = True
        if i == 0:
            down_sample = False
        features = downblock(features,
                            n_channels,
                            block_size=block_size,
                            downsample=downsample,
                            name=f'img_label_encoder_down_block_{i}')
    
    features = tf.reduce_mean(features, axis=[1,2], keepdims=True)
    mu_log_sigma = conv2d_normal_reg(features, 2*latent_dim, 1, 'relu', 'img_label_encoder_gaussian_params')
    mu_log_sigma = tf.squeeze(mu_log_sigma, axis=[1,2])
    encoder_output = mu_log_sigma 

    return tfk.Model(inputs=[img_input, label_input], outputs=[encoder_output], name='image_with_label_encoder')