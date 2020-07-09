import tensorflow.keras as tfk
import tensorflow as tf
import tensorflow_probability as tfp

from blendinator.models.blocks import conv2d_normal_reg
tfd = tfp.distributions


def combinatory(latent_dim, input_shape, nb_conv_1x1):
    """
    The gaussian encoder is a submodel of the probabilistic Unet.
    It samples from a Multivariate distribution parameterized by the output of the gaussian encoder,
    concatenate it to the unet features and convolves to produce the final ont hot encoded output.

    The inputs are [Tensor((None, 128, 128, n), dtype=float32), Tensor((None, 2*altent_dim), dtype=float32)]
    The output is a Tensor((None, 128, 128, 3), dtype=float32))
    """

    z_input = tfk.layers.Input(shape=2*latent_dim, name='Latent_space')
    features_input = tfk.layers.Input(shape=(input_shape[0], input_shape[1], 3), name='U-net_Features')

    """ Create the gaussian and sample from it """
    mu = z_input[:, :latent_dim]
    log_sigma = z_input[:, latent_dim:]
    sample = tfp.layers.DistributionLambda(make_distribution_fn=lambda t:\
            tfd.MultivariateNormalDiag(loc=t[0], scale_diag=tf.exp(t[1])),
            convert_to_tensor_fn=lambda s: s.sample(), name='Sample_from_the_LS')([mu, log_sigma])

    """ Make the sampling concatanable to the features"""
    shape = features_input.shape
    spatial_shape = [shape[axis] for axis in [1,2]]
    multiples = [1] + spatial_shape
    multiples.insert(3, 1)
    if len(sample.shape) == 2:
        sample = tfk.layers.Reshape((1, 1, latent_dim))(sample)
    broadcast_sample = tf.tile(sample, multiples)
    features = tf.concat([features_input, broadcast_sample], axis =-1)
    # features = features_input
    for i in range(nb_conv_1x1):
        features = conv2d_normal_reg(features, 3, 3, 'relu', f'combinatory_conv_{i}')
    mu_log_sigma = features
    # combinatory_output = features
    return tf.keras.Model(inputs=[features_input, z_input], outputs=[features], name='Combinatory')