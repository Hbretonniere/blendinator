import tensorflow.keras as tfk

from blendinator.models.blocks import upblock, downblock, conv2d_normal_reg


def unet(input_shape, list_num_channels, block_size):
    """
    Unet
    The unet is a submodel of the probabilistic Unet. I don't think there is a need of making the Encoder and the decoder sub-submodels.
    The Encoder use the downblock function, the Decoder the upblock function.

    The Input is a Tensor((None, 128, 128, 1), dtype=float32)

    The output is a Tensor((None, 128, 128, n), dtype=float32)), with $n>1$
    """

    unet_input = tfk.layers.Input(shape=input_shape, name='Image')
    features = [unet_input]
    for i, n_channels in enumerate(list_num_channels):
        if i == 0:
            downsample = False
        else:
            downsample = True

        features.append(downblock(features[-1],
                                n_channels,
                                block_size=block_size,
                                downsample=downsample,
                                name=f'Unet_downblock_{i}'))
    encoder_output = features[1:]
    
    n = len(encoder_output) - 2

    lower_reso_features = encoder_output[-1]
    for i in range(n, -1, -1):
        same_reso_features = encoder_output[i]
        n_channels = list_num_channels[i]
        lower_reso_features = upblock(lower_reso_features,
                                    same_reso_features,
                                    output_channels=n_channels,
                                    block_size=block_size,
                                    name=f'Unet_upblock_{n-i}')

    unet_output = conv2d_normal_reg(lower_reso_features, 3, 3, activation='relu', name='last_unet_deconv')
    
    return tfk.Model(unet_input, unet_output, name='unet')