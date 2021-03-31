import numpy as np
import tensorflow as tf
from tensorflow.train import latest_checkpoint
import tensorflow.keras as tfk
import sys

from blendinator.models.combinatory import combinatory
from blendinator.models.unet import unet
from blendinator.models.encoders import image_encoder, image_with_label_encoder
from blendinator.loss import cross_entropy

class ProbaUNet:
    """ProbaUNet
    
    Attributes
    ----------
    input_shape :  int 
        The input shape of the model

    latent_dim : int
        Dimension of the latent space (dimension of the Multivariate Gaussian Normal)

    block_size : int
        Number of convolution layers in each block

    channels : list of int
        List of channels for the successive downblocks

    last_conv : int
        Number of convolution layers to apply to the output of the Unet and the gaussian encoder

    """
    def __init__(self, input_shape, latent_dim, channels, block_size, last_conv, optimizer=None):
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.channels = channels
        self.block_size = block_size
        self.last_conv = last_conv
        self.optimizer = optimizer if optimizer else tfk.optimizers.Adam 
        
        self._setup_model()
        
    def _setup_model(self):

        """ Create the different sub-models of the Proba Unet ;
        - the unet, which ouput a 3 classes temporary segmap 
        - the img_encoder, which output the parameters (mu, sigma) of a latent_dim dimensional gaussian from the encoder of an image
        - the img_with_label_encoder, which output the parameters (mu, sigma) of a latent_dim dimensional gaussian from a concatenation of the input image and the corresponding segmentation map (label)
        - the last_CNN, which convolves (last_conv times) a concatenation of the output of the unet and one of the two encoder to produce the final seg map
        The two others model are just a combination of the others, but do not have any specific weights. It's the ProbaUnet in training mode (sampling from the img_encoder_with_labels) and the ProbaUnet in prediction mode (sampling from the img_encoder). Updating one or the other just update the previously explained models."""
        
        self.unet = unet(self.input_shape, self.channels, self.block_size)
        self.img_encoder = image_encoder(self.input_shape, self.channels, self.block_size, self.latent_dim)
        self.img_with_label_encoder = image_with_label_encoder(self.input_shape, self.channels, self.block_size, self.latent_dim)
        self.last_CNN = combinatory(self.latent_dim, self.input_shape, self.last_conv)
        self.training_model = self._init_training_model()
        self.prediction_model = self._init_prediction_model()

    def _init_training_model(self):
        """ Create the Proba Unet used for training, using the other sub models """
        
        img_input = tfk.layers.Input(shape=self.input_shape, name='image')
        label_input = tfk.layers.Input(shape=self.input_shape, name= 'label')

        unet_output = self.unet(img_input)
        img_encoder_output = self.img_encoder([img_input])
        img_with_label_encoder_output = self.img_with_label_encoder([img_input, label_input])

        seg_map = self.last_CNN([
            unet_output, 
            img_with_label_encoder_output])

        return tfk.Model(
            inputs=[img_input, label_input],
            outputs=[seg_map, img_encoder_output, img_with_label_encoder_output],
            name="ProbaUNet_training")

    def _init_prediction_model(self):
        """ Create the Proba Unet used for prediction, using the other sub models """
        
        img_input = tfk.layers.Input(shape=self.input_shape, name='image')
        output = self.last_CNN([
            self.unet(img_input), 
            self.img_encoder(img_input)])

        return tfk.Model(img_input, output, name='ProbaUnet_prediction')

    def eval_step(self, features, labels, beta):
        punet_output = self.model([features, labels])
        return cross_entropy(label, punet_output, beta)

    def train_step(self, features, labels, lr, beta):
        """ Run the Punet in training mode (training_model), compute the loss and update the weights of all the submodels for one step. Return the total loss, the reconstruction loss and the KL. """
        
        with tf.GradientTape() as tape:
            punet_output = self.training_model([features, labels])
            total_loss, rec_loss, kl_loss = cross_entropy(self.latent_dim, labels, punet_output, beta)
            grads = tape.gradient(total_loss, self.training_model.trainable_variables)
        self.optimizer(lr).apply_gradients(zip(grads, self.training_model.trainable_variables))
        return [total_loss, rec_loss, kl_loss]

    def save_weights(self, epoch, checkpoint_path):
        """ Save the weights of all the submodels, with a name corresponding to the current training step """
        
        self.unet.save_weights(f'{checkpoint_path}/unet_weights/epoch_{epoch+1}')
        self.img_encoder.save_weights(f'{checkpoint_path}/img_encoder_weights/epoch_{epoch+1}')
        self.img_with_label_encoder.save_weights(f'{checkpoint_path}/img_with_label_encoder_weights/epoch_{epoch+1}')
        self.model_combiner.save_weights(f'{checkpoint_path}/model_combiner_weights/epoch_{epoch+1}')
        return()

    def load_weights(self, checkpoint_path):
        """ Load the weights into all the submodels of the PUnet, according to the epoch written in the `checkpoint` file inside the checkpoint_path"""
        self.unet.load_weights(latest_checkpoint(f'{checkpoint_path}/unet_weights/'))
        self.img_encoder.load_weights(latest_checkpoint(f'{checkpoint_path}/img_encoder_weights/'))
        self.img_with_label_encoder.load_weights(latest_checkpoint(f'{checkpoint_path}/img_with_label_encoder_weights/'))
        self.model_combiner.load_weights(latest_checkpoint(f'{checkpoint_path}/model_combiner_weights/'))
        return()

    def predict_and_plot(self, features, labels, nb_to_plot, mode):
        
        import matplotlib.pyplot as plt
        if mode == 'predict':
            predictions = self.training_model([features, labels])[0]
        elif mode == 'training':
            predictions = self.prediction_model([features])[0]
        else :
            sys.exit("You must chose the mode of prediction : 'predict' to use the ProbaUnet in prediction mode, 'training' for training mode")

        fig, ax = plt.subplots(nb_to_plot*2, 3, figsize=(nb_to_plot, nb_to_plot/3 * 2))
        ax[i, 0].set_title('Input Image')
        ax[i, 1].set_title('Input Gt')
        ax[i, 2].set_title('Predicted segmap')
        for i in range(nb_to_plot):
            segmap = np.argmax(predictions[i], axis=-1)
            ax[i, 0].imshow(features[i])
            ax[i, 1].imshow(labels[i])
            ax[i, 2].imshow(segmap)
            
        return