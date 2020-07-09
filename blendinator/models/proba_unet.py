import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

from blendinator.models.combinatory import combinatory
from blendinator.models.unet import unet
from blendinator.models.encoders import image_encoder, image_with_label_encoder
from blendinator.loss import cross_entropy

class ProbaUNet:
    """ProbaUNet
    
    Attributes
    ----------

    latent_dim : int
        Dimension of the latent space (dimension of the Multivariate Gaussian Normal)

    block_size : int
        Number of convolution layers in each block

    channels : list of int
        List of channels for the successive downblocks

    last_conv : int
        Number of output convolution layers

    """
    def __init__(self, input_shape, latent_dim, channels, block_size, last_conv, optimizer=None):
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.channels = channels
        self.block_size = block_size
        self.last_conv = last_conv
        self.optimizer = optimizer if optimizer else tfk.optimizers.Adam 
        
        self._setup_model()
        

    def _create_model(self):
        img_input = tfk.layers.Input(shape=self.input_shape, name='image')
        label_input = tfk.layers.Input(shape=self.input_shape, name= 'label')
        mode = tfk.layers.Input(shape=(), batch_size=1, name='mode')
        
        latent_space_prediction = image_encoder(self.input_shape, self.channels, self.block_size, self.latent_dim)(img_input)
        latent_space_training = image_with_label_encoder(self.input_shape, self.channels, self.block_size, self.latent_dim)([img_input, label_input])
        unet_features = unet(self.input_shape, self.channels, self.block_size)(img_input)
        model_combiner = combinatory(self.latent_dim, self.input_shape, self.last_conv)
        
        # is_training = tf.cast(0, dtype="float32")
        prediction = tfk.backend.switch(
            mode == np.array(0., dtype=np.float32), 
            model_combiner([unet_features, latent_space_training]),
            model_combiner([unet_features, latent_space_prediction]))

        return tfk.Model(
            inputs=[img_input, label_input, mode], 
            outputs=[prediction, latent_space_prediction, latent_space_training], 
            name='Proba_Unet')

    def _setup_model(self):
        self.unet = unet(self.input_shape, self.channels, self.block_size)
        self.img_encoder = image_encoder(self.input_shape, self.channels, self.block_size, self.latent_dim)
        self.img_label_encoder = image_with_label_encoder(self.input_shape, self.channels, self.block_size, self.latent_dim)
        self.model_combiner = combinatory(self.latent_dim, self.input_shape, self.last_conv)
        self.training_model = self._init_training_model()
        self.prediction_model = self._init_prediction_model()

    def _init_training_model(self):
        img_input = tfk.layers.Input(shape=self.input_shape, name='image')
        label_input = tfk.layers.Input(shape=self.input_shape, name= 'label')

        seg_map = self.model_combiner([
            self.unet(img_input), 
            self.img_with_label_encoder([img_input, label_input])])

        img_encoder_output = self.img_encoder([img_input])
        img_with_label_encoder_output = self.img_encoder([img_input])

        return tfk.Model(
            inputs=[img_input, label_input],
            outputs=[seg_map, img_encoder_output, img_with_label_encoder_output],
            name="ProbaUNet_training")

    def _init_prediction_model(self):
        img_input = tfk.layers.Input(shape=self.input_shape, name='image')
        output = self.model_combiner([
            self.unet(img_input), 
            self.img_encoder(img_input)])

        return tfk.Model(img_input, output, name='ProbaUnet_prediction')

    def eval_step(self, features, label, beta):

        mode = tf.constant([1.])
        punet_output = self.model([features, label, mode])
        return cross_entropy(label, punet_output, beta)

    def train_step(self, features, label, lr, beta):

        with tf.GradientTape() as tape:
            mode = tf.constant([0.])
            punet_output = self.model([features, label, mode])
            total_loss, rec_loss, kl_loss = cross_entropy(label, punet_output, beta)
            grads = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer = tfk.optimizers.Adam(learning_rate=lr)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return [total_loss, rec_loss, kl_loss]

    def load_weights(self, checkpoint_path):
        checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        self.model.load_weights(checkpoint)
        print(f'Weights loaded on the model from {checkpoint}')

    def plot_prediction(self, features, labels, nb_to_plot, threshold1=0.9, threshold2=1.5):
        import matplotlib.pyplot as plt
        predictions = self.model.predict(features, nb_realisation, nb_batch)
        fig, ax = plt.subplots(nb_to_plot, 5, figsize=(nb_to_plot, nb_to_plot/5 * 5))
        for i in range(nb_to_plot):

            thresholded = np.where(np.mean(predictions[i], axis=0) > threshold1, 1, 0)
            thresholded = np.where(np.mean(predictions[i], axis=0) > threshold2, 2, thresholded)
            ax[i, 0].imshow(features[i])
            ax[i, 1].imshow(labels[i])
            ax[i, 0].imshow(np.mean(predictions[i], axis=0))
            ax[i, 0].imshow(np.var(predictions[i], axis=0))
            ax[i, 0].imshow(thresholded)

    def save_weights(self):
        pass
