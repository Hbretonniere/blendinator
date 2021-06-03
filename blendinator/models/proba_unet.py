import numpy as np
import tensorflow as tf
from tensorflow.train import latest_checkpoint
import tensorflow.keras as tfk
import sys
# from tqdm import tqdm
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from blendinator.models.combinatory import combinatory
from blendinator.models.unet import unet
from blendinator.models.encoders import image_encoder, image_with_label_encoder
from blendinator.loss import cross_entropy, weighted_cross_entropy
import json
import os

class ProbaUNet:
    """ProbaUNet

    Attributes
    ----------
    input_shape : int
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
    def __init__(self, input_shape, latent_dim, channels, block_size, last_conv, training_path, batch_size=32, loss='None', device='GPU', optimizer=None, history=None):

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.channels = channels
        self.block_size = block_size
        self.last_conv = last_conv
        self.optimizer = optimizer if optimizer else tfk.optimizers.Adam
        self.device = device
        self._setup_model()
        self.loss = loss
        self.hparams = {'input_shape':input_shape,
                        'latent_dim':latent_dim,
                        'channels':channels,
                        'block_size':block_size,
                        'last_conv':last_conv,
                        'optimizer':str(self.optimizer),
                        'batch_size':batch_size,
                        'loss':loss}
        self.training_path = training_path
        if os.path.isfile(self.training_path+'losses.npy'):
            print('already trained model')
            self.history = np.load(self.training_path+'losses.npy')
            print(type(self.history[0]))
            self.history[0] = self.history[0].tolist()
            self.history[1] = self.history[1].tolist()
            self.history[2] = self.history[2].tolist()
            self.hparams['trained_step'] = len(self.history[0])
            print(type(self.history[0]))
        else:
            print('new training')
            self.history = [[], [], []]
            self.hparams['trained_step'] = 0
        

    def _setup_model(self):

        """ Create the different sub-models of the Proba Unet ;
        - the unet, which ouput a 3 classes temporary segmap
        - the img_encoder, which output the parameters (mu, sigma) of a latent_dim dimensional gaussian
             from the encoder of an image
        - the img_with_label_encoder, which output the parameters (mu, sigma) of a latent_dim dimensional
            gaussian from a concatenation of the input image and the corresponding segmentation map (label)
        - the last_CNN, which convolves (last_conv times) a concatenation of the output of the unet
            and one of the two encoder to produce the final seg map
        The two others model are just a combination of the others, but do not have any specific weights.
        It's the ProbaUnet in training mode (sampling from the img_encoder_with_labels) and
         the ProbaUnet in prediction mode (sampling from the img_encoder).
         Updating one or the other just update the previously explained models."""

        self.unet = unet(self.input_shape, self.channels, self.block_size)
        self.img_encoder = image_encoder(self.input_shape, self.channels, self.block_size, self.latent_dim)
        self.img_with_label_encoder = image_with_label_encoder(self.input_shape, self.channels, self.block_size, self.latent_dim)
        self.last_CNN = combinatory(self.latent_dim, self.input_shape, self.last_conv)
        self.training_model = self._init_training_model()
        self.prediction_model = self._init_prediction_model()

    def _init_training_model(self):
        """ Create the Proba Unet used for training, using the other sub models """

        img_input = tfk.layers.Input(shape=self.input_shape, name='image')
        label_input = tfk.layers.Input(shape=self.input_shape, name='label')

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
        return cross_entropy(labels, punet_output, beta)

    def train_step(self, features, labels, lr, beta):
        """ Run the Punet in training mode (training_model), compute the loss and update the weights of all the submodels for one step. 
            Return the total loss, the reconstruction loss and the KL. """

        with tf.GradientTape() as tape:
            punet_output = self.training_model([features, labels])
            if self.loss == 'weighted':
                total_loss, rec_loss, kl_loss = weighted_cross_entropy(self.latent_dim, labels, punet_output, beta)
            else:
                total_loss, rec_loss, kl_loss = cross_entropy(self.latent_dim, labels, punet_output, beta)
            grads = tape.gradient(total_loss, self.training_model.trainable_variables)
        self.optimizer(lr).apply_gradients(zip(grads, self.training_model.trainable_variables))
        return [total_loss, rec_loss, kl_loss]

    def train(self, train_data, epochs, step_per_epoch,
              lrs, betas, save_frequency=1):
        
        with tf.device(self.device):
            with tqdm(total=epochs, desc='Epoch', position=0) as bar1:
                for epoch in range(epochs):
                    print('epoch Number : ', epoch)
                    bar1.update(1)
                    with tqdm(total=step_per_epoch, desc='batch', position=1) as bar2:
                        for batch_nb, (image, label) in enumerate(train_data):
                            if batch_nb % 200 == 0:
                                print(batch_nb)
                            bar2.update(1)
                            total_loss, rec_loss, kl_loss = self.train_step(image, label, lrs[epoch], betas[epoch])
                            self.history[0].append(total_loss)
                            self.history[1].append(rec_loss)
                            self.history[2].append(kl_loss)

                            # if k % eval_every_n_step == 0:
                            #     val_tot, val_rec, val_kl = 0, 0, 0
                            #     for valid,(image, label) in enumerate(eval_data):
                            #         total_loss, rec_loss, kl_loss = PUNet.eval_step(image, label)
                            #         val_tot += total_loss
                            #         val_rec += rec_loss
                            #         val_kl += kl_loss

                            #     eval_loss_results.append(val_tot/valid)
                            #     eval_rec_loss_results.append(val_rec/valid)
                            #     eval_kl_loss_results.append(val_kl/valid)
                            bar2.set_description(f"PUnet, loss={self.history[0][-1]}")
                            # k+=1
                    if epoch % save_frequency == 0:
                        self.save_weights(epoch)
        
        ''' Save the loss '''
        np.save(self.training_path+'losses.npy', self.history)
        
        ''' Update the training history infos'''
        self.hparams['training_data_size'] = step_per_epoch * self.hparams['batch_size']
        self.hparams['trained_step'] += epochs*step_per_epoch

        with open(self.training_path+'hparams.txt', 'w') as file:
             file.write(json.dumps(self.hparams)) # use `json.loads` to do the reverse

        return self.history

#     def train_and_plot(self, train_data, plot_data, epochs, step_per_epoch,
#                        lrs, betas, history,
#                        plot_steps, nb_to_plot=10, save_frequency=1):
#         with tf.device(self.device):
#             with tqdm(total=epochs, desc='Epoch', position=0) as bar1:
#                 for epoch in range(epochs):
#                     bar1.update(1)
#                     with tqdm(total=step_per_epoch, desc='batch', position=1) as bar2:
#                         for batch_nb, (image, label) in enumerate(train_data):
#                             bar2.update(1)
#                             total_loss, rec_loss, kl_loss = self.train_step(image, label, lrs[epoch], betas[epoch])
#                             history[0].append(total_loss)
#                             history[1].append(rec_loss)
#                             history[2].append(kl_loss)
#                             bar2.set_description(f"PUnet, loss={history[0][-1]}")
#                             # print(image.shape)
#                             if (epoch*step_per_epoch)+batch_nb in plot_steps:
#                                 predictions = self.prediction_model([plot_data[0][:nb_to_plot]])[1]
#                                 # print('predicitons : ', predictions.shape)
#                                 fig, ax = plt.subplots(nb_to_plot, 3, figsize=(nb_to_plot/3*2, nb_to_plot * 2))
#                                 ax[0, 0].set_title('Input Image')
#                                 ax[0, 1].set_title('Input Gt')
#                                 ax[0, 2].set_title('Predicted segmap')
#                                 for i in range(nb_to_plot):
#                                     # try:
#                                         segmap = np.argmax(predictions[i], axis=-1)
#                                         ax[i, 0].imshow(plot_data[0][:nb_to_plot][i])
#                                         ax[i, 1].imshow(plot_data[1][:nb_to_plot][i])
#                                         ax[i, 2].imshow(segmap)
#                                     # except Exception:
#                                         # print('pb')
#                                 plt.savefig(f'./training_plots/training_step{epoch*step_per_epoch+batch_nb}')

#                     if epoch % save_frequency == 0:
#                         self.save_weights(epoch, self.training_path)
#         return(history)

    def save_weights(self, epoch):
        """ Save the weights of all the submodels, with a name corresponding to the current training step """

        self.unet.save_weights(f'{self.training_path}unet_weights/epoch_{epoch+1}')
        self.img_encoder.save_weights(f'{self.training_path}img_encoder_weights/epoch_{epoch+1}')
        self.img_with_label_encoder.save_weights(f'{self.training_path}img_with_label_encoder_weights/epoch_{epoch+1}')
        self.last_CNN.save_weights(f'{self.training_path}last_CNN_weights/epoch_{epoch+1}')
        return()

    def load_weights(self):
        """ Load the weights into all the submodels of the PUnet, according to the epoch written in the `checkpoint` file inside the checkpoint_path"""
        self.unet.load_weights(latest_checkpoint(f'{self.training_path}unet_weights/'))
        self.img_encoder.load_weights(latest_checkpoint(f'{self.training_path}img_encoder_weights/'))
        self.img_with_label_encoder.load_weights(latest_checkpoint(f'{self.training_path}img_with_label_encoder_weights/'))
        self.last_CNN.load_weights(latest_checkpoint(f'{self.training_path}last_CNN_weights/'))
        return()



    def print_models(self, path, show_shape=True):

        tfk.utils.plot_model(self.unet, to_file=path+'unet.png', show_shapes=True)
        tfk.utils.plot_model(self.img_encoder, to_file=path+'img_encoder.png', show_shapes=True)
        tfk.utils.plot_model(self.img_with_label_encoder, to_file=path+'img_with_label_encoder.png', show_shapes=True)
        tfk.utils.plot_model(self.last_CNN, to_file=path+'last_CNN.png', show_shapes=True)
        tfk.utils.plot_model(self.training_model, to_file=path+'training_model.png', show_shapes=True)
        tfk.utils.plot_model(self.prediction_model, to_file=path+'prediction_model.png', show_shapes=True)
        return 0
