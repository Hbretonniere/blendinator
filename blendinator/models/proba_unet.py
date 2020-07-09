import tensorflow.keras as tfk
import tensorflow as tf
import matplotlib.pyplot as plt
from blendinator.models.combinatory import combinatory
from blendinator.models.unet import unet
from blendinator.models.encoders import image_only_gaussian_encoder, image_gt_gaussian_encoder

class ProbaUNet:
    """ProbaUNet
    
    Attributes
    ----------
    block_size : int
        Number of convolution layers in each block
    last_conv : int
        Number of output convolution layers
    channels : lsit of int
        List of channels for the successive downblocks

    """
    def __init__(self, input_shape, latent_dim, channels, block_size, last_conv):
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.channels = channels
        self.block_size = block_size
        self.last_conv = last_conv
        self.opt = None
        self.loss = None
        self._setup_model()
        self.model = self._create_model()

    def _setup_model(self):
        self.unet = unet(self.input_shape, self.channels, self.block_size)
        self.img_only_encoder = image_only_gaussian_encoder(self.input_shape, self.channels, self.block_size, self.latent_dim)
        self.img_gt_encoder = image_gt_gaussian_encoder(self.input_shape, self.channels, self.block_size, self.latent_dim)
        self.combinatory = combinatory(self.latent_dim, self.input_shape, self.last_conv)

    def _create_model(self):
        img_input = tfk.layers.Input(shape=self.input_shape, name='image')
        gt_input = tfk.layers.Input(shape=self.input_shape, name= 'ground_truth')
        mode = tfk.layers.Input(shape=(), batch_size=1, name='mode')
        
        ls_img = self.img_only_encoder(img_input)
        ls_seg = self.img_gt_encoder([img_input, gt_input])
        unet_features = self.unet(img_input)
        
        is_training = tf.cast(0, dtype="float32")
        prediction = tfk.backend.switch(mode==is_training.numpy(), self.combinatory([unet_features, ls_seg]),
            self.combinatory([unet_features, ls_img]))

        model = tfk.Model(inputs=[img_input, gt_input, mode], outputs=[prediction, ls_img, ls_seg], name='Proba_Unet')

        return(model)


    def loss(self, label, prediction, beta):
        
        batch_size = tf.cast(tf.shape(label)[0], tf.float32)
        flat_labels = tf.cast(tf.reshape(label, [-1]), dtype=tf.int32)
        flat_labels = tf.one_hot(indices=flat_labels, depth=3, axis=-1)
        flat_labels = tf.stop_gradient(flat_labels)
        flat_logits = tf.reshape(prediction[0], [-1, 3])
        rec_loss = tf.nn.softmax_cross_entropy_with_logits(labels=flat_labels, logits=flat_logits)
        rec_loss = tf.reduce_sum(rec_loss) / batch_size

        z_mean_img, z_sigma_img = prediction[1][:, :latent_dim], prediction[1][:, latent_dim:] 
        z_mean_seg, z_sigma_seg = prediction[2][:, :latent_dim], prediction[2][:, latent_dim:]
        gaussian_img = tfd.MultivariateNormalDiag(loc=z_mean_img, scale_diag=tf.exp(z_sigma_img))
        gaussian_seg = tfd.MultivariateNormalDiag(loc=z_mean_seg, scale_diag=tf.exp(z_sigma_seg))
        kl_loss = tf.reduce_mean(tfd.kl_divergence(gaussian_seg, gaussian_img))
        
        total_loss = (rec_loss + beta * kl_loss)

        return [total_loss, rec_loss, kl_loss]

    def eval_step(self, features, label, beta):

        mode = tf.constant([1.])
        punet_output = self.model([features, label, mode])
        return loss(label, punet_output, beta)

    def train_step(self, features, label, lr, beta):

        with tf.GradientTape() as tape:
            mode = tf.constant([0.])
            punet_output = self.model([features, label, mode])
            total_loss, rec_loss, kl_loss = loss(label, punet_output, beta)
            grads = tape.gradient(total_loss, self.model.trainable_variables)
        optimizer = tfk.optimizers.Adam(learning_rate=lr)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return [total_loss, rec_loss, kl_loss]
    
    def train(self, train_data, epochs, step_per_epoch, lrs, betas):

        with tf.device('GPU:0'):
            with tqdm(total=nb_epochs, desc='Epcoh', position=0) as bar1:
                for epoch in range(epochs):
                    bar1.update(1)
                    with tqdm(total=step_per_epoch, desc='batch', position=1) as bar2:
                        for batch_nb, (features, gt) in enumerate(train_data):
                            bar2.update(1)

                            total_loss, rec_loss, kl_loss = train_step(features, gt, lrs[nb_prev_epochs+epoch], beta[nb_prev_epochs+epoch])
                            train_loss_results.append(total_loss)
                            train_rec_loss_results.append(rec_loss)
                            train_kl_loss_results.append(kl_loss)

                            if k % eval_every_n_step == 0:
                                val_tot, val_rec, val_kl = 0, 0, 0
                                for valid,(features, gt) in enumerate(eval_data):
                                    total_loss, rec_loss, kl_loss = eval_step(features, gt)
                                    val_tot += total_loss
                                    val_rec += rec_loss
                                    val_kl += kl_loss

                                eval_loss_results.append(val_tot/valid)
                                eval_rec_loss_results.append(val_rec/valid)
                                eval_kl_loss_results.append(val_kl/valid)
                            bar2.set_description('PUnet, loss=%g' % train_loss_results[-1])
                            k+=1
                        if epoch % 2 == 0:
                            proba_unet.save_weights(checkpoint_path + '/checkpoint_'+ str(len(train_loss_results)))



    def predict(self, features, nb_realisation, nb_batch):

        mode = tf.constant([1.])
        batch_size = int(features[:, 0, :, :].shape[0]/nb_batch)
        predictions = np.zeros((nb_batch, batch_size, nb_realisation, self.input_shape[0], self.input_shape[1]))
        for batch in range(nb_batch):
            for realisation in range(nb_realisation):
                sample = self.model([features[int(batch_size*batch):int(batch_size*(batch+1))], features[int(batch_size*batch):int(batch_size*(batch+1))], mode])[0]
                prediction[batch, :, realisation, :, :] = np.argmax(sample, axis=-1)
        return np.reshape(predictions, (nb_batch*batch_size, nb_realisation, self.input_shape[0], self.input_shape[1]))
    
    def load_weights(self, checkpoint_path):

        checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        self.model.load_weights(checkpoint)
        print(f'Weights loaded on the model from {checkpoint}')

    def plot_prediction(self, features, labels, nb_to_plot, threshold1=0.9, threshold2=1.5):
        
        predictions = self.model.predict(features, nb_realisation, nb_batch)
        fig, ax = plt.subplots(nb_to_plot, 5, figsize=(nb_to_plot, nb_to_plot/5 * 5)
        for i in range(nb_to_plot):

            thresholded = np.where(np.mean(predictions[i], axis=0) > threshold1, 1, 0)
            thresholded = np.where(np.mean(predictions[i], axis=0) > threshold2, 2, thresholded)
            ax[i, 0].imshow(features[i])
            ax[i, 1].imshow(labels[i])
            ax[i, 0].imshow(np.mean(predictions[i], axis=0))
            ax[i, 0].imshow(np.var(predictions[i], axis=0))
            ax[i, 0].imshow(thresholded)


    # def save_weights(self):
    #     return self.model.weights
    
    # def compile(self, optimiser, loss):
    #     self.opt = opt
    #     self.loss = lsos
        
    # def build(self):
    #     pass

    # def train(self, images, labels, epochs=50):
    #     if self.opt is None:
    #         print("You should compile the model first")
    #         return
        
    #     for epoch in epochs:
    #         ...

    # def predict(self, images):
    #     pass
    
