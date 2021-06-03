import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tfd = tfp.distributions


def cross_entropy(latent_dim, label, prediction, beta):
        
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

def weighted_cross_entropy(latent_dim, label, prediction, beta):
        ''' 
        Weighted cross entropy loss + beta KL divergence 
        Does the softmax cross entropy between 2 matrices : the labels and the prediction.

        label : tensor [?, 64, 64, 1] : True classes of the pixels.  
        prediction : tensor [3, ...] : the output of the PUNEt :
                      prediction[0] is [?, 64, 64, 3], the predicted classes, continuous (softmax)
                      prediction[1] and [2] are the output of the prob part.
        beta : int, the weight of the KL loss
        
        The softmax cross entropy works between matrices of shape (?*64*64, 3). 
        To do so, we need to one hot encode the labels, and the flatten.
        For the logits, we need to just flatten the batch, image dimensions.
        
        '''
        batch_size = tf.cast(tf.shape(label)[0], tf.float32)
        weights = np.zeros_like(prediction[0])
        weights[0] +=  0.2
        weights[1] += 0.8
        weights[2]+= 0.9
        weights = tf.convert_to_tensor(weights.astype('float32'))
        flat_weights = tf.reshape(weights, [-1, 3])
        
        flat_labels = tf.cast(tf.reshape(label, [-1]), dtype=tf.int32)
        flat_labels = tf.one_hot(indices=flat_labels, depth=3, axis=-1) * flat_weights
        flat_labels = tf.stop_gradient(flat_labels)
        flat_logits = tf.reshape(prediction[0], [-1, 3]) * flat_weights # 0 is the 4 classes array, 1 and 2 are the z and sigmas of img and seg
        
        rec_loss = tf.nn.softmax_cross_entropy_with_logits(labels=flat_labels, logits=flat_logits) 
        rec_loss = tf.reduce_sum(rec_loss) / batch_size

        z_mean_img, z_sigma_img = prediction[1][:, :latent_dim], prediction[1][:, latent_dim:] 
        z_mean_seg, z_sigma_seg = prediction[2][:, :latent_dim], prediction[2][:, latent_dim:]
        gaussian_img = tfd.MultivariateNormalDiag(loc=z_mean_img, scale_diag=tf.exp(z_sigma_img))
        gaussian_seg = tfd.MultivariateNormalDiag(loc=z_mean_seg, scale_diag=tf.exp(z_sigma_seg))
        kl_loss = tf.reduce_mean(tfd.kl_divergence(gaussian_seg, gaussian_img))
        
        total_loss = (rec_loss + beta * kl_loss)

        return [total_loss, rec_loss, kl_loss]