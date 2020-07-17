import tensorflow as tf
import tensorflow_probability as tfp

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
        
        batch_size = tf.cast(tf.shape(label)[0], tf.float32)
        flat_labels = tf.cast(tf.reshape(label, [-1]), dtype=tf.int32)
        flat_labels = tf.one_hot(indices=flat_labels, depth=3, axis=-1)
        flat_labels = tf.stop_gradient(flat_labels)
        flat_logits = tf.reshape(prediction[0], [-1, 3])
        mask = np
        rec_loss = tf.nn.softmax_cross_entropy_with_logits(labels=flat_labels, logits=flat_logits)
        rec_loss = tf.reduce_sum(rec_loss) / batch_size

        z_mean_img, z_sigma_img = prediction[1][:, :latent_dim], prediction[1][:, latent_dim:] 
        z_mean_seg, z_sigma_seg = prediction[2][:, :latent_dim], prediction[2][:, latent_dim:]
        gaussian_img = tfd.MultivariateNormalDiag(loc=z_mean_img, scale_diag=tf.exp(z_sigma_img))
        gaussian_seg = tfd.MultivariateNormalDiag(loc=z_mean_seg, scale_diag=tf.exp(z_sigma_seg))
        kl_loss = tf.reduce_mean(tfd.kl_divergence(gaussian_seg, gaussian_img))
        
        total_loss = (rec_loss + beta * kl_loss)

        return [total_loss, rec_loss, kl_loss]