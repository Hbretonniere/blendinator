# %%
# import the necessary packages

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow import keras
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Input, concatenate, Reshape
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import he_normal
from tensorflow.compat.v1 import truncated_normal_initializer
import tensorflow_probability as tfp
from tqdm import trange
tfd = tfp.distributions
from tqdm import tqdm
# tf.keras.backend.set_floatx('float32')


def asinh_norm(x, y):
    return tf.asinh(tf.divide(x, tf.math.reduce_max(x))), y


""" Training Parameters """
nb_epochs = 50
batch_size = 32
lrs = 1e-3 * np.exp(-0.1 * np.arange(300))
eval_every_n_step = 200

""" Import and preprocess data (for me, segs = ground truth)"""
data_path = '/data57/hbretonniere/FVAE/FVAE_emulated_data/for_deblender/phymasks/'
name = 'TU_FVAE_divided_1_3_re_field_0123.npy'  # there is 148996 stamps
checkpoint_path = "/data57/hbretonniere/deblending/tf2/check_test/"
print("checkpoint path : ", checkpoint_path)

""" Import data """
imgs = np.expand_dims(np.load(data_path + 'stamps_WithNoise_' + name), axis=-1).astype('float32')
segs = np.expand_dims(np.load(data_path + 'stamps_seg_' + name), axis=-1)

train_slice = 0.85 * imgs.shape[0]
eval_slice = 1 * imgs.shape[0]
train_steps_per_epoch = int(train_slice / batch_size)

train_data = tf.data.Dataset.from_tensor_slices((imgs[:train_slice], segs[:train_slice])).shuffle(100000, reshuffle_each_iteration=True).map(asinh_norm).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
eval_data = tf.data.Dataset.from_tensor_slices((imgs[train_slice:eval_slice], segs[train_slice:eval_slice])).shuffle(100000, reshuffle_each_iteration=True).map(asinh_norm).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)



"""Achitecture hyper paremeters """
base_channels = 32
list_num_channels = [base_channels, 2*base_channels, 4*base_channels,
                6*base_channels, 6*base_channels, 6*base_channels, 6*base_channels]    # List of the filters to do both in the Unet and the proba part
num_conv_per_block = 1
nb_conv_1x1 = 1  # num of conv to make after the UNet/proba concatenation
stamp_size = 128 # input shape (if lower, we may reduce the unumber of filters in the list_num_channels)
latent_dim = 10 



"""
### Image and gt+image encoders
Here are define two sub-models corresponding to the probailistic part of the Unet. The first encode the image, the second encode a concatenation of the image and the ground truth seg.
Inputs : 
- Tensor((None, 128,128,1), dtype=float32)) for the first one
- [Tensor((None, 128,128,1), dtype=float32)), Tensor((None, 128,128,1), dtype=float32))] for the second one

they both output a code vector, of dimension [2*latent_dim], which will be interpreted as [mu, log_sigma]. I think it's better for now since it's a bit complicated to pass a distribution in input of a submodel. The sampling is made on the combinatory model.
"""

''' Image encoder'''
img_encoder_input = Input(shape=(stamp_size, stamp_size, 1), dtype='float32', name='Image')
features = img_encoder_input
for i, n_channels in enumerate(list_num_channels):
    down_sample = True
    if i == 0:
        down_sample = False
    features = downblock(features,
                          n_channels,
                          num_convs=num_conv_per_block,
                          down_sample_input=down_sample,
                          name='img_down_block_'+str(i))
features = tf.reduce_mean(features, axis=[1,2], keepdims=True)
mu_log_sigma = Conv2D(2*latent_dim, (1,1), 1, 'SAME', kernel_initializer=he_normal, bias_initializer=truncated_normal_initializer(stddev=0.001),
                          kernel_regularizer=tf.keras.regularizers.l2(1.0), bias_regularizer=tf.keras.regularizers.l2(1.0),
                          activation='relu')(features)
mu_log_sigma = tf.squeeze(mu_log_sigma, axis=[1,2])
#     mu = mu_log_sigma[:, :latend_dim]
#     log_sigma = mu_log_sigma[:, latent_dim:]
img_encoder_output = mu_log_sigma  #[mu, log_sigma]
img_encoder = tf.keras.Model(img_encoder_input, img_encoder_output, name='img_encoder')
img_encoder.summary()

''' GT encoder'''
gt_input = Input(shape=(stamp_size, stamp_size, 1), name='Ground_Truth')#, dtype='float64')
# gt = tf.cast(gt_input, tf.int32)
img_input = Input(shape=(stamp_size, stamp_size, 1), name='Image')#, dtype='float64')
spatial_shape = img_input.get_shape()[-3:-1]

# gt = tf.reshape(gt, shape=[-1])
# gt = tf.one_hot(indices = gt, depth=3, axis = 3)
# gt = gt_input - 0.5

features = tf.concat([img_input, gt_input], axis=-1)
# features = gt

for i, n_channels in enumerate(list_num_channels):
    down_sample = True
    if i == 0:
        down_sample = False
    features = downblock(features,
                          n_channels,
                          num_convs=num_conv_per_block,
                          down_sample_input=down_sample,
                          name='img_down_block_'+str(i))
features = tf.reduce_mean(features, axis=[1,2], keepdims=True)
mu_log_sigma = Conv2D(2*latent_dim, (1,1), 1, 'SAME', kernel_initializer=he_normal, bias_initializer=truncated_normal_initializer(stddev=0.001),
                          kernel_regularizer=tf.keras.regularizers.l2(1.0), bias_regularizer=tf.keras.regularizers.l2(1.0),
                          activation='relu')(features)
mu_log_sigma = tf.squeeze(mu_log_sigma, axis=[1,2])
gt_encoder_output = mu_log_sigma  #[mu, log_sigma]
gt_encoder = tf.keras.Model(inputs=[img_input, gt_input], outputs=[gt_encoder_output], name='gt_encoder')
# gt_encoder = tf.keras.Model(inputs=[gt_input], outputs=[gt_encoder_output], name='gt_encoder')
gt_encoder.summary()


"""
### Combinatory
Here is the last submodel, which sample from a Multivariate distribution parameterized by the output of the previous gaussian encoder, concatenate it to the unet features and convolves to produce the final ont hot encoded output.

The inputs are [Tensor((None, 128, 128, n), dtype=float32), Tensor((None, 2*altent_dim), dtype=float32)]

The output is a Tensor((None, 128, 128, 3), dtype=float32))
"""

z_input = Input(shape=img_encoder_output.shape[1:], name='Latent_space')
features_input = Input(shape=unet_output.shape[1:], name='U-net_Features')

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
    sample = Reshape((1, 1, latent_dim))(sample)
broadcast_sample = tf.tile(sample, multiples)

features = tf.concat([features_input, broadcast_sample], axis =-1)
# features = features_input
for i in range(nb_conv_1x1):
    features = Conv2D(3, (1,1), 1, 'SAME',
                           kernel_initializer=he_normal,
                      bias_initializer=truncated_normal_initializer(stddev=0.001),
                          kernel_regularizer=tf.keras.regularizers.l2(1.0), bias_regularizer=tf.keras.regularizers.l2(1.0),
                           activation='relu')(features)
# combinatory_output = features
combinatory = tf.keras.Model(inputs=[features_input, z_input], outputs=[features], name='Combinatory')
combinatory.summary()

"""
## Probabilistic U-net
Finaly, we create the global model, which englobes all the submodels : 
The input are the image and the GT: [Tensor((None, 128, 128, 1), dtype=float32, Tensor((None, 128, 128, 1), dtype=float32)]

The outputs are :
- A 3 dimension image (corresponding to the three possible class, it s a probabilistic one hot like output :  [Tensor((None, 128, 128, 3), dtype=float32]
- The parameters of the gaussian distribution from the image only : Tensor((None, 2* latent_dim), dtype=float32]
- The parameters of the gaussian distribution from the image + GT : Tensor((None, 2* latent_dim), dtype=float32)]
"""

""" Punet """
img_input = Input(shape=(128, 128, 1), name='image')#, dtype='float64')
gt_input = Input(shape=(128, 128, 1), name= 'ground_truth')#, dtype='float64')
mode = Input(shape=(), batch_size=1, name='mode')
ls_img = img_encoder(img_input)
ls_seg = gt_encoder([img_input, gt_input])
unet_features = unet(img_input)
is_training = tf.cast(0, dtype="float32")
print(mode)
prediction = K.switch(mode==is_training.numpy(), combinatory([unet_features, ls_seg]),
                      combinatory([unet_features, ls_img]))

# Verify if the switch works
# was_training = K.switch(mode==is_training.numpy(), tf.constant('is_training', shape=(1,)), tf.constant('is_not_training', shape=(1,)))
# proba_unet = keras.Model(inputs=[img_input, gt_input, mode], outputs=[prediction, ls_img, ls_seg, was_training], name='Proba_Unet')

proba_unet = keras.Model(inputs=[img_input, gt_input, mode], outputs=[prediction, ls_img, ls_seg], name='Proba_Unet')
proba_unet.summary()


""" LOSS
 The loss contain two terms : the reconstruction loss (rec_loss) anf the kl.
The rec is made by flattening the labels (gt) and the logits (prediction), and apply a soft max cross entropy to see the error
the KL is using the tensorf flow probability KL distribution between the two multivariate normal diag outputed by the punet
"""

def loss_fn(y_true, y_predict):
    batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
    flat_labels = tf.cast(tf.reshape(y_true, [-1]), dtype=tf.int32)
    flat_labels = tf.one_hot(indices=flat_labels, depth=3, axis=-1)
#     flat_labels = tf.reshape(flat_labels[0], [-1, 3])
    flat_labels = tf.stop_gradient(flat_labels)
    flat_logits = tf.reshape(y_predict[0], [-1, 3])
    rec_loss = tf.nn.softmax_cross_entropy_with_logits(labels=flat_labels, logits=flat_logits)
    rec_loss = tf.reduce_sum(rec_loss) / batch_size

    z_mean_img, z_sigma_img = y_predict[1][:, :latent_dim], y_predict[1][:, latent_dim:] 
    z_mean_seg, z_sigma_seg = y_predict[2][:, :latent_dim], y_predict[2][:, latent_dim:]
    gaussian_img = tfd.MultivariateNormalDiag(loc=z_mean_img, scale_diag=tf.exp(z_sigma_img))
    gaussian_seg = tfd.MultivariateNormalDiag(loc=z_mean_seg, scale_diag=tf.exp(z_sigma_seg))
    kl_loss = tf.reduce_mean(tfd.kl_divergence(gaussian_seg, gaussian_img))
#     kl_loss =keras.losses.KLDivergence() 
    
    total_loss = (rec_loss + beta * kl_loss)

    return [total_loss, rec_loss, kl_loss] 

# A train step is defined by running the Punet with a batch with the "training behavior" (sampling from the gt+img distrib), computing the losses, applying the gradient and updating the weights 
@tf.function
def train_step(x, y, lr):
    with tf.GradientTape() as tape:
        mode = tf.constant([0.])
        prediction = proba_unet([x, y, mode])
        total_loss, rec_loss, kl_loss = loss_fn(y, prediction, lr)
        grads = tape.gradient(total_loss, proba_unet.trainable_variables)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    optimizer.apply_gradients(zip(grads, proba_unet.trainable_variables))
#     return [total_loss, rec_loss, kl_loss, prediction[3]]
    return [total_loss, rec_loss, kl_loss]

# An eval step is defined by running the Punet with a batch with the "Testing behavior" (sampling from the img only distrib) and computing the losses 
def eval_step(x, y):
    mode = tf.constant([1.])
    prediction = proba_unet([x, y, mode])
    total_loss, rec_loss, kl_loss = loss_fn(y, prediction)    
#     return [total_loss, rec_loss, kl_loss, prediction[3]]
    return [total_loss, rec_loss, kl_loss]

# A test step is defined by running the Punet with a batch with the "Testing behavior" (sampling from the img only distrib)
def test_step(x,y):
    mode = tf.constant([1.])
    prediction = proba_unet([x, y, mode])
    return prediction

beta = 2

""" Here I load the losses and the checkpoint if there is some (I should definitly use keras pre made call backs, my training loop is kinda ugly... But it works..."""
try :
    with open(checkpoint_path+'/losses.p', "rb") as fp:
        losses = pickle.load(fp)
    train_loss_results, train_rec_loss_results, train_kl_loss_results = losses[0]
    eval_loss_results, eval_rec_loss_results, eval_kl_loss_results = losses[1]
    nb_prev_epochs = losses[2]

    already_trained = True
    print(f"There are saved weights from a training of {len(train_loss_results)} steps, loading them and restoring the losses history")
except :
    already_trained = False
    train_loss_results, train_rec_loss_results, train_kl_loss_results = [], [], []
    eval_loss_results, eval_rec_loss_results, eval_kl_loss_results = [], [], []
    print("No weights found, training from scratch")

if already_trained:
    k = len(train_loss_results)
    latest = tf.train.latest_checkpoint(checkpoint_path)
    proba_unet.load_weights(latest)
else:
    k = 1


""" Training Loop 
I do a manual training loop beacause of the switch behavior between the train and evaluate. If you don t do that, .fit should work...
I manually save every training and evaluate loss, one again this is not good... But works.
I use tqdm to show the advance of the training, and the loss value at each batch


""" 
k = 1
with tf.device("GPU:0"):
    with tqdm(total = nb_epochs, desc='Epoch', position=0) as bar1:
        for epoch in range(nb_epochs):
            bar1.update(1)
            with tqdm(total = train_steps_per_epoch, desc='batch', position=1) as bar2:
                for batch_nb, (features, gt) in enumerate(train_data):
                    bar2.update(1)

                    total_loss, rec_loss, kl_loss = train_step(features, gt, lrs[nb_prev_epochs+epoch])
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

proba_unet.save_weights(checkpoint_path + '/checkpoint_'+ str(len(train_loss_results)))
with open(eval_every_n_step + '/losses.p', "wb") as fp:
    pickle.dump([[train_loss_results, train_rec_loss_results, train_kl_loss_results],[eval_loss_results, eval_rec_loss_results, eval_kl_loss_results], nb_prev_epochs + nb_epochs], fp)

eval_range = np.arange(eval_every_n_step, (len(eval_loss_results)+1)*eval_every_n_step, eval_every_n_step)


fig, ax = plt.subplots(3, 1, figsize=(10, 30))
ax[0].plot(train_loss_results, label='training')
ax[0].plot(eval_range, eval_loss_results, label='eval')
ax[0].legend()
ax[0].set_yscale('log')
ax[0].set_title('total loss')
ax[1].plot(train_rec_loss_results, label='training')
ax[1].plot(eval_range, eval_rec_loss_results, label='eval')
ax[1].set_yscale('log')
ax[1].legend()
ax[1].set_title('rec loss')
ax[2].plot(train_kl_loss_results, label='training')
ax[2].plot(eval_range, eval_kl_loss_results, label='eval')
ax[2].set_title('KL loss')
ax[2].set_yscale('log')
ax[2].legend()
plt.savefig(checkpoint_path + '/Losses.png')
