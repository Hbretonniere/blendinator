from blendinator.models.proba_unet import ProbaUNet
import numpy as np
from astropy.io import fits
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow.keras as tfk
from analyze_utils import *
import logging

''' print in job '''
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('test.log', 'a'))
print = logger.info


def asinh_norm(x, y):
    return tf.asinh(tf.divide(x, tf.math.reduce_max(x))), y

''' Hyper parameters '''

block_size = 1
last_conv = 1
ls_dim = 16
eval_every_n_step = 200

base_channels = 32
list_channels = [8, 16, 32, 64]

stamps_size = 64

# list_channels = [8, 16, 32, 64, 128, 128, 128]

''' Training params '''
nb_epochs = 1
batch_size = 32
lrs = [1e-4] * nb_epochs #* np.exp(-0.1 * np.arange(nb_epochs)) #
betas = [0.001] * nb_epochs #np.logspace(-5, 0, nb_epochs)
# betas = [0.5]

""" Import and preprocess data (for me, segs = ground truth)"""
local_path = 'checkpoints/'
tycho_path = "./data/checkpoints/check_test/"

checkpoint_path = tycho_path #local_path

local_path = './data/'
# tycho_path = '/data/hbretonniere/Euclid/fvae_emulated/'
tycho_path = '/data/hbretonniere/Euclid/fvae_emulated/TU/'

data_path = tycho_path #local_path

# name = 'deblending_field_1_stamps_and_seg.fits'  # there is 148996 stamps
name = 'new_euc_sim_TU_DC_WithoutNoise.fits'  # there is 148996 stamps
# checkpoint_path = "/data57/hbretonniere/deblending/tf2/checkpoints_deblendator/"
# print("checkpoint path : ", checkpoint_path)

""" Import data """
# data = fits.open(data_path + name)[0].data
# imgs = np.expand_dims(data[0], axis=-1).astype('float32')
# segs = np.expand_dims(data[1], axis=-1)

full_imgs = fits.open(tycho_path+'2new_euc_sim_TU_DC_WithoutNoise.fits')[0].data
full_imgs += np.random.normal(0, 2.96e-3, full_imgs.shape)
full_segs = np.load(tycho_path+'2new_euc_sim_TU_DC_seg.npy', allow_pickle=True)[1].array

imgs, segs = cut_grid(full_imgs, full_segs, stamps_size, show=False, x_start=0, y_start=0)
imgs = np.expand_dims(imgs, axis=-1).astype('float32')
segs = np.expand_dims(segs, axis=-1)
print(imgs.shape, segs.shape)
train_slice = int(0.9 * imgs.shape[0])
eval_slice = int(0.95 * imgs.shape[0])
train_steps_per_epoch = int(train_slice / batch_size)

train_data = tf.data.Dataset.from_tensor_slices((imgs[:train_slice], segs[:train_slice])).shuffle(100000, reshuffle_each_iteration=True).map(asinh_norm).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
eval_data = tf.data.Dataset.from_tensor_slices((imgs[train_slice:eval_slice], segs[train_slice:eval_slice])).shuffle(100000, reshuffle_each_iteration=True).map(asinh_norm).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

test_images = imgs[-100:, :, :, 0]
test_segs = segs[-100:, :, :, 0]

# print('training images shape : ', imgs[:train_slice].shape)
# print('eval images shape : ', imgs[train_slice:eval_slice].shape)
# print('test images shape : ', test_images.shape)

''' Create the model '''
PUNet = ProbaUNet((64, 64, 1), ls_dim, list_channels, block_size, last_conv, checkpoint_path, loss='weighted')
PUNet.print_models('models_summary/')

''' Train '''
history = PUNet.train(train_data,
                      nb_epochs, train_steps_per_epoch, lrs, betas)

''' Plot the training losses '''
fig, ax = plt.subplots(1,3, figsize=(15, 5))
ax[0].plot(history[0])
ax[0].set_title('Total loss')
ax[0].set_yscale('symlog')

ax[1].plot(history[1])
ax[1].set_yscale('symlog')
ax[1].set_title('Reconstruction loss')

ax[2].plot(history[2])
ax[2].set_yscale('symlog')
ax[2].set_title('KL loss')
plt.savefig(checkpoint_path+'losses.png')

''' Predict a bunch of images '''
s = 30
nb_to_plot = 10
t = s + nb_to_plot
pred_fig = predict_and_plot(PUNet, asinh_norm(test_images[s:t], test_segs[s:t])[0], test_segs[s:t], nb_to_plot, 'training')
pred_fig.savefig(checkpoint_path+'prediction.png')
