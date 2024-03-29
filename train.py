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


# def asinh_norm(x, y):
#     return tf.asinh(tf.divide(x, tf.math.reduce_max(x))), y

''' Hyper parameters '''

block_size = 3
last_conv = 3
ls_dim = 16
eval_every_n_step = 200

base_channels = 32
list_channels = [16, 32, 64, 128]

stamps_size = 128

# list_channels = [8, 16, 32, 64, 128, 128, 128]

''' Training params '''
nb_epochs = 2
batch_size = 32
lrs = [1e-4]  * np.exp(-0.1 * np.arange(nb_epochs)) #
betas = [0.01] * np.logspace(-5, 0, nb_epochs)
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
name = 'for_deblendingsim_TU_DC_WithoutNoise.fits'
# name = 'new_euc_sim_TU_DC_WithoutNoise.fits'  # there is 148996 stamps
# checkpoint_path = "/data57/hbretonniere/deblending/tf2/checkpoints_deblendator/"
# print("checkpoint path : ", checkpoint_path)

""" Import data """
# data = fits.open(data_path + name)[0].data
# imgs = np.expand_dims(data[0][1000:], axis=-1).astype('float32')
# segs = np.expand_dims(data[1][1000:], axis=-1)

full_imgs1 = fits.open(tycho_path+name)[0].data
full_imgs1 += np.random.normal(0, 4.69e-4, full_imgs1.shape)
full_segs1 = np.load(tycho_path+'for_deblendingsim_TU_DC_seg.npy', allow_pickle=True)[1].array

full_imgs2 = fits.open(tycho_path+'first_'+name)[0].data
full_imgs2 += np.random.normal(0, 4.69e-4, full_imgs2.shape)
full_segs2 = np.load(tycho_path+'first_for_deblendingsim_TU_DC_seg.npy', allow_pickle=True)[1].array

full_imgs = np.zeros((15300, 30100))
full_imgs[:, :15300] = full_imgs1
full_imgs[::, 15000:] = full_imgs2[:, 200:]

full_segs = np.zeros((15300, 30100))
full_segs[:, :15300] = full_segs1
full_segs[::, 15000:] = full_segs2[:, 200:]

imgs, segs = cut_grid(full_imgs, full_segs, stamps_size, show=False, x_start=200, y_start=200)


fig, ax = plt.subplots(20, 2, figsize=(10, 40))
for i in range(20):
    ax[i, 0].imshow(imgs[i, :, :, 0])
    ax[i, 1].imshow(segs[i, :, :, 0])
plt.savefig(checkpoint_path+'inputs.png')
    
train_slice = int(0.9 * imgs.shape[0])
eval_slice = int(0.95 * imgs.shape[0])
train_steps_per_epoch = int(train_slice / batch_size)

''' If the stamps are not yet normalized '''
# train_data = tf.data.Dataset.from_tensor_slices((imgs[:train_slice], segs[:train_slice])).shuffle(100000, reshuffle_each_iteration=True).map(asinh_norm).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
# eval_data = tf.data.Dataset.from_tensor_slices((imgs[train_slice:eval_slice], segs[train_slice:eval_slice])).shuffle(100000, reshuffle_each_iteration=True).map(asinh_norm).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

''' If the stamps are already normalized '''

train_data = tf.data.Dataset.from_tensor_slices((imgs[:train_slice], segs[:train_slice])).shuffle(100000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
eval_data = tf.data.Dataset.from_tensor_slices((imgs[train_slice:eval_slice], segs[train_slice:eval_slice])).shuffle(100000, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

test_images = imgs[-500:, :, :]
test_segs = segs[-500:, :, :]
plot_imgs = imgs[:10], segs[:10]

del full_imgs
del full_segs
del imgs
del segs

# print('training images shape : ', imgs[:train_slice].shape)
# print('eval images shape : ', imgs[train_slice:eval_slice].shape)
# print('test images shape : ', test_images.shape)

''' Create the model '''
PUNet = ProbaUNet((stamps_size, stamps_size, 1), ls_dim, list_channels, block_size, last_conv, checkpoint_path, loss='classic')
PUNet.print_models('models_summary/')

''' Train '''
history = PUNet.train(train_data,
                      nb_epochs, train_steps_per_epoch, lrs, betas, plot_frequency=20, plot_images=plot_imgs)

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
s = 0
nb_to_plot = 10
t = s + nb_to_plot
pred_fig = predict_and_plot(PUNet, test_images[s:t], test_segs[s:t], nb_to_plot, 'training')
pred_fig.savefig(checkpoint_path+'prediction.png')
