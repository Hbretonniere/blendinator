from blendinator import ProbaUNet
import numpy as np
from astropy.io import fits
import tensorflow as tf
from train import train
from predict import predict
from predict import summary

import matplotlib.pyplot as plt

def asinh_norm(x, y):
    return tf.asinh(tf.divide(x, tf.math.reduce_max(x))), y


""" Training Parameters """
list_channels = [8, 16, 32, 64, 128, 128, 128, 128]
nb_epochs = 1
batch_size = 32
lrs = np.zeros(nb_epochs) + 1.e-4 # 1e-3 * np.exp(-0.1 * np.arange(nb_epochs))
print(lrs)
betas = np.ones(nb_epochs)

eval_every_n_step = 200

""" Import and preprocess data (for me, segs = ground truth)"""
checkpoint_path = "./data/checkpoints/check_test"

""" Import data """
imgs = fits.open('./data/small_train_dset_imgs.fits')[0].data[:10000]
segs = fits.open('./data/small_train_dset_segs.fits')[0].data[:10000]

train_slice = int(0.9 * imgs.shape[0])
eval_slice = int(1 * imgs.shape[0])
train_steps_per_epoch = int(train_slice / batch_size)

train_data = tf.data.Dataset.from_tensor_slices((imgs[:train_slice], segs[:train_slice])).shuffle(100000, reshuffle_each_iteration=True).map(asinh_norm).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
eval_data = tf.data.Dataset.from_tensor_slices((imgs[train_slice:eval_slice], segs[train_slice:eval_slice])).shuffle(100000, reshuffle_each_iteration=True).map(asinh_norm).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

test_images = imgs[train_slice:train_slice+10, :, :, :]
test_segs = segs[train_slice:train_slice+10]

print('test images shape : ', imgs.shape)
print('test images shape : ', test_images.shape)

print("importing the Punet object")

PUnet = ProbaUNet((128, 128, 1), 10, list_channels, 1, 1)

# print('\n \n Summary of the training model')
# PUnet.training_model.summary()
# print('\n \n Summary of the prediction model')
# PUnet.prediction_model.summary()

train_model = PUnet.training_model

print('\n \n Training the model')

# # ''' Oui je sais... je fais exactement ce que tu m as dit que je ne devrais pas faire....'''
# history = [[],[],[]]

# history = train(PUnet, train_data, nb_epochs, train_steps_per_epoch, lrs, betas, history, checkpoint_path)

# plt.figure()
# plt.plot(history[0])

print('Restoring the model')
PUNet_restored = ProbaUNet((128, 128, 1), 10, list_channels, 1, 1)
PUNet_restored.load_weights(checkpoint_path)


predictions = predict(PUNet_restored, test_images, 10)

summaries = summary(predictions, 0.5, 1.01)

fig, ax = plt.subplots(5, 5)
ax[0, 0].set_title('input img')
ax[0, 1].set_title('input label')
ax[0, 2].set_title('mean pred')
ax[0, 3].set_title('var pred')
ax[0, 4].set_title('thres pred')
for i in range(5):
    ax[i, 0].imshow(test_images[i, :, :, 0])
    ax[i, 1].imshow(test_segs[i, :, :, 0])
    ax[i, 2].imshow(summaries[0, i])
    ax[i, 3].imshow(summaries[1, i])
    ax[i, 4].imshow(summaries[2, i])

plt.show()