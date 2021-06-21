import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval


def show_data(img, seg, stamp_size, cmap='zscale'):
    fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    interval = ZScaleInterval()
    a = interval.get_limits(img)

    ax[0].imshow(img, vmin=a[0], vmax=a[1], cmap='bone')
    ax[0].set_yticks(np.linspace(0, img.shape[0], int(img.shape[0]//stamp_size)))
    ax[0].set_xticks(np.linspace(0, img.shape[1], int(img.shape[1]//stamp_size)))
    ax[0].grid(color='red', alpha=0.5)
    
    ax[1].imshow(seg)
    ax[1].set_yticks(np.linspace(0, img.shape[0], int(img.shape[0]//stamp_size)))
    ax[1].set_xticks(np.linspace(0, img.shape[1], int(img.shape[1]//stamp_size)))
    ax[1].grid(color='red', alpha=0.5)


def cut_grid(img, seg, stamp_size, show=False, x_start=128, y_start=128):
    
    x_field, y_field = np.shape(img)[0], np.shape(img)[1]
    x_field -= x_start
    y_field -= y_start
    n_row = x_field//stamp_size
    n_column = y_field//stamp_size
    nb_stamp_test = n_row * n_column
    
    stamps_img_test = np.zeros((nb_stamp_test, stamp_size, stamp_size, 1), dtype=np.float32)
    stamps_seg_test = np.zeros((nb_stamp_test, stamp_size, stamp_size, 1), dtype=np.float32)
    
    patch_size = [stamp_size, stamp_size]
    x_i = x_start                        
    stamp = 0
    for _ in range(n_row):               # go through all the rows
        y_i = y_start                    
        for _ in range(n_column):        # go trough all the columns
            stamps_img_test[stamp, :, :, 0] = np.arcsinh(img[x_i:x_i + stamp_size, y_i:y_i + stamp_size]
                                                         /img[x_i:x_i + stamp_size, y_i:y_i + stamp_size].max())
            stamps_seg_test[stamp, :, :, 0] = seg[x_i:x_i + stamp_size, y_i:y_i + stamp_size]
    
            y_i += stamp_size               # got to the first pixel of the next stamp
            stamp += 1
        x_i += stamp_size                   # got to the first pixel of the next stamp
    if show :
        fig, ax = plt.subplots(n_row, n_column, figsize=(10, 3))
        ax = ax.flatten()
        for i in range(n_row*n_column):
            interval = ZScaleInterval()
            a = interval.get_limits(stamps_img_test[i, :, :, 0])
            ax[i].imshow(stamps_img_test[i, :, :, 0], vmin=a[0], vmax=a[1])

        fig, ax = plt.subplots(n_row, n_column, figsize=(10, 3))
        ax = ax.flatten()
        for i in range(n_row*n_column):
            a = interval.get_limits(stamps_img_test[i, :, :, 0])
            ax[i].imshow(stamps_seg_test[i, :, :, 0])

    return stamps_img_test, stamps_seg_test



def predict_and_plot(model, features, labels, nb_to_plot, mode):


    predictions = model.prediction_model([features])

    fig, ax = plt.subplots(nb_to_plot, 3, figsize=(3*5,nb_to_plot*5))
    ax[0, 0].set_title('Input Image')
    ax[0, 1].set_title('Input Gt')
    ax[0, 2].set_title('Predicted segmap')
    print(np.shape(predictions))
    for i in range(nb_to_plot):
        segmap = np.argmax(predictions[i], axis=-1)
        ax[i, 0].imshow(features[i, :, :, 0])
        ax[i, 1].imshow(labels[i, :, :, 0])
        ax[i, 2].imshow(segmap)

    return fig