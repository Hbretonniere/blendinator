import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np


def predict(PUNet, image, nb_realisation):
    nb_image = image.shape[0]
    predictions = np.zeros((nb_image, nb_realisation, image.shape[1], image.shape[2]))
    for i in range(nb_image):
        for realisation in range(nb_realisation):
            sample = PUNet.prediction_model(image[i:i+1])
            predictions[i, realisation, :, :] = np.argmax(sample, axis=-1)
    return predictions

def summary(predictions, threshold_iso, threshold_blend):
    shape = (3,) + (np.shape(predictions)[0],) + np.shape(predictions)[2:]
    print('shape :', shape)
    summaries = np.zeros(shape)
    summaries[0] = np.mean(predictions, axis=1)
    summaries[1] = np.var(predictions, axis=1)
    summaries[2] = np.where(predictions[0]>threshold_iso, 1, 0)
    summaries[2] = np.where(predictions[0]>threshold_blend, 2, summaries[2])
    return(summaries)