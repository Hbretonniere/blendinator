import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

@tf.function
def predict(PUNet, images, nb_realisation):
    nb_image = images.shape[0]
    predictions = np.zeros((nb_image, nb_realisation, images.shape[1], images.shape[2]))
    for realisation in range(nb_realisation):
        sample = PUNet.prediction_model(images)
        predictions[:, realisation, :, :] = np.argmax(sample.eval, axis=-1) # .eval because of the decorator wich make the function tensorflow graph like
    return predictions

def summary(predictions, threshold_iso, threshold_blend):
    shape = (3,) + (np.shape(predictions)[0],) + np.shape(predictions)[2:]
    print('shape :', shape)
    summaries = np.zeros(shape)
    summaries[0] = np.mean(predictions, axis=1)
    summaries[1] = np.var(predictions, axis=1)
    summaries[2] = np.where(summaries[1]>threshold_iso, 1, 0)
    summaries[2] = np.where(summaries[1]>threshold_blend, 2, summaries[2])
    return(summaries)