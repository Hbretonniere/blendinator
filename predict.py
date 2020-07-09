import tensorflow as tf
import tensorflow.keras as tfk

def predict(model, image, nb_realisation, nb_batch):
    # mode = tf.constant([1.])
    batch_size = int(features[:, 0, :, :].shape[0]/nb_batch)
    predictions = np.zeros((nb_batch, batch_size, nb_realisation, image_shape[0], image_shape[1]))
    for batch in range(nb_batch):
        for realisation in range(nb_realisation):
            sample = model([image[int(batch_size*batch):int(batch_size*(batch+1))], features[int(batch_size*batch):int(batch_size*(batch+1))], mode])[0]
            prediction[batch, :, realisation, :, :] = np.argmax(sample, axis=-1)
    return np.reshape(predictions, (nb_batch*batch_size, nb_realisation, image_shape[0], image_shape[1]))