import tensorflow as tf
import tensorflow.keras as tfk

from tqdm import tqdm
from blendinator.models import ProbaUNet


def train(PUNet, train_data, epochs, step_per_epoch,
    lrs, betas, history, checkpoint_path, save_frequency=1):
    with tf.device('GPU:0'):
        with tqdm(total=epochs, desc='Epoch', position=0) as bar1:
            for epoch in range(epochs):
                bar1.update(1)
                with tqdm(total=step_per_epoch, desc='batch', position=1) as bar2:
                    for batch_nb, (image, label) in enumerate(train_data):
                        bar2.update(1)
                        total_loss, rec_loss, kl_loss = PUNet.train_step(image, label, lrs[epoch], betas[epoch])
                        history[0].append(total_loss)
                        history[1].append(rec_loss)
                        history[2].append(kl_loss)

                        # if k % eval_every_n_step == 0:
                        #     val_tot, val_rec, val_kl = 0, 0, 0
                        #     for valid,(image, label) in enumerate(eval_data):
                        #         total_loss, rec_loss, kl_loss = PUNet.eval_step(image, label)
                        #         val_tot += total_loss
                        #         val_rec += rec_loss
                        #         val_kl += kl_loss

                        #     eval_loss_results.append(val_tot/valid)
                        #     eval_rec_loss_results.append(val_rec/valid)
                        #     eval_kl_loss_results.append(val_kl/valid)
                        bar2.set_description(f"PUnet, loss={history[0][-1]}")
                        # k+=1
                if epoch % save_frequency == 0:
                    PUNet.save_weights(epoch, checkpoint_path)
    return(history)

def train_with_plot(PUNet, train_data, epochs, step_per_epoch,
    lrs, betas, history, checkpoint_path, plot_data, save_frequency=1, plot_frequency=100):
    import matplotlib.pyplot as plt
    
    plot_image, plot_label = plot_data[0], plot_data[1]
    
    with tf.device('GPU:0'):
        with tqdm(total=epochs, desc='Epoch', position=0) as bar1:
            for epoch in range(epochs):
                bar1.update(1)
                with tqdm(total=step_per_epoch, desc='batch', position=1) as bar2:
                    for batch_nb, (image, label) in enumerate(train_data):
                        bar2.update(1)
                        total_loss, rec_loss, kl_loss = PUNet.train_step(image, label, lrs[epoch], betas[epoch])
                        history[0].append(total_loss)
                        history[1].append(rec_loss)
                        history[2].append(kl_loss)
                        bar2.set_description(f"PUnet, loss={history[0][-1]}")

                        if batch_nb % plot_frequency == 0 :
                            predictions = np.zeros((plot_image.shape[0], 10, plot_image.shape[1], plot_image.shape[2]))

                            for realisation in range(10):
                                sample = PUNet.prediction_model(plot_image)
                                predictions[:, realisation, :, :] = np.argmax(sample.eval, axis=-1) # .eval because of the decorator wich make the function tensorflow graph like
                            summaries = summary(predictions, 0.5, 1.01)
                            print(sample.shape)
                            fig, ax = plt.subplots(5, 6, figsize=(10,10))
                            for i in range(5):
                                ax[i, 0].imshow(plot_image[i, :, :, 0])
                                ax[i, 1].imshow(plot_label[i, :, :, 0])
                                for j in range(3):
                                    ax[i, 2+j].imshow(sample[i, :, :, j], vmin = np.min(sample[:, :, j]),
                                                     vmax = np.max(sample[:, :, j]))
                                ax[i, 5].imshow(np.argmax(sample[i], axis=-1))
                                plt.savefig(f'./images/training_noargmax_img_step_{epoch*step_per_epoch + batch_nb}')

                            fig, ax = plt.subplots(5, 5, figsize=(10,10))
                            for i in range(5):
                                ax[i, 0].imshow(plot_image[i, :, :, 0])
                                ax[i, 1].imshow(plot_label[i, :, :, 0])
                                ax[i, 2].imshow(summaries[0, i])
                                ax[i, 3].imshow(summaries[1, i])
                                ax[i, 4].imshow(summaries[2, i], vmin=0, vmax=2)
                                plt.savefig(f'./images/training_img_step_{epoch*step_per_epoch + batch_nb}')

    return(history)