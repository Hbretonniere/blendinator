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
    lrs, betas, history, checkpoint_path, save_frequency=1, plot_frequency=100):
    import matplotlib.pyplot as plt
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
                        
                        if batch_number % plot_frequency == 0 :
                            prediction = PUNet.training_model(image, label)[0]
                            fig, ax = plt.subplots(5, 4)

                            for i in range(5):
                                ax[i, 0].imshow(test_images[i, :, :, 0])
                                ax[i, 1].imshow(test_segs[i, :, :, 0])
                                ax[i, 2].imshow(np.mean(predictions[i], axis=0))
                                ax[i, 3].imshow(np.var(predictions[i], axis=0))
                                plt.savefig(f'./data/training_img_step_{nb_epoch*step_per_epoch + nb_step}')

                if epoch % save_frequency == 0:
                    PUNet.save_weights(epoch, checkpoint_path)
    return(history)