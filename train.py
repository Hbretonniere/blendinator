import tensorflow as tf
import tensorflow.keras as tfk

from tqdm import tqdm
from blendinator.models import ProbaUNet


def train(model, train_data, epochs, step_per_epoch, lrs, betas):

    with tf.device('GPU:0'):
        with tqdm(total=epochs, desc='Epoch', position=0) as bar1:
            for epoch in range(epochs):
                bar1.update(1)
                with tqdm(total=step_per_epoch, desc='batch', position=1) as bar2:
                    for batch_nb, (features, gt) in enumerate(train_data):
                        bar2.update(1)

                        total_loss, rec_loss, kl_loss = model.train_step(features, gt, lrs[epoch], betas[epoch])
                        train_loss_results.append(total_loss)
                        train_rec_loss_results.append(rec_loss)
                        train_kl_loss_results.append(kl_loss)

                        if k % eval_every_n_step == 0:
                            val_tot, val_rec, val_kl = 0, 0, 0
                            for valid,(features, gt) in enumerate(eval_data):
                                total_loss, rec_loss, kl_loss = model.eval_step(features, gt)
                                val_tot += total_loss
                                val_rec += rec_loss
                                val_kl += kl_loss

                            eval_loss_results.append(val_tot/valid)
                            eval_rec_loss_results.append(val_rec/valid)
                            eval_kl_loss_results.append(val_kl/valid)
                        bar2.set_description(f"PUnet, loss={train_loss_results[-1]}")
                        k+=1
                    if epoch % 2 == 0:
                        model.save_weights(f"{checkpoint_path}/checkpoint_{len(train_loss_results)}")