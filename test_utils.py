from skimage.segmentation import clear_border
from skimage import measure
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.ops.init_ops import VarianceScaling
from tensorflow.compat.v1 import truncated_normal_initializer
from tensorflow.keras import backend as K
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dropout, Input, concatenate, Reshape
from tensorflow.keras.backend import resize_images
tfd = tfp.distributions

'''   Measure.label` put a different number on every separated object. In the case of a blend,  we will have two objects : the union of the two galaxies without the intresection, and the intersection (the blend). To consider each object even blended as one object, we put all the non bg pixels to one '''


def he_normal(seed=None):
    """He normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.
    Arguments:
        seed: A Python integer. Used to seed the random generator.
    Returns:
          An initializer.
    References:
          He et al., http://arxiv.org/abs/1502.01852
    Code:
        https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/python/keras/initializers.py
    """
    return VarianceScaling(scale=2., mode='fan_in', distribution='normal', seed=seed)


def downblock(features,
              output_channels,
              kernel_shape=(3, 3),
              stride=1,
              rate=1,
              pad='SAME',
              num_convs=3,
              w_initializers=he_normal(),
              b_initializers=truncated_normal_initializer(stddev=0.001),
              w_regularizers=None,
              b_regularizers=None,
              nonlinearity='relu',
              down_sample_input=True,
              data_format='channels_last',
              name='down_block'):

    if down_sample_input:
        features = AveragePooling2D(pool_size=2, strides=2,
                                    padding='SAME', data_format=data_format)(features)

    # print('downblock shape input :', np.shape(features))
    for j in range(num_convs):
        features = Conv2D(output_channels, kernel_shape, stride, pad, data_format=data_format, dilation_rate=rate,
                          kernel_initializer=w_initializers, bias_initializer=b_initializers,
                          kernel_regularizer=w_regularizers, bias_regularizer=b_regularizers, name=name+"_"+str(j), activation=nonlinearity)(features)
        # print('downblock shapw :', np.shape(features))
    return features


def upblock(lower_res_inputs,
            same_res_inputs,
            output_channels,
            kernel_shape=(3, 3),
            stride=1,
            rate=1,
            pad='SAME',
            num_convs=2,
            w_initializers=he_normal(),
            b_initializers=truncated_normal_initializer(stddev=0.001),
            w_regularizers=None,
            b_regularizers=None,
            nonlinearity=relu,
            data_format='channels_last',
            name='up_block'):
    features = resize_images(lower_res_inputs, height_factor=2, width_factor=2, data_format=data_format, interpolation='bilinear')
#     same_res_inputs = tf.cast(same_res_inputs, dtype=tf.float32)
    features = tf.concat([features, same_res_inputs], axis=-1)

    for j in range(num_convs):
        features = Conv2D(output_channels, kernel_shape, stride, pad, dilation_rate=rate, data_format=data_format,
                          kernel_initializer=w_initializers, bias_initializer=b_initializers,
                          kernel_regularizer=w_regularizers, bias_regularizer=b_regularizers, activation=nonlinearity, name=name+'_'+str(j))(features)

    return features


def create_punet(list_num_channels, num_conv_per_block, nb_conv_1x1, stamp_size, latent_dim):
    unet_input = Input(shape=(stamp_size, stamp_size, 1))#, dtype='float64')
    features=[unet_input]
    for i, n_channels in enumerate(list_num_channels):
        down_sample = True
        if i == 0:
            down_sample = False
        features.append(downblock(features[-1],
                                  n_channels,
                                  num_convs=num_conv_per_block,
                                  down_sample_input=down_sample,
                                  name='Unet_down_block_'+str(i)))
    encoder_output = features[1:]

    n = len(encoder_output) - 2

    lower_reso_features = encoder_output[-1]
    for i in range(n, -1, -1):
        same_reso_features = encoder_output[i]
        n_channels = list_num_channels[i]
        lower_reso_features = upblock(lower_reso_features,
                                      same_reso_features,
                                      output_channels=n_channels,
                                      num_convs=num_conv_per_block,
                                      name='Unet_upblock_'+str(n-i))



    unet_output = lower_reso_features
    unet = tf.keras.Model(unet_input, unet_output, name='unet')

    ''' Image encoder'''
    img_encoder_input = Input(shape=(stamp_size, stamp_size, 1), dtype='float64')
    features = img_encoder_input
    for i, n_channels in enumerate(list_num_channels):
        down_sample = True
        if i == 0:
            down_sample = False
        features = downblock(features, n_channels, num_convs=num_conv_per_block, down_sample_input=down_sample, name='img_down_block_'+str(i))

    features = tf.reduce_mean(features, axis=[1,2], keepdims=True)
    mu_log_sigma = Conv2D(2*latent_dim, (1,1), 1, 'SAME', kernel_initializer=he_normal(), bias_initializer=truncated_normal_initializer(stddev=0.001), kernel_regularizer=tf.keras.regularizers.l2(1.0), bias_regularizer=tf.keras.regularizers.l2(1.0), activation='relu')(features)
    mu_log_sigma = tf.squeeze(mu_log_sigma, axis=[1,2])
    #     mu = mu_log_sigma[:, :latend_dim]
    #     log_sigma = mu_log_sigma[:, latent_dim:]
    img_encoder_output = mu_log_sigma  #[mu, log_sigma]
    img_encoder = tf.keras.Model(img_encoder_input, img_encoder_output, name='img_encoder')
    img_encoder.summary()

    ''' GT encoder'''
    gt_input = Input(shape=(stamp_size, stamp_size, 1))#, dtype='float64')
    # gt = tf.cast(gt_input, tf.float32)
    img_input = Input(shape=(stamp_size, stamp_size, 1))#, dtype='float64')
    features = tf.concat([img_input, gt_input], axis=-1)

    for i, n_channels in enumerate(list_num_channels):
        down_sample = True
        if i == 0:
            down_sample = False
        features = downblock(features, n_channels, num_convs=num_conv_per_block, down_sample_input=down_sample, name='img_down_block_'+str(i))
    features = tf.reduce_mean(features, axis=[1,2], keepdims=True)
    mu_log_sigma = Conv2D(2*latent_dim, (1,1), 1, 'SAME', kernel_initializer=he_normal(), bias_initializer=truncated_normal_initializer(stddev=0.001), kernel_regularizer=tf.keras.regularizers.l2(1.0), bias_regularizer=tf.keras.regularizers.l2(1.0), activation='relu')(features)
    mu_log_sigma = tf.squeeze(mu_log_sigma, axis=[1,2])
    gt_encoder_output = mu_log_sigma  #[mu, log_sigma]
    gt_encoder = tf.keras.Model(inputs=[img_input, gt_input], outputs=[gt_encoder_output], name='gt_encoder')


    z_input = Input(shape=img_encoder_output.shape[1:])
    features_input = Input(shape=unet_output.shape[1:])

    """ Create the gaussian and sample from it """
    mu = z_input[:, :latent_dim]
    log_sigma = z_input[:, latent_dim:]
    sample = tfp.layers.DistributionLambda(make_distribution_fn=lambda t:\
             tfd.MultivariateNormalDiag(loc=t[0], scale_diag=tf.exp(t[1])), convert_to_tensor_fn=lambda s: s.sample())([mu, log_sigma])

    """ Make the sampling concatanable to the features"""
    shape = features_input.shape
    spatial_shape = [shape[axis] for axis in [1,2]]
    multiples = [1] + spatial_shape
    multiples.insert(3, 1)
    if len(sample.shape) == 2:
        sample = Reshape((1, 1, latent_dim))(sample)
    broadcast_sample = tf.tile(sample, multiples)

    features = tf.concat([features_input, broadcast_sample], axis =-1)
    # features = features_input
    for i in range(nb_conv_1x1):
        reconstructed = Conv2D(3, (1,1), 1, 'SAME', kernel_initializer=he_normal(), bias_initializer=truncated_normal_initializer(stddev=0.001), kernel_regularizer=tf.keras.regularizers.l2(1.0), bias_regularizer=tf.keras.regularizers.l2(1.0), activation='relu')(features)
    combinatory_output = reconstructed
    combinatory = tf.keras.Model(inputs=[features_input, z_input], outputs=[reconstructed], name='Combinatory')
    combinatory.summary()

    """ Punet """
    img_input = Input(shape=(128, 128, 1))#, dtype='float64')
    gt_input = Input(shape=(128, 128, 1))#, dtype='float64')
    mode = Input(shape=(), batch_size=1)
    ls_img = img_encoder(img_input)
    ls_seg = gt_encoder([img_input, gt_input])
    unet_features = unet(img_input)
    is_training = tf.cast(0, dtype="float32")
    prediction = K.switch(mode==is_training.numpy(), combinatory([unet_features, ls_seg]), combinatory([unet_features, ls_img]))

    # Verify if the switch works
    # was_training = K.switch(mode==is_training.numpy(), tf.constant('is_training', shape=(1,)), tf.constant('is_not_training', shape=(1,)))
    # proba_unet = keras.Model(inputs=[img_input, gt_input, mode], outputs=[prediction, ls_img, ls_seg, was_training], name='Proba_Unet')

    proba_unet = keras.Model(inputs=[img_input, gt_input, mode], outputs=[prediction, ls_img, ls_seg], name='Proba_Unet')
    proba_unet.summary()
    return proba_unet

def measure_obj(img, cb):
    tmp_map = np.where(img == 2, 1, img)
    obj_map = clear_border(measure.label(tmp_map), cb)
    del tmp_map
    return obj_map

""" images shape : (nb_batch, batch_size, 128, 128)
    return prediction of shape (nb_batch, batch_size, nb_samples, 128, 128) """

def pred_from_trained(checkpoint_path, model, images, nb_batch, nb_sample, stamp_size=128):
    
    latest = tf.train.latest_checkpoint(checkpoint_path)
    print(latest)
    model.load_weights(latest)
    batch = int(images[:, 0, :, :].shape[0]/nb_batch)

    test_samples = np.zeros((nb_batch, batch, nb_sample, stamp_size, stamp_size))
    for j in range(nb_batch):
        for i in range(nb_sample):
            tmp = model([images[int(batch*j):int(batch*(j+1))], images[int(batch*j):int(batch*(j+1))], tf.constant([1.])])[0]
            test_samples[j, :, i, :, :] = np.argmax(tmp, axis=-1)
    test_samples = test_samples.reshape(nb_batch*batch, nb_sample, stamp_size, stamp_size)
    return test_samples

def produce_cat(img, gt, pred, gt_obj_map, pred_obj_map, gt_cat, is_score=True, is_show=None):

        cat = pd.DataFrame(columns=['TU_Index', 'TU_Index_blend', 'x', 'y', 'loc', 'distance', 'area', 'area_blend', 'obj_status', 'blend_status', 'mag', 'rad', 'IOU', 'IOU_blend', 'var', 'var_blend'])
        sz = 80    # size of the stamp to compute the IOU on
        mid = 40
        index = -1   # index of the object in the catalogue
        dr = 5       # epsilon to try to match the galaxy
        k = 1        # object number skip the background "object"
        no_match = 0 
        seen_ = 0
        pb = 0
        plot = 0

        while k <= pred_obj_map.max():
            if k % 100 ==0:
                print(f'k ={k}')
            index += 1    #index for the predicted catalogue

            '''create the stamp of the objects'''
            pred_obj_stamp = np.zeros((3, sz, sz))
            gt_stamp = np.zeros((sz, sz))

            try:
                '''detect the k-th object'''
                a = np.where(pred_obj_map == k)      # coords of all the pixels belonging to the kth object
                if len(a[0]) == 0:
                    print(f"{k}-th object is null")
                    k += 1
                    continue
                sizex, sizey = a[0].max()+1 -a[0].min(), a[1].max()+1 -a[1].min()   # corresponding size of the galaxy (square)

                cat = cat.append(pd.Series(), ignore_index=True)

                ''' Add the location to the catalogue to do the better detection'''
                cat['loc'].iloc[-1] = [a[0].min(), a[0].max(), a[1].min(), a[1].max()]

                ''' Add the coords to the catalogue (for now, the centroid is just the middle of the stamp)'''
                cat['x'].iloc[-1] = int(np.mean(a[0]))
                cat['y'].iloc[-1] = int(np.mean(a[1]))

                ''' Add the area to the catalogue '''
                cat['area'].iloc[-1] = [int(len(a[0])), None]
                if not is_score :
                    k += 1
                    continue

                pred_obj_stamp[:, int(mid-sizex/2):int(mid+sizex/2), int(mid-sizey/2):int(mid+sizey/2)] = pred[:, a[0].min():a[0].max()+1, a[1].min():a[1].max()+1]
                cat['area_blend'].iloc[-1] = [int(len(np.where(pred_obj_stamp[0]==2)[0])), None]
                
                ''' Macth the object(s) of the TU catalogue : we look in x+/-dr, y+-dr and see if there is a galaxy 
                    in this interval in the TU cat (x/y are inverted compared to numpy...) '''

                indexes_tu = np.where((gt_cat[:, 3] > a[0].min()- dr) & (gt_cat[:, 3] < a[0].max()+dr) & (gt_cat[:, 2] > a[1].min()-dr) & (gt_cat[:, 2] < a[1].max()+dr))[0]
        #         print(f'nb of matching gal : {len(index_tu)}')

                if len(indexes_tu) == 0:      # there is no matching index
                    index_tu = -1           # faux positif

                else:
                    index_tu = indexes_tu[0]
        #         display(cat)

                ''' Add the Detection flag'''
                if index_tu == -1:      # False positive
                    cat['obj_status'].iloc[-1] = 'FP'   # False positive for Detection flag
                    k += 1
                    if pred_obj_stamp.max() > 1:
                        cat['blend_status'].iloc[-1] = 'P'
                    else:
                        cat['blend_status'].iloc[-1] = 'N'
                    continue

                cat['obj_status'].iloc[-1] = 'TP'

                ''' Isolate the matching true object : same method but with the coordinates of the matching object '''

                a = np.where(gt_obj_map == gt_obj_map[int(gt_cat[index_tu, 3])-2:int(gt_cat[index_tu, 3])+2, int(gt_cat[index_tu, 2])-2:int(gt_cat[index_tu, 2])+2].max())
        #             print('2', a[0].min(), a[0].max())
                if gt_obj_map[int(gt_cat[index_tu, 3])-2:int(gt_cat[index_tu, 3])+2, int(gt_cat[index_tu, 2])-2:int(gt_cat[index_tu, 2])+2].max() == 0:
                    print(" Matching object does not exist")
                    cat = cat.drop(cat.shape[0]-1, 0)
                    k += 1
                    no_match += 1
                    continue

                sizex = a[0].max()+1 -a[0].min()
                sizey = a[1].max()+1 -a[1].min()

        #             gt_stamp[int(mid-sizex/2):int(mid+sizex/2), int(mid-sizey/2):int(mid+sizey/2)] = gt_obj_map[a[0].min():a[0].max()+1, a[1].min():a[1].max()+1]
                gt_stamp[int(mid-sizex/2):int(mid+sizex/2), int(mid-sizey/2):int(mid+sizey/2)] = gt[a[0].min():a[0].max()+1, a[1].min():a[1].max()+1]
                
                cat['area'].iloc[-1][1] = int(len(a[0]))
                cat['area_blend'].iloc[-1][1] = int(len(np.where(gt_stamp == 2)[0]))

                '''Check if the object is a TP(1)/FP(2)/TN(3)/FN(4) blend'''

                if gt_stamp.max() > 1 and pred_obj_stamp[0].max() > 1:
                    cat['blend_status'].iloc[-1] = 'TP'
                    indexes_tu = np.where((gt_cat[:, 3] > a[0].min()) & (gt_cat[:, 3] < a[0].max()) & (gt_cat[:, 2] > a[1].min()) & (gt_cat[:, 2] < a[1].max()))[0]
                    cat['TU_Index'].iloc[-1] = int(gt_cat[indexes_tu[0], 0])
                    cat['TU_Index_blend'].iloc[-1] = int(gt_cat[indexes_tu[1], 0])
                    cat['mag'].iloc[-1] = round(gt_cat[indexes_tu[0], 4], 2)
                    cat['rad'].iloc[-1] = round(gt_cat[indexes_tu[0], 6], 2)
        #             k += 1
                    cat['distance'] = round(np.sqrt((gt_cat[indexes_tu[0], 2] - gt_cat[indexes_tu[1], 2])**2 - (gt_cat[indexes_tu[0], 3] - gt_cat[indexes_tu[1], 3])**2), 1)
                
                if gt_stamp.max() == 1 and pred_obj_stamp[0].max() > 1:
                    cat['TU_Index'].iloc[-1] = int(gt_cat[index_tu, 0])
                    cat['blend_status'].iloc[-1] = 'FP'
                    cat['mag'].iloc[-1] = round(gt_cat[index_tu, 4])
                    cat['rad'].iloc[-1] = round(gt_cat[index_tu, 6])
        #             k += 1
                if gt_stamp.max() == 1 and pred_obj_stamp[0].max() == 1:
                    cat['TU_Index'].iloc[-1] = int(gt_cat[index_tu, 0])
        #             print(tu_cat[index_tu, 4])
                    cat['mag'].iloc[-1] = round(gt_cat[index_tu, 4])
                    cat['rad'].iloc[-1] = round(gt_cat[index_tu, 6])
                    cat['blend_status'].iloc[-1] = 'TN'
                    blend_status = 'TN'
        #             k += 1
                if gt_stamp.max() > 1 and pred_obj_stamp[0].max() == 1:
                    cat['blend_status'].iloc[-1] = 'FN'
                    indexes_tu = np.where((gt_cat[:, 3] > a[0].min()- dr) & (gt_cat[:, 3] < a[0].max()+dr) & (gt_cat[:, 2] > a[1].min()-dr) & (gt_cat[:, 2] < a[1].max()+dr))[0]
                    if gt_cat[indexes_tu[0],0] in cat['TU_Index'].values:
                        seen_ +=1
                        place = 1
                        other = 0
                    else:
                        place = 0
                        other = 1
                    cat['TU_Index'].iloc[-1] = int(gt_cat[indexes_tu[place], 0])
                    cat['TU_Index_blend'].iloc[-1] = int(gt_cat[indexes_tu[other], 0])
                    cat['mag'].iloc[-1] = round(gt_cat[indexes_tu[place], 4], 2)
                    cat['rad'].iloc[-1] = round(gt_cat[indexes_tu[place], 6], 2)
                    cat['distance'] = round(np.sqrt((gt_cat[indexes_tu[place], 2] - gt_cat[indexes_tu[other], 2])**2 + (gt_cat[indexes_tu[place], 3] - gt_cat[indexes_tu[other], 3])**2), 1)

                stamp_img = img[0, 0, a[0].min()-50:a[0].max()+50, a[1].min()-50:a[1].max()+50]

                '''Compute the IOU'''
                if (cat['blend_status'].iloc[-1] == 'TP') or (cat['blend_status'].iloc[-1] == 'FP'):
                    '''IOU not blended region'''
                    wo_background_obj = np.where(gt_stamp==0, np.max(gt_stamp) + 1, gt_stamp)
                    wo_background_obj = np.where(gt_stamp==2, 1, wo_background_obj)

                    wo_blend_pred = np.where(pred_obj_stamp == 2, 1, pred_obj_stamp)
                    diff_obj = wo_background_obj - wo_blend_pred
                    intersection_obj = len(diff_obj[diff_obj==0])
                    union_obj = len(np.where(wo_background_obj[wo_background_obj == 1])[0])\
                                + len(np.where(wo_blend_pred[wo_blend_pred == 1])[0])
                    if union_obj == 0 :
                        IOU_obj = 'NaN'
                    else :
                        IOU_obj = np.round(intersection_obj / union_obj,2)

                    '''IOU blended region'''

                    wo_background_blend = np.where(((gt_stamp==0) | (gt_stamp == 1)), np.max(gt_stamp) + 1, gt_stamp)
                    wo_isol_pred = np.where(pred_obj_stamp == 1, 0, pred_obj_stamp)
                    diff_blend = wo_background_blend - wo_isol_pred
                    intersection_blend = len(diff_blend[diff_blend==0])
                    union_blend = len(np.where(wo_background_blend[wo_background_blend == 2])[0])\
                                + len(np.where(wo_isol_pred[wo_isol_pred == 2])[0])
                    if union_blend == 0 :
                        IOU_blend = 'NaN'
                    else :
                        IOU_blend = round(intersection_blend / union_blend, 2)
                    cat['IOU'].iloc[-1] = IOU_obj
                    cat['IOU_blend'].iloc[-1] = IOU_blend

                else :
                    #we put the bg to another value so it does not count in the well predicted pixels
                    wo_background = np.where(gt_stamp==0, np.max(gt_stamp) + 1, gt_stamp) 
                    diff = wo_background - pred_obj_stamp[0]
                    intersection = len(np.where(diff==0)[0])
                    union =  len(np.where((gt_stamp + pred_obj_stamp[0]) > 0)[0])
                    if union == 0 :
                        IOU = 1
                    else :
                        IOU = intersection / union
                    cat['IOU'].iloc[-1] = round(IOU, 2)

                ''' Add the variance '''
                cat['var'].iloc[-1] = round(np.mean(pred_obj_stamp[2, np.where((pred_obj_stamp[1] > 0) & (pred_obj_stamp[1] <= 1))[0], np.where((pred_obj_stamp[1] > 0) & (pred_obj_stamp[1] <= 1))[1]])
, 4)

                cat['var_blend'].iloc[-1] = round(np.mean(pred_obj_stamp[2, np.where((pred_obj_stamp[1] > 1) & (pred_obj_stamp[1] <= 2))[0], np.where((pred_obj_stamp[1] > 1) & (pred_obj_stamp[1] <= 2))[1]])
, 4)

                k += 1

                if is_show :
                    if plot < 20 :
                        if (cat['blend_status'].iloc[-1] == 'TP'):

                            fig, ax = plt.subplots(1, 6, figsize=(10,50))
                            ax[0].imshow(pred_obj_stamp[0], vmax=2)
                            ax[0].title.set_text(f"Pred \n area:{cat['area'].iloc[-1][0]},{cat['area_blend'].iloc[-1][0]}")
                            ax[1].imshow(gt_stamp, vmax = 2)
                            ax[1].title.set_text(f"gt, dist {cat['distance'].iloc[-1]} \n area:{cat['area'].iloc[-1][1]},{cat['area_blend'].iloc[-1][1]}")
                            ax[2].imshow(diff_obj[0])
                            ax[2].title.set_text(f"dif no blend, \n IOU = {cat['IOU'].iloc[-1]}")
                            ax[3].imshow(diff_blend[1])
                            ax[3].title.set_text(f"dif blend, \n IOU = {cat['IOU_blend'].iloc[-1]}")
                            ax[4].imshow(pred_obj_stamp[2])
                            ax[4].title.set_text(f"var \n var blend = {cat['var_blend'].iloc[-1]} \n iso : {cat['var'].iloc[-1]}")
                            ax[5].imshow(stamp_img[int(stamp_img.shape[0]/2-32):int(stamp_img.shape[0]/2+32),int(stamp_img.shape[1]/2-32):int(stamp_img.shape[1]/2+32)])
                            ax[5].title.set_text(f"mag = {np.round(cat['mag'].iloc[-1]),2}")
                            plot += 1

                        if (cat['blend_status'].iloc[-1] == 'FP'):

                            fig, ax = plt.subplots(1, 6, figsize=(10,50))
                            ax[0].imshow(pred_obj_stamp[0], vmax=2)
                            ax[0].title.set_text(f"Pred \n area:{cat['area'].iloc[-1][0]},{cat['area_blend'].iloc[-1][0]}")
                            ax[1].imshow(gt_stamp, vmax = 2)
                            ax[1].title.set_text(f"gt, dist {cat['distance'].iloc[-1]} \n area:{cat['area'].iloc[-1][1]},{cat['area_blend'].iloc[-1][1]}")
                            ax[2].imshow(diff_obj[0])
                            ax[2].title.set_text(f"dif no blend, \n IOU = {cat['IOU'].iloc[-1]}")
                            ax[3].imshow(diff_blend[1])
                            ax[3].title.set_text(f"dif blend, \n IOU = {cat['IOU_blend'].iloc[-1]}")
                            ax[4].imshow(pred_obj_stamp[2])
                            ax[4].title.set_text(f"var \n var blend = {cat['var_blend'].iloc[-1]} \n iso : {cat['var'].iloc[-1]}")
                            ax[5].imshow(stamp_img[int(stamp_img.shape[0]/2-32):int(stamp_img.shape[0]/2+32),int(stamp_img.shape[1]/2-32):int(stamp_img.shape[1]/2+32)])
                            ax[5].title.set_text(f"mag = {np.round(cat['mag'].iloc[-1]),2}")
                            plot += 1

                        if cat['blend_status'].iloc[-1] ==  'FN':
                            fig, ax = plt.subplots(1, 5, figsize=(10,50))
                            ax[0].imshow(pred_obj_stamp[0], vmax=2)
                            ax[0].title.set_text(f"Pred \n area:{cat['area'].iloc[-1][0]},{cat['area_blend'].iloc[-1][0]}")
                            ax[1].imshow(gt_stamp, vmax = 2)
                            ax[1].title.set_text(f"gt, dist {cat['distance'].iloc[-1]} \n area:{cat['area'].iloc[-1][1]},{cat['area_blend'].iloc[-1][1]}")
                            ax[2].imshow(diff)
                            ax[2].title.set_text(f"dif, \n IOU = {cat['IOU'].iloc[-1]}")
                            ax[3].imshow(pred_obj_stamp[2])
                            ax[3].title.set_text(f" iso : {cat['var'].iloc[-1]}")
                            ax[4].imshow(stamp_img[int(stamp_img.shape[0]/2-32):int(stamp_img.shape[0]/2+32),int(stamp_img.shape[1]/2-32):int(stamp_img.shape[1]/2+32)])
                            ax[4].title.set_text(f"mag = {np.round(cat['mag'].iloc[-1]),2}")
                            plot += 1
                          

            except:
                print("Unexpected error")
                cat = cat.drop(cat.shape[0]-1, 0)
                k += 1
                pb += 1
        print(f'there is {no_match} not matching object')
        return cat, no_match

def plot_maps(img, seg, obj, pred, pred_obj, n_column, n_row, patch_size):
    fs = 40

    plt.figure(figsize=(70,50))
    plt.imshow(img[0,0])#[:3 * 128, :20*128], cmap='flag')
    plt.xticks(np.arange(0,n_column*patch_size+256,128))
    plt.yticks(np.arange(0,n_row*patch_size+256,128))
    plt.grid()
    plt.title('Input image', fontsize=fs)

    plt.figure(figsize=(70,50))
    plt.imshow(seg)
    plt.xticks(np.arange(0,n_column*patch_size+256,128))
    plt.yticks(np.arange(0,n_row*patch_size+256,128))
    plt.grid()
    plt.title('Ground Truth Segmap', fontsize=fs)


    plt.figure(figsize=(70,50))
    plt.imshow(obj)
    plt.xticks(np.arange(0,n_column*patch_size+256,128))
    plt.yticks(np.arange(0,n_row*patch_size+256,128))
    plt.grid()
    plt.title('Ground truth separated objects', fontsize=fs)

    plt.figure(figsize=(70,50))
    plt.imshow(pred[0])
    plt.xticks(np.arange(0,n_column*patch_size+256,128))
    plt.yticks(np.arange(0,n_row*patch_size+256,128))
    plt.grid()
    plt.title('Predicted Segmap', fontsize=fs)

    plt.figure(figsize=(70,50))
    plt.imshow(pred[1])
    plt.xticks(np.arange(0,n_column*patch_size+256,128))
    plt.yticks(np.arange(0,n_row*patch_size+256,128))
    plt.grid()
    plt.title('Proba Predicted Segmap', fontsize=fs)

    plt.figure(figsize=(70,50))
    plt.imshow(pred[2])
    plt.xticks(np.arange(0,n_column*patch_size+256,128))
    plt.yticks(np.arange(0,n_row*patch_size+256,128))
    plt.grid()
    plt.title('Var Predicted Segmap', fontsize=fs)


    plt.figure(figsize=(70,50))
    plt.imshow(pred_obj)
    plt.xticks(np.arange(0,n_column*patch_size+256,128))
    plt.yticks(np.arange(0,n_row*patch_size+256,128))
    plt.grid()
    plt.title('Predicted separated objects', fontsize=fs)
    
    plt.figure(figsize=(70,50))
    plt.imshow(seg - pred[0, :seg.shape[0], :seg.shape[1]])
    plt.title("GT - prediction")
#     plt.colorbar()


def plot_species(cat, indexes, nb_2_plot, img, seg, pred):
    n = indexes.shape[0]
    if n > nb_2_plot:
        n = nb_2_plot
    
    row_size  = 5
    fig, ax = plt.subplots(n, 4, figsize=(row_size, n*((row_size+1.3)/4)), sharex=True, sharey=True)
    j = 0
    for i in indexes[:n]:

        ax[j, 0].imshow(pred[2, int(cat['x'].iloc[i]-20):int(cat['x'].iloc[i]+20),
                             int(cat['y'].iloc[i]-20):int(cat['y'].iloc[i]+20)], cmap='gray')
        ax[j, 0].set_title(f"{cat['var'].iloc[i]}, {cat['var_blend'].iloc[i]}")

        ax[j, 1].imshow(pred[0,int(cat['x'].iloc[i]-20):int(cat['x'].iloc[i]+20),
                             int(cat['y'].iloc[i]-20):int(cat['y'].iloc[i]+20)], vmax=2)
        ax[j, 1].set_title("Pred")


        ax[j, 2].imshow(seg[int(cat['x'].iloc[i]-20):int(cat['x'].iloc[i]+20),
                            int(cat['y'].iloc[i]-20):int(cat['y'].iloc[i]+20)], vmax=2)
        ax[j, 2].set_title("Groud Truth")

        ax[j, 3].imshow(img[0, 0, int(cat['x'].iloc[i]-20):int(cat['x'].iloc[i]+20),
                            int(cat['y'].iloc[i]-20):int(cat['y'].iloc[i]+20)])
        ax[j, 2].set_title("Image")

        j += 1

def score_per_bin(classe, cat, residual_cat, param, start, stop, step, tot_completness_iso, tot_completness_blend, tot_IOU_iso,  tot_IOU_blend, tot_IOU_blend_iso, newcmp, is_show=True):
    completness = []
    iou = []
    iou_blend = []
    iou_blend_ = 0
    x = np.arange(start, stop, step)
    nb_per_bin = []
    for i in x:
#         try :
            if classe == 'iso':
                # conditions to be TP_obj : be TP_obj, not be TP_blend, and  i < mag < i+step
                nb_TP = len(cat[(cat['obj_status'] == 'TP') & (cat['blend_status'] == 'TN') & (cat[param] >= i) & (cat[param] < i+step)])        # conditions to be FN_obj : be FN_obj, not be a blend : and  i < mag < i+step
                nb_FN = len(residual_cat[(residual_cat['blend_status'] == 'N') & (i <= residual_cat[param]) & (residual_cat[param] <= i+step)])
                iou_ = np.mean(cat['IOU'][cat[(cat['obj_status'] == 'TP') & (cat['blend_status'] == 'TN') & (cat[param] >= i) & (cat[param] < i + step)].index])                

            elif classe == 'blend':
                nb_TP = len(cat[(cat['blend_status'] == 'TP') & (cat[param] >= i) & (cat[param] < i+step)])        # conditions to be FN_obj : be FN_obj, not be a blend : and  i < mag < i+step
                nb_FN = len(cat[(cat['blend_status'] == 'FN') & (cat[param] >= i) & (cat[param] < i+step)])
                iou_ = np.mean(cat['IOU'][cat[(cat['blend_status'] == 'TP') & (cat[param] >= i) & (cat[param] < i + step)].index])
                iou_blend_ = np.mean(cat['IOU_blend'][cat[(cat['blend_status'] == 'TP') & (cat['mag'] >= i) & (cat['mag'] < i + step)].index])

                
            nb_per_bin.append(nb_TP + nb_FN)
            if nb_TP == 0:
                completness.append(0)
                iou.append(0)
                iou_blend.append(0)
                continue
            completness.append(nb_TP/(nb_TP + nb_FN))
            iou.append(iou_)
            iou_blend.append(iou_blend_)
#         except:
#             print("Unexpected error:", sys.exc_info()[0])
#             completness.append(0)
#             iou.append(0)
    completness, iou = np.asarray(completness), np.asarray(iou)
    
    if classe == 'iso':
        tot_completness = tot_completness_iso
        tot_iou = tot_IOU_iso
        nb_fig = 2
    if classe == 'blend':
        tot_completness = tot_completness_blend
        tot_iou = tot_IOU_blend_iso
        tot_iou_blend = tot_IOU_blend
        nb_fig = 3

#     with plt.xkcd():
    fig, ax = plt.subplots(1, nb_fig, figsize=(20, 10))
    a = ax[0].scatter(x, completness, c=nb_per_bin, cmap=newcmp)
    plt.colorbar(a, ax = ax[0], fraction=0.06, pad=0.04,)
    ax[0].set_title(f"Completness : {tot_completness}")

    a = ax[1].scatter(x, iou, c=nb_per_bin, cmap=newcmp)
    plt.colorbar(a, ax = ax[1], fraction=0.06, pad=0.04)
    ax[1].set_title(f"IOU: {tot_iou}")
    
    if classe == 'blend':
        a = ax[2].scatter(x, iou_blend, c=nb_per_bin, cmap=newcmp)
        plt.colorbar(a, ax = ax[2], fraction=0.06, pad=0.04)
        ax[2].set_title(f"IOU: {tot_iou_blend}")

    return completness, iou, nb_per_bin




# def produce_cat(img, gt, pred, gt_obj_map, pred_obj_map, gt_cat, is_score=True, is_show=None):
    
#         cat = pd.DataFrame(columns=['TU_Index', 'TU_Index_blend', 'x', 'y', 'obj_status', 'blend_status', 'mag', 'rad', 'IOU', 'IOU_blend', 'var', 'var_blend', 'loc'])
#         sz = 80    # size of the stamp to compute the IOU on
#         mid = 40
#         index = -1   # index of the object in the catalogue
#         dr = 5       # epsilon to try to match the galaxy
#         k = 1        # object number skip the background "object"
#         no_match = 0 
#         seen_ = 0
#         pb = 0
#         plot = 0

#         while k <= pred_obj_map.max():
#             if k % 100 ==0:
#                 print(f'k ={k}')
#             index += 1    #index for the predicted catalogue

#             '''create the stamp of the objects'''
#             pred_obj_stamp = np.zeros((3, sz, sz))
#             gt_stamp = np.zeros((sz, sz))

#             try:
#                 '''detect the k-th object'''
#                 a = np.where(pred_obj_map == k)      # coords of all the pixels belonging to the kth object
#                 if len(a[0]) == 0:
#                     print(f"{k}-th object is null")
#                     k += 1
#                     continue
#                 sizex, sizey = a[0].max()+1 -a[0].min(), a[1].max()+1 -a[1].min()   # corresponding size of the galaxy (square)

#                 cat = cat.append(pd.Series(), ignore_index=True)

#                 ''' Add the location to the catalogue to do the better detection'''
#                 cat['loc'].iloc[-1] = [a[0].min(), a[0].max(), a[1].min(), a[1].max()]

#                 ''' Add the coords to the catalogue (for now, the centroid is just the middle of the stamp)'''
#                 cat['x'].iloc[-1] = int(np.mean(a[0]))
#                 cat['y'].iloc[-1] = int(np.mean(a[1]))

#                 if not is_score :
#                     k += 1
#                     continue

#                 pred_obj_stamp[:, int(mid-sizex/2):int(mid+sizex/2), int(mid-sizey/2):int(mid+sizey/2)] = pred[:, a[0].min():a[0].max()+1, a[1].min():a[1].max()+1]

#                 ''' Macth the object(s) of the TU catalogue : we look in x+/-dr, y+-dr and see if there is a galaxy 
#                     in this interval in the TU cat (x/y are inverted compared to numpy...) '''

#                 indexes_tu = np.where((gt_cat[:, 3] > a[0].min()- dr) & (gt_cat[:, 3] < a[0].max()+dr) & (gt_cat[:, 2] > a[1].min()-dr) & (gt_cat[:, 2] < a[1].max()+dr))[0]
#         #         print(f'nb of matching gal : {len(index_tu)}')

#                 if len(indexes_tu) == 0:      # there is no matching index
#                     index_tu = -1           # faux positif

#                 else:
#                     index_tu = indexes_tu[0]
#         #         display(cat)

#                 ''' Add the Detection flag'''
#                 if index_tu == -1:      # False positive
#                     cat['obj_status'].iloc[-1] = 'FP'   # False positive for Detection flag
#                     k += 1
#                     if pred_obj_stamp.max() > 1:
#                         cat['blend_status'].iloc[-1] = 'P'
#                     else:
#                         cat['blend_status'].iloc[-1] = 'N'
#                     continue

#                 cat['obj_status'].iloc[-1] = 'TP'

#                 ''' Isolate the matching true object : same method but with the coordinates of the matching object '''

#                 a = np.where(gt_obj_map == gt_obj_map[int(gt_cat[index_tu, 3])-2:int(gt_cat[index_tu, 3])+2, int(gt_cat[index_tu, 2])-2:int(gt_cat[index_tu, 2])+2].max())
#         #             print('2', a[0].min(), a[0].max())
#                 if gt_obj_map[int(gt_cat[index_tu, 3])-2:int(gt_cat[index_tu, 3])+2, int(gt_cat[index_tu, 2])-2:int(gt_cat[index_tu, 2])+2].max() == 0:
#                     print(" Matching object does not exist")
#                     cat = cat.drop(cat.shape[0]-1, 0)
#                     k += 1
#                     no_match += 1
#                     continue

#                 sizex = a[0].max()+1 -a[0].min()
#                 sizey = a[1].max()+1 -a[1].min()

#         #             gt_stamp[int(mid-sizex/2):int(mid+sizex/2), int(mid-sizey/2):int(mid+sizey/2)] = gt_obj_map[a[0].min():a[0].max()+1, a[1].min():a[1].max()+1]
#                 gt_stamp[int(mid-sizex/2):int(mid+sizex/2), int(mid-sizey/2):int(mid+sizey/2)] = gt[a[0].min():a[0].max()+1, a[1].min():a[1].max()+1]


#                 '''Check if the object is a TP(1)/FP(2)/TN(3)/FN(4) blend'''

#                 if gt_stamp.max() > 1 and pred_obj_stamp[0].max() > 1:
#                     cat['blend_status'].iloc[-1] = 'TP'
#                     indexes_tu = np.where((gt_cat[:, 3] > a[0].min()) & (gt_cat[:, 3] < a[0].max()) & (gt_cat[:, 2] > a[1].min()) & (gt_cat[:, 2] < a[1].max()))[0]
#                     cat['TU_Index'].iloc[-1] = int(gt_cat[indexes_tu[0], 0])
#                     cat['TU_Index_blend'].iloc[-1] = int(gt_cat[indexes_tu[1], 0])
#                     cat['mag'].iloc[-1] = round(gt_cat[indexes_tu[0], 4], 2)
#                     cat['rad'].iloc[-1] = round(gt_cat[indexes_tu[0], 6], 2)
#         #             k += 1
#                 if gt_stamp.max() == 1 and pred_obj_stamp[0].max() > 1:
#                     cat['TU_Index'].iloc[-1] = int(gt_cat[index_tu, 0])
#                     cat['blend_status'].iloc[-1] = 'FP'
#                     cat['mag'].iloc[-1] = round(gt_cat[index_tu, 4])
#                     cat['rad'].iloc[-1] = round(gt_cat[index_tu, 6])
#         #             k += 1
#                 if gt_stamp.max() == 1 and pred_obj_stamp[0].max() == 1:
#                     cat['TU_Index'].iloc[-1] = int(gt_cat[index_tu, 0])
#         #             print(tu_cat[index_tu, 4])
#                     cat['mag'].iloc[-1] = round(gt_cat[index_tu, 4])
#                     cat['rad'].iloc[-1] = round(gt_cat[index_tu, 6])
#                     cat['blend_status'].iloc[-1] = 'TN'
#                     blend_status = 'TN'
#         #             k += 1
#                 if gt_stamp.max() > 1 and pred_obj_stamp[0].max() == 1:
#                     cat['blend_status'].iloc[-1] = 'FN'
#                     indexes_tu = np.where((gt_cat[:, 3] > a[0].min()- dr) & (gt_cat[:, 3] < a[0].max()+dr) & (gt_cat[:, 2] > a[1].min()-dr) & (gt_cat[:, 2] < a[1].max()+dr))[0]
#                     if gt_cat[indexes_tu[0],0] in cat['TU_Index'].values:
#                         seen_ +=1
#                         place = 1
#                         other = 0
#                     else:
#                         place = 0
#                         other = 1
#                     cat['TU_Index'].iloc[-1] = int(gt_cat[indexes_tu[place], 0])
#                     cat['TU_Index_blend'].iloc[-1] = int(gt_cat[indexes_tu[other], 0])
#                     cat['mag'].iloc[-1] = round(gt_cat[indexes_tu[place], 4], 2)
#                     cat['rad'].iloc[-1] = round(gt_cat[indexes_tu[place], 6], 2)

#                 stamp_img = img[0, 0, a[0].min()-50:a[0].max()+50, a[1].min()-50:a[1].max()+50]

#                 '''Compute the IOU'''
#                 if (cat['blend_status'].iloc[-1] == 'TP') or (cat['blend_status'].iloc[-1] == 'FP'):
#                     '''IOU not blended region'''
#                     wo_background_obj = np.where(gt_stamp==0, np.max(gt_stamp) + 1, gt_stamp)
#                     wo_background_obj = np.where(gt_stamp==2, 1, wo_background_obj)

#                     wo_blend_pred = np.where(pred_obj_stamp == 2, 1, pred_obj_stamp)
#                     diff_obj = wo_background_obj - wo_blend_pred
#                     intersection_obj = len(diff_obj[diff_obj==0])
#                     union_obj = len(np.where(wo_background_obj[wo_background_obj == 1])[0])\
#                                 + len(np.where(wo_blend_pred[wo_blend_pred == 1])[0])
#                     if union_obj == 0 :
#                         IOU_obj = 'NaN'
#                     else :
#                         IOU_obj = np.round(intersection_obj / union_obj,2)

#                     '''IOU blended region'''

#                     wo_background_blend = np.where(((gt_stamp==0) | (gt_stamp == 1)), np.max(gt_stamp) + 1, gt_stamp)
#                     wo_isol_pred = np.where(pred_obj_stamp == 1, 0, pred_obj_stamp)
#                     diff_blend = wo_background_blend - wo_isol_pred
#                     intersection_blend = len(diff_blend[diff_blend==0])
#                     union_blend = len(np.where(wo_background_blend[wo_background_blend == 2])[0])\
#                                 + len(np.where(wo_isol_pred[wo_isol_pred == 2])[0])
#                     if union_blend == 0 :
#                         IOU_blend = 'NaN'
#                     else :
#                         IOU_blend = round(intersection_blend / union_blend, 2)
#                     cat['IOU'].iloc[-1] = IOU_obj
#                     cat['IOU_blend'].iloc[-1] = IOU_blend

#                 else :
#                     #we put the bg to another value so it does not count in the well predicted pixels
#                     wo_background = np.where(gt_stamp==0, np.max(gt_stamp) + 1, gt_stamp) 
#                     diff = wo_background - pred_obj_stamp[0]
#                     intersection = len(np.where(diff==0)[0])
#                     union =  len(np.where((gt_stamp + pred_obj_stamp[0]) > 0)[0])
#                     if union == 0 :
#                         IOU = 1
#                     else :
#                         IOU = intersection / union
#                     cat['IOU'].iloc[-1] = round(IOU, 2)

#                 ''' Add the variance '''
#                 cat['var'].iloc[-1] = round(np.mean(pred_obj_stamp[2, np.where((pred_obj_stamp[1] > 0) & (pred_obj_stamp[1] <= 1))[0], np.where((pred_obj_stamp[1] > 0) & (pred_obj_stamp[1] <= 1))[1]])
# , 4)

#                 cat['var_blend'].iloc[-1] = round(np.mean(pred_obj_stamp[2, np.where((pred_obj_stamp[1] > 1) & (pred_obj_stamp[1] <= 2))[0], np.where((pred_obj_stamp[1] > 1) & (pred_obj_stamp[1] <= 2))[1]])
# , 4)

#                 k += 1

#                 if is_show :
#                     if plot < 20 :
#                         if (cat['blend_status'].iloc[-1] == 'TP'):

#                             fig, ax = plt.subplots(1, 6, figsize=(10,50))
#                             ax[0].imshow(pred_obj_stamp[0], vmax=2)
#                             ax[0].title.set_text(f"Prediction \n obj : {cat['obj_status'].iloc[-1]}, blend : {cat['blend_status'].iloc[-1]}")
#                             ax[1].imshow(gt_stamp, vmax = 2)
#                             ax[1].title.set_text(f'ground_truth')
#                             ax[2].imshow(diff_obj[0])
#                             ax[2].title.set_text(f"diff non blended, \n IOU = {cat['IOU'].iloc[-1]}")
#                             ax[3].imshow(diff_blend[1])
#                             ax[3].title.set_text(f"diff blended, \n IOU = {cat['IOU_blend'].iloc[-1]}")
#                             ax[4].imshow(pred_obj_stamp[2])
#                             ax[4].title.set_text(f"var \n var blend = {cat['var_blend'].iloc[-1]} \n iso : {cat['var'].iloc[-1]}")
#                             ax[5].imshow(stamp_img[int(stamp_img.shape[0]/2-32):int(stamp_img.shape[0]/2+32),int(stamp_img.shape[1]/2-32):int(stamp_img.shape[1]/2+32)])
#                             ax[5].title.set_text(f"mag = {np.round(cat['mag'].iloc[-1]),2}")
#                             plot += 1

#                         if (cat['blend_status'].iloc[-1] == 'FP'):

#                             fig, ax = plt.subplots(1, 6, figsize=(10,50))
#                             ax[0].imshow(pred_obj_stamp[0], vmax=2)
#                             ax[0].title.set_text(f"Prediction \n obj : {cat['obj_status'].iloc[-1]}, blend : {cat['blend_status'].iloc[-1]}")
#                             ax[1].imshow(gt_stamp, vmax = 2)
#                             ax[1].title.set_text(f'ground_truth')
#                             ax[2].imshow(diff_obj[0])
#                             ax[2].title.set_text(f"diff non blended, \n IOU = {cat['IOU'].iloc[-1]}")
#                             ax[3].imshow(diff_blend[1])
#                             ax[3].title.set_text(f"diff blended, \n IOU = {cat['IOU_blend'].iloc[-1]}")
#                             ax[4].imshow(pred_obj_stamp[2])
#                             ax[4].title.set_text(f"var \n var blend = {cat['var_blend'].iloc[-1]} \n iso : {cat['var'].iloc[-1]}")
#                             ax[5].imshow(stamp_img[int(stamp_img.shape[0]/2-32):int(stamp_img.shape[0]/2+32),int(stamp_img.shape[1]/2-32):int(stamp_img.shape[1]/2+32)])
#                             ax[5].title.set_text(f"mag = {np.round(cat['mag'].iloc[-1]),2}")
#                             plot += 1

#                         if cat['blend_status'].iloc[-1] ==  'FN':
#                             fig, ax = plt.subplots(1, 5, figsize=(10,50))
#                             ax[0].imshow(pred_obj_stamp[0], vmax=2)
#                             ax[0].title.set_text(f"Prediction \n obj : {cat['obj_status'].iloc[-1]}, blend : {cat['blend_status'].iloc[-1]}")
#                             ax[1].imshow(gt_stamp, vmax = 2)
#                             ax[1].title.set_text(f'ground_truth')
#                             ax[2].imshow(diff)
#                             ax[2].title.set_text(f"diff, \n IOU = {cat['IOU'].iloc[-1]}")
#                             ax[3].imshow(pred_obj_stamp[2])
#                             ax[3].title.set_text(f" iso : {cat['var'].iloc[-1]}")
#                             ax[4].imshow(stamp_img[int(stamp_img.shape[0]/2-32):int(stamp_img.shape[0]/2+32),int(stamp_img.shape[1]/2-32):int(stamp_img.shape[1]/2+32)])
#                             ax[4].title.set_text(f"mag = {np.round(cat['mag'].iloc[-1]),2}")

#             except:
#                 print("Unexpected error")
#                 cat = cat.drop(cat.shape[0]-1, 0)
#                 k += 1
#                 pb += 1
#         print(f'there is {no_match} not matching object')
#         return cat, no_match
