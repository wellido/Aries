import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Model


sys.path.append(".")
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
from torchvision import transforms
from tqdm import trange

# from FD.lenet import Net
# from learn.utils import MNIST

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x


def get_activations(test_data, model, batch_size=50, dims=2048,
                    cuda=False, overlap=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FD score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    (x_train, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_test = y_test.reshape(1, -1)
    x_train = x_train.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)

    if overlap:
        x_test = test_data.astype('float32') / 255
        # x_test -= x_train_mean

    else:
        x_test = x_test.astype('float32') / 255
        # x_test -= x_train_mean


    n_batches = len(x_test) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        batch = x_test[start: end]
        # print(batch.shape)
        pred = model.predict(batch)
        pred_arr[start: end] = pred.reshape(batch_size, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(test_data, model, batch_size=50,
                                    dims=2048, cuda=False, overlap=False):
    """Calculation of the statistics used by the FD.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(test_data, model, batch_size, dims, cuda, overlap)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act


def _compute_statistics_of_path(test_data, model, batch_size, dims, cuda, overlap):
    m, s, act = calculate_activation_statistics(test_data, model, batch_size,
                                                dims, cuda, overlap)

    return m, s, act


def calculate_fid_given_paths(test_data, model, batch_size, cuda, dims):
    """Calculates the FD of two paths"""
    m2, s2, act2 = _compute_statistics_of_path(test_data, model, batch_size,
                                               dims, cuda, overlap=True)

    return m2, s2, act2



if __name__ == '__main__':

    model_path = "../../models/cifar10/VGG16.h5"
    model = tf.keras.models.load_model(model_path)
    inter_model = Model(inputs=model.input, outputs=model.layers[-3].output)
    # test_dirs = sorted(os.listdir('./dataset_bg'))
    feat_path = 'inter_data/dataset_feature/'
    data_base = "cifar_data/new_data_"
    fd_bg = []
    '''
    training dataset (overlap=False--> source dataset)
    test dataset (overlap=True--> sample set)
    '''
    # training dataset (overlap=False--> source dataset)
    batch_size = 50
    # ResNet20
    # dims = 64

    # VGG16
    dims = 256

    m1, s1, act1 = _compute_statistics_of_path('', inter_model, batch_size,
                                               dims, 0, overlap=False)
    data_types = ['brightness',	'contrast',	'defocus_blur',	'elastic_transform', 'fog',	'frost',
                  'gaussian_blur', 'jpeg_compression', 'motion_blur', 'pixelate', 'shot_noise',
                  'snow', 'zoom_blur']

    for i in trange(len(data_types)):
        test_data = np.load("/home/qhu/qhu-data/Quan_study/datasets/cifar10/CIFAR-10-C/" + data_types[i] + ".npy")[:10000]
        # test dataset (overlap=True--> sample set)
        m2, s2, act2 = calculate_fid_given_paths(test_data,
                                                 inter_model,
                                                 batch_size,
                                                 0,
                                                 dims)

        fd_value = calculate_frechet_distance(m1, s1, m2, s2)
        print('FD: ', fd_value)
        fd_bg.append(fd_value)
        # saving features for nn regression
        np.save(feat_path + 'new_data_%s_mean' % (data_types[i]), m2)
        np.save(feat_path + 'new_data_%s_variance' % (data_types[i]), s2)
        np.save(feat_path + 'new_data_%s_feature' % (data_types[i]), act2)

    np.save('inter_data/dataset_feature/fd_cifar10_vgg16_4test.npy', fd_bg)
