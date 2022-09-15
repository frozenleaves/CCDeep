# -*- coding: utf-8 -*-
# @FileName: train_segment.py
# @Author: Li Chengxin 
# @Time: 2022/6/30 13:49

from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from glob import glob
from tqdm import tqdm
from tifffile import imread
from csbdeep.utils import Path, normalize
from stardist import fill_label_holes, random_label_cmap
from stardist.models import Config2D, StarDist2D
from CCDeep import config

np.random.seed(42)
lbl_cmap = random_label_cmap()


def prepare_data():
    """prepare segment training and valid data"""
    X = sorted(glob(config.train_dataset_20x + '/*tif'))
    Y = sorted(glob(config.train_label_20x + '/*tif'))
    assert all(Path(x).name == Path(y).name for x, y in zip(X, Y))
    X = list(map(imread, X))
    Y = list(map(imread, Y))
    axis_norm = (0, 1)
    X = [normalize(x, 1, 99.8, axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = int(len(X) * config.valid_size)
    ind_val, ind_train = ind[:n_val], ind[n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train]
    return X_val, Y_val, X_trn, Y_trn


def random_fliprot(img, mask):
    """image augmentation step"""
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim)))
    mask = mask.transpose(perm)
    for ax in axes:
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask


def random_intensity_change(img):
    """image augmentation step"""
    img = img * np.random.uniform(0.6, 2) + np.random.uniform(-0.2, 0.2)
    return img


def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02 * np.random.uniform(0, 1)
    x = x + sig * np.random.normal(0, 1, x.shape)
    return x, y


def train(export_to_fiji=False):
    conf = Config2D(n_channel_in=1, train_batch_size=8, train_shape_completion=False)
    conf.train_patch_size = 512, 512
    conf.train_completion_crop = 64
    vars(conf)
    X_val, Y_val, X_trn, Y_trn = prepare_data()
    model = StarDist2D(conf, name=config.segment_model_name_20x, basedir=config.segment_model_saved_dir_20x)
    model.train(X_trn, Y_trn, validation_data=(X_val, Y_val), augmenter=augmenter)
    model.optimize_thresholds(X_val, Y_val)
    if export_to_fiji:
        model.export_TF()

