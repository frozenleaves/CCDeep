#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: evaluate.py
# @Author: Li Chengxin 
# @Time: 2022/4/20 17:29

"""
用于评估和检验输出模型的预测能力
"""

import os
import numpy as np
import tensorflow as tf
import config
from prepare_data import get_dataset
from train import get_model
import cv2


def read_img(dic_img_path, mcy_img_path):
    dic_img = cv2.imread(dic_img_path, -1)
    mcy_img = cv2.imread(mcy_img_path, -1)
    img = np.dstack([cv2.resize(dic_img, (config.image_width, config.image_height)),
                     cv2.resize(mcy_img, (config.image_width, config.image_height))])
    return img / 255


if __name__ == '__main__':

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the test dataset
    test_dataset, test_count = get_dataset(dataset_root_dir_mcy=config.test_dir_mcy_20x)
    test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE)

    # load the model
    model = get_model()
    model.load_weights(filepath=config.save_model_dir_20x_best)

    # Get the accuracy on the test set
    loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)


    for test_images, test_labels in test_dataset:
        test_step(test_images, test_labels)
        print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
                                                           test_accuracy.result()))

    print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result()*100))

    dic_root_S = '/home/zje/CellClassify/train_dataset/train_data_20x_dev1.3.2/train_dic/test/S/'
    mcy_root_S = '/home/zje/CellClassify/train_dataset/train_data_20x_dev1.3.2/train_mcy/test/S/'
    dic_root_G = '/home/zje/CellClassify/train_dataset/train_data_20x_dev1.3.2/train_dic/test/G/'
    mcy_root_G = '/home/zje/CellClassify/train_dataset/train_data_20x_dev1.3.2/train_mcy/test/G/'
    dic_root_M = '/home/zje/CellClassify/train_dataset/train_data_20x_dev1.3.2/train_dic/test/M/'
    mcy_root_M = '/home/zje/CellClassify/train_dataset/train_data_20x_dev1.3.2/train_mcy/test/M/'

    dic_G = [os.path.join(dic_root_G, x) for x in os.listdir(dic_root_G)]
    mcy_G = [os.path.join(mcy_root_G, x) for x in os.listdir(mcy_root_G)]
    dic_M = [os.path.join(dic_root_M, x) for x in os.listdir(dic_root_M)]
    mcy_M = [os.path.join(mcy_root_M, x) for x in os.listdir(mcy_root_M)]
    dic_S = [os.path.join(dic_root_S, x) for x in os.listdir(dic_root_S)]
    mcy_S = [os.path.join(mcy_root_S, x) for x in os.listdir(mcy_root_S)]

    d = '/home/zje/CellClassify/train_dataset/train_data_20x_dev1.3.2/train_dic/test/S/7d0eb6678a0bbea78961133b0d6552ab.tif'
    m = '/home/zje/CellClassify/train_dataset/train_data_20x_dev1.3.2/train_mcy/test/S/7d0eb6678a0bbea78961133b0d6552ab.tif'
    img = read_img(d, m)
    print(img.dtype)
    predictions = model(tf.convert_to_tensor(np.array([img]), dtype=tf.float64), training=False)
    print('single predict: ', predictions)
    true = 0
    step = 1
    predict_img = []
    for i in zip(dic_G, mcy_G):
        predict_img.append(read_img(i[0], i[1]))
        step += 1
        if step > 100:
            break
    for i in zip(dic_S, mcy_S):
        predict_img.append(read_img(i[0], i[1]))
        step += 1
        if step > 150:
            break
    for i in zip(dic_M, mcy_M):
        predict_img.append(read_img(i[0], i[1]))
        step += 1
        if step > 200:
            break
    dt = tf.convert_to_tensor(np.array(predict_img))
    print(dt.shape)
    predictions = model(dt, training=False)
    print(predictions)
    rets = []
    for i in predictions:
        rets.append(np.argwhere(i == np.max(i))[0][0])
    print(rets)
    print('all', len(rets))
    print('G', rets.count(0))
    print('M', rets.count(1))
    print('S', rets.count(2))
    #     if ret == 1:
    #         true +=1
    #         print(f'\rM accuracy: {true/step} test step: {step}', end='')
    #     step += 1

    # for test_images, test_labels in test_dataset:
    #     print(test_labels)
    #     print(test_images.shape)
    #     predictions = model(test_images, training=False)
    #     rets = []
    #     for i in predictions:
    #         rets.append(np.argwhere(i == np.max(i))[0][0])
    #     print(rets)
