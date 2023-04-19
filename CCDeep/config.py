#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: config.py
# @Author: Li Chengxin 
# @Time: 2022/4/18 15:44

"""
项目的一些基础参数配置， 包括训练数据集的存放位置，模型的存放位置等
"""


from __future__ import annotations
import os
import time


TIMES = 20  # Image magnification

AUGMENTATION_IN_TRAINING = False

RAW_INPUT_IMAGE_SIZE = (2048, 2048)  # predict image size

# some training parameters
EPOCHS = 100
BATCH_SIZE = 128
NUM_CLASSES = 3  # cell phase num
image_height = 100
image_width = 100
LEARNING_RATE = 1e-6
channels = 2  # image channels


raw = r'H:\CCDeep-data\raw-data\train\raw'   # 训练原始数据集

train_process_20x_detail_data_savefile = f'./logs/{time.ctime()}train_detail_20x.csv'
train_process_60x_detail_data_savefile = f'./logs/{time.ctime()}train_detail_60x.csv'

save_model_dir_60x = '../models/classify/60x/model'
dataset_dir_mcy_60x = '/home/zje/CellClassify/train_dataset/train_data_60x/train_mcy'
train_dir_mcy_60x = os.path.join(dataset_dir_mcy_60x, "train")
valid_dir_mcy_60x = os.path.join(dataset_dir_mcy_60x, "valid")
test_dir_mcy_60x = os.path.join(dataset_dir_mcy_60x, "test")
dataset_dir_dic_60x = '/home/zje/CellClassify/train_dataset/train_data_60x/train_dic'
train_dir_dic_60x = os.path.join(dataset_dir_dic_60x, "train")
valid_dir_dic_60x = os.path.join(dataset_dir_dic_60x, "valid")
test_dir_dic_60x = os.path.join(dataset_dir_dic_60x, "test")

# save_model_dir_20x = './models/classify/20x/final/model'
# save_model_dir_20x_best = './models/classify/20x/best/model'

save_model_dir_20x = './models/classify/20x/final-dev/model'
save_model_dir_20x_best = './models/classify/20x/best-dev/model'

dataset_dir_20x = r'H:\CCDeep-data\raw-data\train\classification'
# dataset_dir_20x = r'F:\projects\CellClassify\CCDeep\CCDeep_train_data\classify'
# dataset_dir_20x = r'H:\CCDeep-data\raw-data\train\debug_dataset'

dataset_dir_mcy_20x = os.path.join(dataset_dir_20x, 'mcy')
dataset_dir_dic_20x = os.path.join(dataset_dir_20x, 'dic')

# dataset_dir_mcy_20x = './CCDeep_train_data/classify/train_mcy'
train_dir_mcy_20x = os.path.join(dataset_dir_mcy_20x, "train")
valid_dir_mcy_20x = os.path.join(dataset_dir_mcy_20x, "valid")
test_dir_mcy_20x = os.path.join(dataset_dir_mcy_20x, "test")
# dataset_dir_dic_20x = './CCDeep_train_data/classify/train_dic'
train_dir_dic_20x = os.path.join(dataset_dir_dic_20x, "train")
valid_dir_dic_20x = os.path.join(dataset_dir_dic_20x, "valid")
test_dir_dic_20x = os.path.join(dataset_dir_dic_20x, "test")

# segmentation model config

# 60x
segment_model_name_60x = 'segment_60x_model'
segment_model_saved_dir_60x = '/home/zje/CellClassify/saved_models/saved_60x_segment_model/'
tain_dataset = '/home/zje/CellClassify/train_dataset/segment_train_60x/train/images'
train_label = '/home/zje/CellClassify/train_dataset/segment_train_60x/train/masks'
valid_size = 0.1

# 20x
segment_model_name_20x = 'segment_20x_model'
segment_model_saved_dir_20x = './models/segment/20x/'
train_dataset_20x = './CCDeep_train_data/segment/train/images/'
train_label_20x = './CCDeep_train_data/segment/train/masks/'

valid_size_20x = 0.1

# choose a network
# model = "resnet18"
# model = "resnet34"
model = "resnet50"
# model = "resnet101"
# model = "resnet152"
