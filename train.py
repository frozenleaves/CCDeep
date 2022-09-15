#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: train.py
# @Author: Li Chengxin 
# @Time: 2022/7/6 5:03

import argparse
from CCDeep import train_segment, train_classify, evaluate

parser = argparse.ArgumentParser(description="Welcome to use CCDeep!", add_help=False)
parser.add_argument('-s', '--segment', action='store_true', default=False, help="train segmentation model")
parser.add_argument('-c', '--classify', action='store_true', default=False, help="train classification model")
parser.add_argument('-e', '--evaluate', action='store_true', default=False, help="evaluate classification model acuuracy")

args = parser.parse_args()

if args.segment:
    train_segment.train()

if args.classify:
    train_classify.train()

if args.evaluate:
    evaluate.evaluate()
