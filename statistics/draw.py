#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: draw.py
# @Author: Li Chengxin 
# @Time: 2022/6/22 2:38


import numpy as np
import matplotlib.pyplot as plt
from statistics_detail import RefinedParser

path = r'G:\20x_dataset\copy_of_xy_16\refined.xlsx'

rp = RefinedParser(path)

data_no_sort = np.array(rp.export_result()[0])
data_sort = np.array(rp.export_result()[1])

data = data_sort
# data = data_no_sort

index = np.arange(len(data))

y_M1 = data[:, 0]
y_G1 = data[:, 1]
y_S = data[:, 2]
y_G2 = data[:, 3]
y_M2 = data[:, 4]

plt.figure(figsize=(20, 30))

# plt.bar(index, y_M1, width=1, color='r', label='M')
# plt.bar(index, y_G1, width=1, bottom = y_M1, color = 'g', label = 'G1')
# plt.bar(index, y_S, width=1, bottom = (y_M1+y_G1), color = 'b', label = 'S')
# plt.bar(index, y_G2, width=1, bottom = (y_M1+y_G1+y_S), color = 'pink', label = 'G2')
# plt.bar(index, y_M2, width=1, bottom = (y_M1+y_G1+y_S+y_G2), color = 'r', label = 'M')

h = 1

plt.barh(index, y_M1, height=h, color='r', label='M')
plt.barh(index, y_G1, height=h, left = y_M1, color = 'g', label = 'G1')
plt.barh(index, y_S, height=h, left = (y_M1+y_G1), color = 'b', label = 'S')
plt.barh(index, y_G2, height=h, left = (y_M1+y_G1+y_S), color = '#FFA500', label = 'G2')
plt.barh(index, y_M2, height=h, left = (y_M1+y_G1+y_S+y_G2), color = 'r')

plt.legend(ncol=4, loc='lower center')
plt.savefig(r'G:\20x_dataset\copy_of_xy_16\cycle.pdf')
plt.show()
