# 评估不同周期像素值强度
import os.path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from matplotlib import pyplot as plot  # 用来绘制图形
from mpl_toolkits.mplot3d import Axes3D  # 用来给出三维坐标系。
import seaborn as sns


def getZ(img, X, Y):
    gray = img[X, Y]
    return gray


def plot_intensity_3d(image):
    height, width = image.shape[:2]
    figure = plot.figure()
    # 画出三维坐标系：
    axes = Axes3D(figure)
    X = np.arange(0, height, 1)
    Y = np.arange(0, width, 1)
    # 生成二维的底部网格线：
    X, Y = np.meshgrid(X, Y)
    # Z = 3 * (X) ** 2 + 2 * (Y) ** 2 + 5
    # 绘制曲面：
    axes.plot_surface(X, Y, getZ(image, X, Y), cmap='rainbow')
    # 显示图形：
    plot.show()

def plot_intensity_heatmap(image):
    fig, ax = plt.subplots()
    sns.heatmap(image, cmap="Blues", fmt='g')
    ax.set(yticklabels=[])  # remove the tick labels
    ax.set(xticklabels=[])  # remove the tick labels
    ax.tick_params(left=False)  # remove the ticks
    ax.tick_params(bottom=False)  # remove the ticks
    plt.show()

def imread(file):
    return tifffile.imread(file)

def image_info(image: np.ndarray):
    max_value = np.max(image)
    mean_value = np.mean(image)
    max_count = np.count_nonzero(image > 200)
    return max_value, mean_value, max_count


def draw_lineage(files_list):
    for files in files_list:
        images = map(imread, files)
        results = list(map(image_info, images))
        max_value = [i[0] for i in results]
        mean_value = [i[1] for i in results]
        max_count = [i[2] for i in results]
        plt.plot(mean_value, label='mean')
    # plt.plot(max_count, label='light count')
    plt.legend()
    plt.show()


def get_file_list(path):
    return [os.path.join(path, i) for i in os.listdir(path)]


def group_cell_line(cell_line_path):
    """将cell line按照周期分组"""
    G = []
    M = []
    S = []
    files_dic = get_file_list(os.path.join(cell_line_path, 'dic'))
    files_mcy = get_file_list(os.path.join(cell_line_path, 'mcy'))
    for i in zip(files_dic, files_mcy):
        if '-G' in i[0]:
            G.append(i)
        if '-S' in i[0]:
            S.append(i)
        if '-M' in i[0]:
            M.append(i)
    return G, S, M


def get_test_data():
    """"""


file_S = r'G:\20x_dataset\copy_of_xy_01\development-dir\cell_lines\2\mcy\2-0-0-S.tif'
file_G = r'G:\20x_dataset\copy_of_xy_01\development-dir\cell_lines\2\mcy\2-0-17-G.tif'
file_M = r'G:\20x_dataset\copy_of_xy_01\development-dir\cell_lines\2\mcy\2-0-77-M.tif'

# plot_intensity_3d(imread(file_S))
# plot_intensity_3d(imread(file_G))
# plot_intensity_3d(imread(file_M))

# plot_intensity_heatmap(imread(file_G))
# plot_intensity_heatmap(imread(file_M))

d1 = r'G:\20x_dataset\copy_of_xy_01\development-dir\cell_lines\17\mcy'
d2 = r'G:\20x_dataset\copy_of_xy_01\development-dir\cell_lines\40\mcy'

# draw_lineage(get_file_list(d1))
# draw_lineage(get_file_list(d2))

group_cell_line(r'G:\20x_dataset\copy_of_xy_01\development-dir\cell_lines\17')