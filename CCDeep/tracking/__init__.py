#  第一步：进行目标检测，利用生成的json文件，可以很方便的给出每一帧每个细胞的位置


# 第二步：将所有目标框中对应的目标抠出来，进行特征提取


# 第三步：进行相似度计算，计算前后两帧目标之间的匹配程度


# 第四步：数据关联，为每个对象分配目标的 ID


# TODO  实现json向bounding box的转化 finished

# TODO  实现特征提取 finished

# TODO  实现前后帧特征匹配

# TODO 处理遮挡后匹配


import matplotlib.pyplot as plt
import numpy as np
import cv2
import base

frame1 = r'G:\20x_dataset\copy_of_xy_01\tif\mcy\copy_of_1_xy01-0000.tif'
frame2 = r'G:\20x_dataset\copy_of_xy_01\tif\mcy\copy_of_1_xy01-0001.tif'

image1 = cv2.imread(frame1, -1)
image2 = cv2.imread(frame2, -1)


from base import Cell




x =np.arange(-2* np.pi, 2*np.pi, 0.01)
y = np.cos(x)

from base import Vector

a = Vector(1, 1)

b = Vector(2, 2)

c = Vector(1, 0)

# print(a.cosDistance(b))
# print(a.cosDistance(c))
# print(a.cosSimilar(b))
# print(a.cosSimilar(c))

def f(x):
    return x/(x+1)

x = np.arange(-10, 10, 0.01)




from base import Rectangle

a = Rectangle(20, 50, 10, 40)

# b = Rectangle(30, 70, 20, 60)
# b = Rectangle(60, 80, 20, 60)
b = Rectangle(10, 60, 20, 30)
# b = Rectangle(30, 40, 20, 30)

bg = np.zeros(shape=(100, 100))
# a.draw(background=bg)
#
# print(a.isIntersect(b))
# print(b.isIntersect(a))
# print(b.isIntersect(b))
#
# print(b.isInclude(a))
# print(a.isInclude(b))

# b.draw(background=bg,isShow=True)

c = Rectangle(-10, 20, -20, 40)
d = Rectangle(5, 15, 10, 20)
# d.draw(bg)
# c.draw(bg, isShow=True)


# from base import Cell
#
# bg = np.zeros(shape=(1024, 1024))
# Cell1 = Cell(position=([545, 640], [576, 670]))
# Cell2 = Cell(position=([565, 660], [596, 690]))
# Cell3 = Cell(position=([800, 860], [900, 990]))
#
#
# c1 = Rectangle(Cell1.bbox[0], Cell1.bbox[1], Cell1.bbox[2], Cell1.bbox[3])
# c2 = Rectangle(*Cell2.bbox)
# c3 = Rectangle(*Cell3.bbox)
#
# print(Cell1 in Cell2)
# print(Cell2 in Cell1)
#
# print(Cell2 in Cell2)
#
# print(Cell1 in Cell3)
# print(Cell3 in Cell1)
#
# c1.draw(bg)
# Cell1.available_range.draw(bg)
# c2.draw(bg)
# Cell2.available_range.draw(bg)
# c3.draw(bg)
# Cell3.available_range.draw(bg, isShow=True)

# s = [rf" python .\main.py -p F:\wangjiaqi\src\s{i}\mcy.tif -bf F:\wangjiaqi\src\s{i}\dic.tif -o F:\wangjiaqi\src\s{i}\ret.json -t" for i in range(1, 12)]
# for i in s:
#     print(i)


def loop(matched_cells_dict: dict):
    cell_dict_keys = list(matched_cells_dict.keys())
    length = len(cell_dict_keys)
    match_result = {}
    for i in range(length - 1):
        cell_1 = matched_cells_dict[cell_dict_keys.pop(0)]
        for j in range(len(cell_dict_keys)):
            cell_2 = matched_cells_dict[cell_dict_keys[j]]
            print((cell_1, cell_2))





