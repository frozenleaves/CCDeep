from __future__ import annotations

import os
import skimage
import cv2
from skimage import exposure
import numpy as np
from tqdm import tqdm
import imutils


class Augmentor(object):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.dtype = image.dtype
        self.image = self.__convert_dtype()

    def __convert_dtype(self):
        min_16bit = np.min(self.__image)
        max_16bit = np.max(self.__image)
        image_8bit = np.array(np.rint(255 * ((self.__image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
        return image_8bit

    def rotate(self, angle):
        return imutils.rotate_bound(self.image, angle)

    def flipHorizontal(self):
        return np.fliplr(self.image)

    def flipVertical(self):
        return np.flipud(self.image)

    def adjustBright(self, gamma):
        """gamma > 1, bright; gamma < 1. dark"""
        return skimage.exposure.adjust_gamma(self.image, gamma=gamma).astype(self.dtype)

    def raw(self):
        return self.image

    def movePosition(self):
        right = np.zeros(shape=(self.image.shape[0], 20), dtype=self.dtype)
        tmp1 = np.hstack([self.image, right])
        down = np.zeros(shape=(20, tmp1.shape[1]), dtype=self.dtype)
        tmp2 = np.vstack([tmp1, down])
        return tmp2


def augment_before_train(raw_folder, saved_folder):
    """
    如果拥有足够的存储空间，或者内存有限，请在开始训练模型之前，使用此函数进行数据增强，并且将配置文件中的AUGMENTATION_IN_TRAINING
    设置为False。否则，请不要调用次函数，并且将AUGMENTATION_IN_TRAINING设置为True。
    Args:
        raw_folder: 原始数据所在目录
        saved_folder: 通过数据增强生成的文件保存目录

    """
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    for f in tqdm(os.listdir(raw_folder)):
        if f.endswith('.db'):
            continue
        file = cv2.imread(os.path.join(raw_folder, f), -1)
        aug = Augmentor(file)
        cv2.imwrite(os.path.join(saved_folder, f), aug.raw())
        if os.path.basename(raw_folder) == 'G':
            continue
        elif os.path.basename(raw_folder) == 'S':
            for s in range(80, 360, 80):
                fname = os.path.join(saved_folder, f.replace(".tif", f"-rotate{s}.tif"))
                cv2.imwrite(fname, aug.rotate(s))
            cv2.imwrite(os.path.join(saved_folder, f.replace(".tif", f"-hflip.tif")), aug.flipHorizontal())
            cv2.imwrite(os.path.join(saved_folder, f.replace(".tif", f"-vflip.tif")), aug.flipHorizontal())
        elif os.path.basename(raw_folder) == 'M':
            for m in range(0, 361, 15):
                save_name = os.path.join(saved_folder, f.replace(".tif", f"-rotate{m}.tif"))
                cv2.imwrite(save_name, aug.rotate(m))
            cv2.imwrite(os.path.join(saved_folder, f.replace(".tif", f"-hflip.tif")), aug.flipHorizontal())
            cv2.imwrite(os.path.join(saved_folder, f.replace(".tif", f"-vflip.tif")), aug.flipHorizontal())
        else:
            raise ValueError(f"{os.path.basename(raw_folder)} is not a available class")


def augment_in_train(img, label):
    aug = Augmentor(img)
    aug_images = [aug.flipHorizontal(), aug.flipVertical(), aug.raw()]
    if label == 0:
        return aug_images
    elif label == 1:
        for i in range(0, 361, 15):
            aug_images.append(aug.rotate(i))
        return aug_images
    else:
        for j in range(80, 360, 80):
            aug_images.append(aug.rotate(j))
        return aug_images


def augment(dic, mcy):
    for i in os.listdir(mcy):
        p_mcy = os.path.join(mcy, i)
        p_dic = os.path.join(dic, i)
        dst_dic, dst_mcy = p_dic.replace('dic', 'aug_dic'), p_mcy.replace('mcy', 'aug_mcy')
        augment_before_train(p_mcy, dst_mcy)
        augment_before_train(p_dic, dst_dic)


if __name__ == '__main__':
    pass
    # augment(r'G:\20x_dataset\train_dataset-dev1.3.2\dic', r'G:\20x_dataset\train_dataset-dev1.3.2\mcy')
    # augment(r'H:\CCDeep-data\raw-data\train\segmentation\copy_of_xy_01\train_classification_dataset\dic',
    #         r'H:\CCDeep-data\raw-data\train\segmentation\copy_of_xy_01\train_classification_dataset\mcy')
    # path = r'G:\20x_dataset\train_dataset-dev1.3.2\train_dic\t'
    # for i in os.listdir(path):
    #     src = os.path.join(path, i)
    #     print(src)
    #     dst = src.replace(r'train_dic\t', r'train_dic\t2')
    #     augment_before_train(src, dst)
