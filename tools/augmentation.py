from __future__ import annotations

import os
from math import fabs, sin, radians, cos
import skimage
import cv2
from skimage import transform, exposure, io
import numpy as np
from tqdm import tqdm
import imutils
import matplotlib.pyplot as plt


class AugmentorV2(object):
    def __init__(self, image: np.ndarray):
        self.__image = image
        self.dtype = image.dtype
        self.image = self.__convert_dtype()

    def __convert_dtype(self):
        min_16bit = np.min(self.__image)
        max_16bit = np.max(self.__image)
        image_8bit = np.array(np.rint(255 * ((self.__image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
        return image_8bit
        # return skimage.transform.resize(self.__image, self.__image.shape, preserve_range=True).astype(self.dtype)

    def rotate(self, angle, rotPoint=None):
        # return skimage.transform.rotate(self.image, angle=angle, resize=True, preserve_range=True).astype(self.dtype)
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


def augmentV2(raw_folder, saved_folder):
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    for f in tqdm(os.listdir(raw_folder)):
        if f.endswith('.db'):
            print(f)
            continue
        file = cv2.imread(os.path.join(raw_folder, f), -1)
        aug = AugmentorV2(file)
        img_rotate1 = aug.rotate(15)
        img_rotate2 = aug.rotate(30)
        img_rotate3 = aug.rotate(45)
        img_rotate4 = aug.rotate(60)
        img_rotate5 = aug.rotate(75)
        img_rotate6 = aug.rotate(90)
        img_rotate7 = aug.rotate(105)
        img_rotate8 = aug.rotate(120)
        img_rotate9 = aug.rotate(135)
        img_rotate10 = aug.rotate(150)
        img_rotate11 = aug.rotate(165)
        img_rotate12 = aug.rotate(180)
        img_rotate13 = aug.rotate(195)
        img_rotate14 = aug.rotate(210)
        img_rotate15 = aug.rotate(225)
        img_rotate16 = aug.rotate(240)
        img_rotate17 = aug.rotate(255)
        img_rotate18 = aug.rotate(270)
        img_rotate19 = aug.rotate(285)
        img_rotate20 = aug.rotate(300)
        img_rotate21 = aug.rotate(315)
        img_rotate22 = aug.rotate(330)
        img_rotate23 = aug.rotate(345)
        img_hflip = aug.flipHorizontal()
        img_vflip = aug.flipVertical()
        # img_bright = aug.adjustBright(0.7)
        # img_dark = aug.adjustBright(1.6)
        # img_mv = aug.movePosition()

        savename_raw = os.path.join(saved_folder, f)
        savename_rotate1 = os.path.join(saved_folder, f.replace('.tif', '-rotate1.tif'))
        savename_rotate2 = os.path.join(saved_folder, f.replace('.tif', '-rotate2.tif'))
        savename_rotate3 = os.path.join(saved_folder, f.replace('.tif', '-rotate3.tif'))
        savename_rotate4 = os.path.join(saved_folder, f.replace('.tif', '-rotate4.tif'))
        savename_rotate5 = os.path.join(saved_folder, f.replace('.tif', '-rotate5.tif'))
        savename_rotate6 = os.path.join(saved_folder, f.replace('.tif', '-rotate6.tif'))
        savename_rotate7 = os.path.join(saved_folder, f.replace('.tif', '-rotate7.tif'))
        savename_rotate8 = os.path.join(saved_folder, f.replace('.tif', '-rotate8.tif'))
        savename_rotate9 = os.path.join(saved_folder, f.replace('.tif', '-rotate9.tif'))
        savename_rotate10 = os.path.join(saved_folder, f.replace('.tif', '-rotate10.tif'))
        savename_rotate11 = os.path.join(saved_folder, f.replace('.tif', '-rotate11.tif'))
        savename_rotate12 = os.path.join(saved_folder, f.replace('.tif', '-rotate12.tif'))
        savename_rotate13 = os.path.join(saved_folder, f.replace('.tif', '-rotate13.tif'))
        savename_rotate14 = os.path.join(saved_folder, f.replace('.tif', '-rotate14.tif'))
        savename_rotate15 = os.path.join(saved_folder, f.replace('.tif', '-rotate15.tif'))
        savename_rotate16 = os.path.join(saved_folder, f.replace('.tif', '-rotate16.tif'))
        savename_rotate17 = os.path.join(saved_folder, f.replace('.tif', '-rotate17.tif'))
        savename_rotate18 = os.path.join(saved_folder, f.replace('.tif', '-rotate18.tif'))
        savename_rotate19 = os.path.join(saved_folder, f.replace('.tif', '-rotate19.tif'))
        savename_rotate20 = os.path.join(saved_folder, f.replace('.tif', '-rotate20.tif'))
        savename_rotate21 = os.path.join(saved_folder, f.replace('.tif', '-rotate21.tif'))
        savename_rotate22 = os.path.join(saved_folder, f.replace('.tif', '-rotate22.tif'))
        savename_rotate23 = os.path.join(saved_folder, f.replace('.tif', '-rotate23.tif'))
        savename_hflip = os.path.join(saved_folder, f.replace('.tif', '-hflip.tif'))
        savename_vflip = os.path.join(saved_folder, f.replace('.tif', '-vflip.tif'))
        savename_bright = os.path.join(saved_folder, f.replace('.tif', '-bright.tif'))
        savename_dark = os.path.join(saved_folder, f.replace('.tif', '-dark.tif'))
        savename_mv = os.path.join(saved_folder, f.replace('.tif', '-mv.tif'))
        # cv2.imwrite(savename_rotate1, img_rotate1)  # S
        # cv2.imwrite(savename_rotate2, img_rotate2)
        # cv2.imwrite(savename_rotate3, img_rotate3)
        # cv2.imwrite(savename_rotate4, img_rotate4)
        # cv2.imwrite(savename_rotate5, img_rotate5)
        # cv2.imwrite(savename_rotate6, img_rotate6)
        # cv2.imwrite(savename_rotate7, img_rotate7)  # S
        # cv2.imwrite(savename_rotate8, img_rotate8)
        # cv2.imwrite(savename_rotate9, img_rotate9)
        # cv2.imwrite(savename_rotate10, img_rotate10)
        # cv2.imwrite(savename_rotate11, img_rotate11)
        # cv2.imwrite(savename_rotate12, img_rotate12)
        # cv2.imwrite(savename_rotate13, img_rotate13)  # S
        # cv2.imwrite(savename_rotate14, img_rotate14)
        # cv2.imwrite(savename_rotate15, img_rotate15)
        # cv2.imwrite(savename_rotate16, img_rotate16)
        # cv2.imwrite(savename_rotate17, img_rotate17)
        # cv2.imwrite(savename_rotate18, img_rotate18)
        # cv2.imwrite(savename_rotate19, img_rotate19)
        # cv2.imwrite(savename_rotate20, img_rotate20)
        # cv2.imwrite(savename_rotate21, img_rotate21)
        # cv2.imwrite(savename_rotate22, img_rotate22)
        # cv2.imwrite(savename_rotate23, img_rotate23)   # s
        # cv2.imwrite(savename_hflip, img_hflip)   # S
        # cv2.imwrite(savename_vflip, img_vflip)   # S
        # cv2.imwrite(savename_bright, img_bright)
        # cv2.imwrite(savename_dark, img_dark)
        cv2.imwrite(savename_raw, aug.raw())
        # cv2.imwrite(savename_mv, img_mv)


def get_src():
    mcy_src_G = r'G:\20x_dataset\train_dataset-dev1.3.2\mcy\G'
    mcy_src_M = r'G:\20x_dataset\train_dataset-dev1.3.2\mcy\M'
    mcy_src_S = r'G:\20x_dataset\train_dataset-dev1.3.2\mcy\S'
    dic_src_G = r'G:\20x_dataset\train_dataset-dev1.3.2\dic\G'
    dic_src_M = r'G:\20x_dataset\train_dataset-dev1.3.2\dic\M'
    dic_src_S = r'G:\20x_dataset\train_dataset-dev1.3.2\dic\S'
    return [dic_src_G, mcy_src_G], [dic_src_M, mcy_src_M], [dic_src_S, mcy_src_S]


def run():
    src_dic, src_mcy = get_src()[0]
    dst_dic, dst_mcy = src_dic.replace('dic', 'augment_dic'), src_mcy.replace('mcy', 'augment_mcy')
    augmentV2(src_dic, dst_dic)
    augmentV2(src_mcy, dst_mcy)

    # augmentV2(dst_dic, dst_dic)
    # augmentV2(dst_mcy, dst_mcy)


if __name__ == '__main__':
    # b = aug.blur(radius=1.0)
    # plt.imshow(b, cmap='gray')
    # plt.show()
    # augment(r'C:\Users\Frozenleaves\PycharmProjects\resource\20x_train_data\mcy\M')
    # src = r'F:\60x_train_data\mcy\M'
    # dst = r'F:\60x_train_data\augment_mcy\M'
    # if not os.path.exists(dst):
    #     os.makedirs(dst)
    #     augmentV2(src, dst)
    # aug = AugmentorV2(cv2.imread(r'E:\20x_train_data-dev1.3.2\mcy\M\0ba24b5a890307a6041cf43175918929.tif', -1))
    # print(aug.image.dtype)
    # plt.imshow(aug.image, cmap='gray')
    # plt.show()
    # plt.imshow(aug.rotate(angle=30), cmap='gray')
    # plt.show()
    # plt.imshow(exposure.adjust_gamma(aug.rotate(angle=30), gamma=1.6), cmap='gray')
    # plt.show()
    # plt.imshow(cv2.equalizeHist(aug.rotate(angle=60)), cmap='gray')
    # plt.show()
    # plt.imshow(skimage.transform.rotate(aug.raw(), angle=60, preserve_range=True, resize=True), cmap='gray')
    # plt.show()
    # print(np.max(aug.image))
    # print(np.max(aug.rotate(angle=30)))

    run()
