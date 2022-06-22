"""
分割细胞图像并预测细胞周期
"""
from __future__ import print_function, unicode_literals, absolute_import, division, annotations

import json
import logging
import math
import os
from copy import deepcopy
from typing import Tuple, List

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import time
import utils
import cv2
import retry
import hashlib
import tensorflow as tf
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
from libtiff import TIFF
import skimage.exposure as exposure
from skimage.util import img_as_ubyte
import config
from train import get_model
import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)


def readTif(filepath, normalize=False):
    """读取并逐帧返回图像数据"""
    tif = TIFF.open(filepath)
    index = 0
    for img in tif.iter_images():
        filename = os.path.basename(filepath).replace('.tif', '-' + str(index).zfill(4) + '.tif')
        index += 1
        if normalize:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) / 255.0
        yield img, filename


class Predictor:
    def __init__(self, times=config.TIMES):
        self.model = get_model()
        if times == 60:
            self.model.load_weights(filepath=config.save_model_dir_60x)
        elif times == 20:
            self.model.load_weights(filepath=config.save_model_dir_20x_best)
        else:
            raise ValueError(f"Image magnification should be 20 or 60, got {config.TIMES} instead")

    def predict(self, images):
        """
        :param images: 一个包含多张图片的数组或列表，其形状为[image_count, image_width, image_height, image_channels]
        :return:
        """
        phaseMap = {0: 'G1/G2', 1: 'M', 2: 'S'}
        img = images
        # img = cv2.resize(img, (128, 128)) / 255.0
        tensor = tf.convert_to_tensor(img, dtype=tf.float64)

        prediction = self.model(tensor, training=False)
        # print(prediction)
        phases = []
        for i in prediction:
            # print(i)
            phase = np.argwhere(i == np.max(i))[0][0]
            # print(phase)
            # print(phaseMap[phase])
            phases.append(phaseMap.get(phase))
        return phases


class Prediction:
    def __init__(self, mcy: np.ndarray, dic: np.ndarray, rois, imagesize=None, predictor=None):
        self.imagesize = imagesize
        self.imgMcy = mcy
        self.imgDic = dic
        self.rois = rois
        if predictor is None:
            self.predictor = Predictor()
        else:
            self.predictor = predictor

    def __convert_dtype(self, __image):
        min_16bit = np.min(__image)
        max_16bit = np.max(__image)
        image_8bit = np.array(np.rint(255 * ((__image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
        return image_8bit

    # def getCell(self, cellfilter=20):
    #     """单细胞图像生成器
    #     """
    #     image_data = []
    #     for img in self.images:
    #         instances = self.parser.idMap.get(img)
    #         for instance in instances:
    #             data = utils.Data()
    #             coord = instances[instance]
    #             if type(self.imgMcy) is np.ndarray:
    #                 image_mcy = self.imgMcy
    #             else:
    #                 raise TypeError(
    #                     f"need image array, but got type {type(self.imgMcy)} in {self.imgMcy}")
    #             if type(self.imgDic) is np.ndarray:
    #                 image_dic = self.imgDic
    #             else:
    #                 raise TypeError(
    #                     f"need  image array, but got type {type(self.imgDic)} in {self.imgDic}")
    #             x0 = int(np.min(coord[0]))
    #             x1 = math.ceil(np.max(coord[0]))
    #             y0 = int(np.min(coord[1]))
    #             y1 = math.ceil(np.max(coord[1]))
    #             if np.min([(x1 - x0), (y1 - y0)]) < 5:
    #                 print(f'filter{x1 - x0, y1 - y0} ')
    #                 continue
    #             data.image_mcy = image_mcy[y0: y1, x0: x1]
    #             data.image_dic = image_dic[y0: y1, x0: x1]
    #
    #             if max(data.image_mcy.shape) < cellfilter:
    #                 print(data.image_mcy.shape)
    #                 continue
    #             cv2.imwrite('/home/zje/CellClassify/predict_data/LogData/dic/' + instance + '.tif',
    #                         self.__convert_dtype(data.image_dic))
    #             cv2.imwrite('/home/zje/CellClassify/predict_data/LogData/mcy/' + instance + '.tif',
    #                         self.__convert_dtype(data.image_mcy))
    #             # data.image_mcy = cv2.resize(data.image_mcy, (config.image_width, config.image_height))
    #             # data.image_dic = cv2.resize(data.image_dic, (config.image_width, config.image_height))
    #             data.image_mcy = cv2.resize(self.__convert_dtype(data.image_mcy),
    #                                         (config.image_width, config.image_height))
    #             data.image_dic = cv2.resize(self.__convert_dtype(data.image_dic),
    #                                         (config.image_width, config.image_height))
    #             data.image_id = instance
    #             image_data.append(data)
    #     return image_data

    def getCell(self):
        image_data = []
        rois_after_filter = []
        for i in self.rois:
            x0 = int(np.min(i[0]))
            x1 = math.ceil(np.max(i[0]))
            y0 = int(np.min(i[1]))
            y1 = math.ceil(np.max(i[1]))
            __mcy = self.imgMcy[x0:x1, y0:y1]
            __dic = self.imgDic[x0:x1, y0:y1]
            if 0 in __mcy.shape:
                continue
            rois_after_filter.append(i)
            mcy = self.__convert_dtype(__mcy)
            dic = self.__convert_dtype(__dic)
            instance_id = hashlib.md5(str(i).encode()).hexdigest()
            data = utils.Data()
            data.image_dic = cv2.resize(dic, (config.image_width, config.image_height)) / 255
            data.image_mcy = cv2.resize(mcy, (config.image_width, config.image_height)) / 255
            data.image_id = instance_id
            # plt.imshow(data.image_mcy)
            # plt.show()

            # cv2.imwrite('/home/zje/CellClassify/predict_data/LogData/dic/' + hashlib.md5(str(i).encode()).hexdigest() + '.tif',
            #             dic)
            # cv2.imwrite('/home/zje/CellClassify/predict_data/LogData/mcy/' + hashlib.md5(str(i).encode()).hexdigest() + '.tif',
            #             mcy)
            image_data.append(data)
        return image_data, rois_after_filter

    def predict_phase(self, images: np.ndarray):
        return self.predictor.predict(images)

    def predict(self) -> Tuple[np.ndarray, List]:
        """返回单帧图像中所有细胞的预测周期"""
        image_data = []
        id_data = []
        cells, rois_after_filter = self.getCell()
        for cell in cells:
            data = np.dstack([cell.image_dic, cell.image_mcy])
            image_data.append(data)
            id_data.append(cell.image_id)
        images = np.array(image_data)
        print('predict cell count ', len(image_data))
        phases = self.predict_phase(images)
        pl = list(phases)
        print(f"G1/G2: {pl.count('G1/G2')}\nS: {pl.count('S')}\nM: {pl.count('M')}")
        return phases, rois_after_filter

    def exportResult(self):
        pass


class Segmenter(object):
    def __init__(self, segment_model=None):
        if segment_model is None:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if config.TIMES == 20:
                self.model = StarDist2D(None, name='segment_20x_model',
                                        basedir='saved_segment_models/saved_20x_segment_model_1.3.4')
            elif config.TIMES == 60:
                self.model = StarDist2D(None, name='segment_60x_model',
                                        basedir='saved_segment_models/saved_60x_segment_model')
            else:
                raise ValueError(f"Image magnification in config.py should be 20 or 60, got {config.TIMES} instead")
        else:
            self.model = segment_model

    @retry.retry(exceptions=ResourceExhaustedError)
    def segment(self, image: np.ndarray):
        axis_norm = (0, 1)  # normalize channels independently
        im = normalize(image, 1, 99.8, axis=axis_norm)
        labels, details = self.model.predict_instances(im)  # TODO 优化此处限速流程 Optimize the speed limit process here
        return labels, details


class Segmentation(object):
    def __init__(self,
                 image_mcy: np.ndarray,
                 image_dic: np.ndarray,
                 imagename: str,
                 segmenter=None,
                 predictor=None,
                 roi_save_path=None,
                 label_save_path=None):
        self.imageName = imagename.replace('.tif', '.png')
        if segmenter is None:
            self.segmentor = Segmenter()
        else:
            self.segmentor = segmenter
        self.predictor = predictor
        self.roi_save_path = roi_save_path
        self.label_save_path = label_save_path
        self.img_mcy = image_mcy
        self.img_dic = image_dic

    def segment(self):
        return self.segmentor.segment(image=self.img_mcy)

    @property
    def labels(self):
        return self.segment()[0]

    @property
    def details(self):
        return self.segment()[1]

    def rois(self):
        return self.details['coord']

    def save_labels(self, label_save_path=None):
        from csbdeep.io import save_tiff_imagej_compatible
        if label_save_path:
            save_tiff_imagej_compatible(label_save_path, self.labels, axes='YX')
        else:
            save_tiff_imagej_compatible(self.label_save_path, self.labels, axes='YX')

    def save_rois(self, roi_save_path=None):
        from stardist import export_imagej_rois
        if roi_save_path:
            export_imagej_rois(roi_save_path, self.rois())
        elif self.roi_save_path:
            export_imagej_rois(self.roi_save_path, self.rois())
        else:
            raise ValueError('roi_save_path need to be give, in this function or in the class init')

    def __get_segment_result(self):
        """将分割的坐标转化为可识读的json字典"""
        rois = self.rois()
        # c = ConverterXY(self.imageName.replace('.tif', '.png'), coordinates=rois)
        # return c.json
        return rois

    def __convert_dtype(self, __image):
        min_16bit = np.min(__image)
        max_16bit = np.max(__image)
        image_8bit = np.array(np.rint(255 * ((__image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
        return image_8bit

    def predict(self):
        """a test function, do not use it!!!"""
        images = []
        coords = self.rois()
        for i in coords:
            x0 = int(np.min(i[0]))
            x1 = math.ceil(np.max(i[0]))
            y0 = int(np.min(i[1]))
            y1 = math.ceil(np.max(i[1]))
            # test_mcy = self.img_mcy[y0:y1, x0:x1]
            # test_dic = self.img_dic[y0:y1, x0:x1]
            # data.image_mcy = cv2.resize(data.image_mcy, (config.image_width, config.image_height))
            # data.image_dic = cv2.resize(data.image_dic, (config.image_width, config.image_height))
            test_mcy = self.img_mcy[x0:x1, y0:y1]
            test_dic = self.img_dic[x0:x1, y0:y1]
            if 0 in test_mcy.shape:
                continue
            test_mcy = self.__convert_dtype(test_mcy)
            test_dic = self.__convert_dtype(test_dic)

            # cv2.imwrite('/home/zje/CellClassify/predict_data/LogData/dic/' + hashlib.md5(str(i).encode()).hexdigest() + '.tif',
            #             test_dic)
            # cv2.imwrite('/home/zje/CellClassify/predict_data/LogData/mcy/' + hashlib.md5(str(i).encode()).hexdigest() + '.tif',
            #             test_mcy)
            # data.image_mcy = cv2.resiz

            test_img = np.dstack([cv2.resize(test_dic, (100, 100)), cv2.resize(test_mcy, (100, 100))])
            images.append(test_img / 255)
            # print(Predictor().predict(np.array([test_img])))
        ret = Predictor().predict(np.array(images))
        print(f'G1/G2: {ret.count("G1/G2")}, S: {ret.count("S")}, M: {ret.count("M")}')

    def __add_predict_phase(self):
        rois = self.__get_segment_result()
        regions = []
        regions_tmp = {
            "shape_attributes":
                {
                    "name": "polygon",
                    "all_points_x": [],
                    "all_points_y": []
                },
            "region_attributes":
                {
                    "phase": None
                }
        }

        predictor = Prediction(mcy=self.img_mcy, dic=self.img_dic, rois=rois, imagesize=self.img_mcy.shape,
                               predictor=self.predictor)

        phases, rois_after_filter = predictor.predict()  # 限速步骤 # TODO 优化此处限速流程 Optimize the speed limit process here
        print(len(rois_after_filter), len(phases))
        for i in zip(rois, phases):
            all_x = []
            all_y = []
            for j in range(i[0].shape[1]):
                all_x.append(float(i[0][:, j][1]))
                all_y.append(float(i[0][:, j][0]))
            phase = i[1]
            t = deepcopy(regions_tmp)
            t["shape_attributes"]["all_points_x"] = all_x
            t["shape_attributes"]["all_points_y"] = all_y
            t["region_attributes"]["phase"] = phase
            regions.append(t)
        tmp = {
            self.imageName: {
                "filename": self.imageName,
                "size": 4194304,
                "regions": regions,
                "file_attributes": {}
            }
        }
        return tmp

    @property
    def predict_result(self):
        return self.__add_predict_phase()

    def export_predict_result(self, filepath):
        """导出分割和预测细胞周期到json文件中"""
        with open(filepath, 'w') as f:
            json.dump(self.__add_predict_phase(), f)


def segment(pcna: os.PathLike | str, bf: os.PathLike | str, output: os.PathLike | str, segment_model=None,
            normalize=False):
    jsons = {}
    mcy_data = readTif(filepath=pcna, normalize=normalize)
    dic_data = readTif(filepath=bf, normalize=normalize)
    segmenter = Segmenter(segment_model=segment_model)
    predictor = Predictor()
    iiiiiii = 0
    while True:
        try:
            mcy_img, imagename = next(mcy_data)
            dic_img, _ = next(dic_data)
            print(f'start segment {os.path.basename(imagename)} ...')
            start_time = time.time()

            seg = Segmentation(image_mcy=mcy_img,
                               imagename=imagename,
                               image_dic=dic_img,
                               segmenter=segmenter,
                               predictor=predictor)
            # seg.predict()

            value = seg.predict_result
            jsons.update(value)
            end_time = time.time()
            print(f'finish segment {os.path.basename(imagename)}', 'ok')
            print(f'cost time {end_time - start_time}s')
            del seg
        except StopIteration:
            break
    json_filename = os.path.basename(pcna).replace('.tif', '.json')
    if output:
        out = output
    else:
        out = json_filename
    with open(out, 'w') as f:
        json.dump(jsons, f)


def convert(img):
    img_mcy = exposure.adjust_gamma(cv2.imread(img, -1), 0.1)
    png = img_as_ubyte(img_mcy)
    return png


if __name__ == '__main__':
    # segment(pcna='/home/zje/CellClassify/predict_data/dataset/beini-dataset/cep192-mcy.tif',
    #         bf='/home/zje/CellClassify/predict_data/dataset/beini-dataset/cep192-dic.tif',
    #         output='/home/zje/CellClassify/predict_data/dataset/beini-dataset/cep192.json', segment_model=None)

    # segment(pcna='/home/zje/CellClassify/predict_data/mcy/copy_of_1_xy03.tif',
    #         bf='/home/zje/CellClassify/predict_data//dic/copy_of_1_xy03.tif',
    #         output='/home/zje/CellClassify/predict_data/copy_of_1_xy03.json', segment_model=None)

    segment(pcna=r'G:\20x_dataset\copy_of_xy_16\raw\copy_of_1_xy16-mcy.tif',
            bf=r'G:\20x_dataset\copy_of_xy_16\raw\copy_of_1_xy16-dic.tif',
            output=r'G:\20x_dataset\copy_of_xy_16\raw\copy_of_1_xy16_new.json', segment_model=None)

    # segment(pcna='/home/zje/CellClassify/train_dataset/mitosis/series11/mcy/copy11.tif',
    #         bf='/home/zje/CellClassify/train_dataset/mitosis/series11/dic/copy11.tif',
    #         output='/home/zje/CellClassify/train_dataset/mitosis/series1/copy11.json', segment_model=None)

    # segment(pcna='/home/zje/CellClassify/predict_data/dataset/test_pcna.tif',
    #         bf='/home/zje/CellClassify/predict_data/dataset/test_dic.tif',
    #         output='/home/zje/CellClassify/predict_data/dataset/test_result.json', segment_model=model)

    # mcy = '/home/zje/CellClassify/predict_data/dataset/60x_test_predict_phase/mcy'
    # dic = '/home/zje/CellClassify/predict_data/dataset/60x_test_predict_phase/dic'
    # ann = '/home/zje/CellClassify/predict_data/dataset/60x_test_predict_phase/20200729-RPE-s2_cpd.json'
    #
    # p = Predict(mcy=mcy, dic=dic, annotation_json=ann, imagesize=(1200, 1200))
    # p.predict(frame=None)
    pass
