"""
定义追踪树，追踪节点
每个追踪树root节点起始于第一帧的细胞
起始追踪初始化第一帧识别的细胞数量个树实例

定义树节点， 每个节点包含细胞的详细信息。
以及追踪信息，帧数，分裂信息等


"""
from __future__ import annotations

import dataclasses
from copy import deepcopy
from functools import wraps, lru_cache

import os
import warnings
from typing import List
import heapq

import matplotlib.pyplot as plt
import tifffile
import numpy as np
import cv2
import json

import treelib
from treelib import Tree, Node

from CCDeep.tracking.prepare import convert_dtype
from base import Cell, Rectangle, Vector, SingleInstance, CacheData
from t_error import InsertError, MitosisError, NodeExistError, ErrorMatchMitosis
from feature import FeatureExtractor

TEST = False
TEST_INDEX = None
CELL_NUM = 0


def imread(filepath: str | os.PathLike) -> np.ndarray:
    return tifffile.imread(filepath)


def get_frame_by_index(image: np.ndarray, index: int) -> np.ndarray:
    return image[index]


def feature_extract(mcy, dic, jsonfile):
    """逐帧返回FeatureExtractor实例，包含当前帧，前一帧，后一帧"""
    _dic = imread(dic)
    _mcy = imread(mcy)
    with open(jsonfile) as f:
        annotations = json.load(f)

    def get_fe(frame_index, frame_name):
        dic_image = get_frame_by_index(_dic, frame_index)
        mcy_image = get_frame_by_index(_mcy, frame_index)
        region = annotations[frame_name.replace('.tif', '.png')]['regions']
        return FeatureExtractor(image_dic=dic_image, image_mcy=mcy_image, annotation=region, frame_index=frame_index)

    for i in range(_mcy.shape[0]):
        current_frame_index = i
        if i == 0:
            before_frame_index = 0
        else:
            before_frame_index = i - 1
        if i == _mcy.shape[0] - 1:
            after_frame_index = i
        else:
            after_frame_index = i + 1
        before_frame_name = os.path.basename(mcy).replace('.tif', '-' + str(before_frame_index).zfill(4) + '.tif')
        after_frame_name = os.path.basename(mcy).replace('.tif', '-' + str(after_frame_index).zfill(4) + '.tif')
        current_frame_name = os.path.basename(mcy).replace('.tif', '-' + str(current_frame_index).zfill(4) + '.tif')
        before_fe = get_fe(before_frame_index, before_frame_name)
        current_fe = get_fe(current_frame_index, current_frame_name)
        after_fe = get_fe(after_frame_index, after_frame_name)
        yield before_fe, current_fe, after_fe


class Filter(SingleInstance):
    """
    过滤一帧中距离较远的细胞，降低匹配候选数量
    基本操作数为帧
    过滤依据：bbox的坐标
    参数：
    """

    def __init__(self):
        super(Filter, self).__init__()

    def filter(self, current: Cell, cells: List[Cell]):
        """

        :param current: 待匹配的细胞
        :param cells: 候选匹配项
        :return: 筛选过的候选匹配项
        """
        return [cell for cell in cells if cell in current]


class Checker(object):
    """检查器，检查参与匹配的细胞是否能够计算"""
    _protocols = [None, 'calcIoU', 'calcCosDistance', 'calcCosSimilar', 'calcEuclideanDistance',
                  'compareDicSimilar', 'compareMcySimilar', 'compareShapeSimilar']

    def __init__(self, protocol=None):
        self.protocol = protocol

    def check(self):
        pass

    def __call__(self, method):

        if self.protocol not in self._protocols:
            warnings.warn(f"Don't support protocol: {self.protocol}, now just support {self._protocols}")
            _protocol = None
        else:
            _protocol = self.protocol

        @wraps(method)
        def wrapper(*args, **kwargs):
            """args[0]: object of func; args[1]: param 1 of func method; args[2]: param 2 of func method"""
            # print(_protocol)
            # print('args:', *args)
            if _protocol is None:
                return method(*args, **kwargs)
            elif _protocol == 'calcIoU':
                return method(*args, **kwargs)
            elif _protocol == 'calcCosDistance':
                return method(*args, **kwargs)
            elif _protocol == 'calcCosSimilar':
                return method(*args, **kwargs)
            elif _protocol == 'compareDicSimilar':
                return method(*args, **kwargs)
            elif _protocol == 'compareMcySimilar':
                return method(*args, **kwargs)
            elif _protocol == 'compareShapeSimilar':
                return method(*args, **kwargs)
            else:
                return method(*args, **kwargs)

        return wrapper


class CellNode(Node):
    """
    追踪节点，包含细胞的tracking ID，以及细胞自身的详细信息，和父子节点关系
    """
    _instance_ = {}
    STATUS = ['ACCURATE', 'ACCURATE-FL', 'INACCURATE', 'INACCURATE-MATCH', 'PREDICTED']

    def __new__(cls, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cls._instance_:
            cls._instance_[key] = super().__new__(cls)
            cls._instance_[key].branch_id = None
            cls._instance_[key].status = None
            cls._instance_[key].track_id = None
            cls._instance_[key].life = 5  # 每个分支初始生命值为5，如果匹配成功则+1，如果没有匹配上，或者利用缺省值填充匹配，则-1，如如果生命值为0，则该分支不再参与匹配
            cls._instance_[key]._init_flag = False
        return cls._instance_[key]

    def __init__(self, cell: Cell, node_type='cell', fill_gap_index=None):
        if not self._init_flag:
            self.cell = cell
            if node_type == 'gap':
                assert fill_gap_index is not None
            super().__init__((cell, self.branch_id))
            self._init_flag = True

    @property
    def identifier(self):
        return self.__hash__()

    def _set_identifier(self, nid):
        if nid is None:
            self._identifier = self.__hash__()
        else:
            self._identifier = nid

    def get_status(self):
        return self.status

    def set_status(self, status):
        if status in self.STATUS:
            self.status = status
        else:
            raise ValueError(f"set error status: {status}")

    def get_branch_id(self):
        if self.branch_id is None:
            raise ValueError("Don't set the branch_id")
        else:
            return self.branch_id

    def set_branch_id(self, branch_id):
        self.branch_id = branch_id

    def get_track_id(self):
        if self.track_id is None:
            raise ValueError("Don't set the track_id")
        else:
            return self.track_id

    def set_track_id(self, track_id):
        self.track_id = track_id

    def __repr__(self):
        return f"Cell Node of {self.cell}, branch {self.branch_id}"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return str(id(self))


class TrackingTree(Tree):
    """追踪树，起始的时候初始化根节点，逐帧扫描的时候更新left子节点，发生分裂的时候添加right子节点"""

    def __init__(self, root: CellNode = None, track_id=None):
        super().__init__()
        self.root = root
        self.track_id = track_id
        self.__last_layer = []

    def __contains__(self, item):
        return item.identifier in self._nodes

    @property
    def last_layer(self):
        return self.__last_layer

    @property
    def last_layer_cell(self):
        cells = set()
        for node in self.leaves():
            cells.add(node.cell)
        return cells

    def update_last_layer(self, node_list: List[CellNode]):
        self.__last_layer = node_list

    def auto_update_last_layer(self):
        self.__last_layer = self.leaves()


class MatchRecorder(object):
    """
    匹配记录器，每匹配一帧，更新一下匹配记录。
    记录内容包括：已经完成匹配的细胞，没有匹配的细胞，匹配的精确度
    """


class MatchCache(object):
    """匹配缓存对象，用于缓存已经匹配过的帧，方便做recheck"""

    def __init__(self):
        self.cache = CacheData()

    def search(self):
        """检索缓存"""

    def add(self):
        """添加缓存"""

    def remove(self):
        """删除缓存"""

    def flush(self):
        """刷新缓存"""


class Match(SingleInstance):
    """
    匹配器，根据前后帧及当前帧来匹配目标并分配ID
    主要用来进行特征比对，计算出相似性
    操作单位：帧
    """

    def __init__(self):
        super(Match, self).__init__()

    def normalize(self, x, _range=(0, np.pi / 2)):
        """将值变换到区间[0, π/2]"""
        return _range[0] + (_range[1] - _range[0]) * x

    def calcIoU(self, cell_1: Cell, cell_2: Cell):
        """
        计算两个细胞的交并比
        返回值范围: float(0-1)
        """
        rect1 = Rectangle(*cell_1.bbox)
        rect2 = Rectangle(*cell_2.bbox)
        if not rect1.isIntersect(rect2):
            return 0
        elif rect1.isInclude(rect2):
            return 1
        else:
            intersection = Rectangle(min(rect1.x_min, rect2.x_min), max(rect1.x_max, rect2.x_max),
                                     min(rect1.y_min, rect2.y_min), max(rect1.y_max, rect2.y_max))
            union = Rectangle(max(rect1.x_min, rect2.x_min), min(rect1.x_max, rect2.x_max),
                              max(rect1.y_min, rect2.y_min), min(rect1.y_max, rect2.y_max))
            return union.area / intersection.area

    def calcCosDistance(self, cell_1: Cell, cell_2: Cell):
        """
        计算两个细胞的余弦距离
        返回值范围: float[0, 2]
        距离越小越相似，通过反正切函数缩放到[0, 1)
        返回值为归一化[0, π/2]之后的余弦值，返回值越小，表示相似度越低
        """
        dist = cell_1.vector.cosDistance(cell_2.vector)
        return np.cos(np.arctan(dist) / (np.pi / 2))

    def calcCosSimilar(self, cell_1: Cell, cell_2: Cell):
        """
        计算两个细胞的中心点的余弦相似度
        返回值范围: float[0, 1]
        值越大越相似
        返回值为归一化[0, π/2]后的正弦值
        """
        score = cell_1.vector.cosSimilar(cell_2.vector)
        return np.sin(self.normalize(score))

    def calcAreaSimilar(self, cell_1: Cell, cell_2: Cell):
        """
        计算两个细胞的面积相似度
        返回值范围: float[0, 1]
        """
        return min(cell_1.area, cell_2.area) / max(cell_1.area, cell_2.area)

    def calcEuclideanDistance(self, cell_1: Cell, cell_2: Cell):
        """
        计算两个细胞中心点的欧氏距离
        返回值范围: float(0,∞)
        距离越小越相似，通过反正切函数缩放到[0, 1)
        返回值为归一化[0, π/2]之后的余弦值，返回值越小，表示相似度越低
        """
        dist = cell_1.vector.EuclideanDistance(cell_2.vector)
        return np.cos(np.arctan(dist) / (np.pi / 2))

    def compareDicSimilar(self, cell_1: Cell, cell_2: Cell):
        """
        比较dic相似度
        返回值范围: float(0, 1)
        """
        dic1 = Vector(cell_1.feature.dic_intensity, cell_1.feature.dic_variance)
        dic2 = Vector(cell_2.feature.dic_intensity, cell_2.feature.dic_variance)
        return np.sin(self.normalize(dic1.cosSimilar(dic2)))

    def compareMcySimilar(self, cell_1: Cell, cell_2: Cell):
        """
        比较mcy相似度
        返回值范围: float(0, 1)
        """
        mcy1 = Vector(cell_1.feature.mcy_intensity, cell_1.feature.mcy_variance)
        mcy2 = Vector(cell_2.feature.mcy_intensity, cell_2.feature.mcy_variance)
        return np.sin(self.normalize(mcy1.cosSimilar(mcy2)))

    def compareShapeSimilar(self, cell_1: Cell, cell_2: Cell):
        """
                      计算两个细胞的轮廓相似度
        返回值范围: float(0, 1)
        score值越小表示相似度越大，可进行取反操作
        返回值为归一化[0, π/2]后的余弦值
        """
        score = cv2.matchShapes(cell_1.contours, cell_2.contours, 1, 0.0)
        # return np.cos(self.normalize(score))
        return score


class Matcher(object):
    """拿子细胞找母细胞，根据下一帧匹配上一帧"""
    _instances = {}

    def __new__(cls, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self):
        self.matcher = Match()

    def draw_bbox(self, bg1, bbox, track_id=None):
        if len(bg1.shape) > 2:
            im_rgb1 = bg1
        else:
            im_rgb1 = cv2.cvtColor(convert_dtype(bg1), cv2.COLOR_GRAY2RGB)
        cv2.rectangle(im_rgb1, (bbox[2], bbox[0]), (bbox[3], bbox[1]),
                      [255, 255, 0], 2)
        cv2.putText(im_rgb1, str(track_id), (bbox[3], bbox[1]), cv2.FONT_HERSHEY_COMPLEX,
                    1, (255, 255, 255), 1)
        return im_rgb1

    def predict_next_position(self, parent: Cell):
        """根据速度预测子细胞可能出现的位置，利用预测的子细胞参与匹配"""
        new_cell = parent.move(parent.move_speed, 1)
        try:
            new_cell.set_feature(parent.feature)
        except ValueError:
            pass
        return new_cell

    def _filter(self, child: Cell, cells: List[Cell]):
        """

        :param current: 待匹配的细胞
        :param cells: 候选匹配项
        :return: 筛选过的候选匹配项
        """
        return [cell for cell in cells if cell.move(cell.move_speed, 1) in child]

    def match_candidates(self, child: Cell, before_cell_list: List[Cell]):
        """匹配候选项"""
        return self._filter(child, before_cell_list)

    def calc_similar(self, parent, child_cell):
        similar = [self.matcher.calcIoU(parent, child_cell),
                   self.matcher.calcAreaSimilar(parent, child_cell),
                   self.matcher.compareShapeSimilar(parent, child_cell),
                   self.matcher.calcEuclideanDistance(parent, child_cell),
                   self.matcher.calcCosDistance(parent, child_cell),
                   self.matcher.calcCosSimilar(parent, child_cell),
                   self.matcher.compareDicSimilar(parent, child_cell),
                   self.matcher.compareMcySimilar(parent, child_cell)]
        np.set_printoptions(precision=6, floatmode='fixed')
        return np.array(similar)

    @lru_cache(maxsize=None)
    def match_similar(self, cell_1: Cell, cell_2: Cell):
        similar = {'IoU': self.matcher.calcIoU(cell_1, cell_2),
                   'shape': self.matcher.compareShapeSimilar(cell_1, cell_2),
                   'area': self.matcher.calcAreaSimilar(cell_1, cell_2)
                   }
        return similar

    def match_duplicate_child(self, parent, unmatched_child_list):
        """
        一旦调用这个函数，意味着候选项中IoU匹配失败，尝试判断是否发生有丝分裂，
        如果发生有丝分裂，则子代细胞和上一帧母细胞之间应该都有交集，此方法会筛选
        出所有可能项，但是会包括其他非子细胞的的相邻细胞，需要进一步判断。
        返回值为{Cell: similar_dict}形式的字典
        """
        matched = {}
        for i in unmatched_child_list:
            similar = self.match_similar(parent, i)
            if similar.get('IoU') > 0.1:
                matched[i] = similar
        if not matched:
            for i in unmatched_child_list:
                similar = self.match_similar(parent, i)
                if similar.get('IoU') > 0:
                    matched[i] = similar
        return matched

    def get_similar_sister(self, parent: Cell, matched_cells_dict: dict, area_t=0.8, shape_t=0.03, area_size_t=1.4,
                           iou_t=0.2):
        """在多个候选项中找到最相似的两个细胞作为子细胞"""
        cell_dict_keys = list(matched_cells_dict.keys())
        for cell in cell_dict_keys:
            if matched_cells_dict[cell].get('IoU') < iou_t:
                cell_dict_keys.remove(cell)
            if cell.area * area_size_t > parent.area:
                cell_dict_keys.remove(cell)
        length = len(cell_dict_keys)
        match_result = {}
        for i in range(length - 1):
            cell_1 = matched_cells_dict[cell_dict_keys.pop(0)]
            for j in range(len(cell_dict_keys)):
                cell_2 = matched_cells_dict[cell_dict_keys[j]]
                score = self.match_similar(cell_1, cell_2)
                if score.get('area') >= area_t and score.get('shape') <= shape_t:
                    match_result[(cell_1, cell_2)] = score.get('area') + score.get('IoU')
        if match_result:
            max_score_cells = max(match_result, key=match_result.get)
            return max_score_cells
        else:
            raise MitosisError('cannot match the suitable daughter cells')

    def select_mitosis_cells(self, parent: Cell, candidates_child_list: List[Cell], area_t=0.8, shape_t=0.03,
                             area_size_t=1.8):
        """
        如果发生有丝分裂，选择两个子细胞， 调用这个方法的时候，确保大概率发生了分裂事件
        如果返回值为有丝分裂，则需要检查两个子细胞的面积是否正常，如果某一个子细胞的面积过大，则认为是误判

        :return ([cell_1, cell2], 'match status')
        """

        matched_cells_dict = self.match_duplicate_child(parent, candidates_child_list)
        if len(matched_cells_dict) < 2:
            raise MitosisError('not enough candidates')
        elif parent.area < area_size_t * min([i.area for i in candidates_child_list]):  # 如果母细胞太小了，不认为会发生有丝分裂，转向单项判断
            raise MitosisError('The cell is too small to have cell division !')
        else:
            if len(matched_cells_dict) == 2:
                cells = list(matched_cells_dict.keys())
                if self.matcher.calcAreaSimilar(cells[0], cells[1]) > area_t and self.matcher.compareShapeSimilar(
                        cells[0], cells[1]) < shape_t:
                    return cells, 'ACCURATE'
                else:
                    for c in cells:
                        if c.area * area_t > parent.area:
                            raise ErrorMatchMitosis('The candidate cells exist big one, maybe error match!')
                    return cells, 'INACCURATE'
            else:
                # max_two = heapq.nlargest(2, [sum(sm) for sm in matched_cells_dict.values()])  # 找到匹配结果中最大的两个值
                try:
                    max_two = self.get_similar_sister(parent, matched_cells_dict)  # 找到匹配结果中最大的两个值
                except MitosisError:
                    raise MitosisError("not enough candidates, after matched.")
                better_sisters = []
                for i in max_two:
                    better_sisters.append(i)
                if self.matcher.calcAreaSimilar(better_sisters[0],
                                                better_sisters[1]) > area_t and self.matcher.compareShapeSimilar(
                    better_sisters[0], better_sisters[1]) < shape_t:
                    return better_sisters, 'ACCURATE'
                else:
                    return better_sisters, 'INACCURATE'

    def select_child(self, score_dict):
        """
        对于有多个IOU匹配的选项，选择相似度更大的那一个, 此处是为了区分发生重叠的细胞，而非发生有丝分裂.
        注意：这个方法匹配出来的结果不一定是准确的，有可能因为细胞交叉导致发生错配，需要在后面的流程中解决 TODO

        """
        candidates = {}
        for cell in score_dict:
            if score_dict[cell].get('IoU') > 0.1:
                candidates[cell] = sum(score_dict[cell].values())
        if not candidates:
            print("第二分支，重新选择")
            print(score_dict)
            for cell in score_dict:
                if score_dict[cell].get('IoU') > 0:
                    candidates[cell] = sum(score_dict[cell].values())
        if not candidates:
            print("第三分支，重新选择")
            for cell in score_dict:
                candidates[cell] = sum(score_dict[cell].values())
        best = max(candidates, key=candidates.get)
        return best

    def match_one(self, predict_child, candidates):
        if len(candidates) == 1:  # 只有一个候选项，判断应该为准确匹配
            # print('matched single:', self.calc_similar(parent, filtered_candidates[0]))
            score = self.calc_similar(predict_child, candidates[0])
            if score[0] > 0:
                return [(candidates[0], 'ACCURATE')]
            else:
                return [(candidates[0], 'ACCURATE-FL')]
        else:
            return False

    def _match(self, parent: Cell, no_filter_candidates: FeatureExtractor):
        """比较两个细胞的综合相似度
        is_new: 是否为新出现在视野中的细胞
        """
        predict_child = self.predict_next_position(parent)

        filtered_candidates = self.match_candidates(predict_child, no_filter_candidates.cells)
        # print(filtered_candidates)
        if len(filtered_candidates) > 0:
            if not self.match_one(predict_child, filtered_candidates):  # 不只有一个选项
                matched_candidates = self.match_duplicate_child(parent, filtered_candidates)
                if parent.phase != 'M':  # TODO 在不精确的周期信息中，有可能有问题，需要进一步判断是否已经发生了有丝分裂但是母细胞不是处于M期
                    return [(self.select_child(matched_candidates), 'ACCURATE')]
                else:
                    matched_result = []
                    try:
                        sisters, status = self.select_mitosis_cells(parent, filtered_candidates)  # 此时细胞一分为二
                        for sister in sisters:
                            matched_result.append((sister, status))
                    except MitosisError as M:  # 细胞可能仍然处于M期，但是已经完成分开，或者只是被误判为M期
                        matched_result.append((self.select_child(matched_candidates), 'INACCURATE'))
                        print(M)
                    except ErrorMatchMitosis as M2:
                        # 细胞可能不均等分裂
                        matched_result.append((self.select_child(matched_candidates), 'INACCURATE'))
                        print(M2)
                    finally:
                        return matched_result
            else:
                return self.match_one(predict_child, filtered_candidates)
        else:
            # 此处视为没有匹配项，填充预测细胞，
            predict_child_cell = Cell(position=predict_child.position, mcy=predict_child.mcy, dic=predict_child.dic,
                                      phase='predict_' + predict_child.phase, frame_index=predict_child.frame + 1,
                                      flag='gap')
            predict_child_cell.set_feature(predict_child.feature)
            no_filter_candidates.add_cell(predict_child_cell)
            return [(predict_child_cell, 'PREDICTED')]

        #
        #
        # elif len(filtered_candidates) > 1:  # 有多个匹配项
        #     matched_result = []
        #     score_dict = {}
        #
        #     accurate_child = None
        #     for child_cell in filtered_candidates:
        #         score = self.calc_similar(predict_child, child_cell)
        #         score_dict[child_cell] = score
        #         if score[0] > 0:  # 判断候选项是否为相邻细胞造成的重叠
        #             if score[0] > 0.5:
        #                 matched_result.append((child_cell, 'ACCURATE'))
        #                 accurate_child = child_cell
        #                 filtered_candidates.remove(child_cell)

        # 如果发生有丝分裂，则可能会出现两个子细胞IOU都小于阈值，这时候需要单独判断
        # if not accurate_child:
        #     try:
        #         sisters, status = self.select_mitosis_cells(accurate_child, filtered_candidates)
        #         matched_result.append((sisters[0], status))
        #         matched_result.append((sisters[1], status))
        #     except Exception as E:
        #         print(E)
        # else:
        #     select_ret = self.select_child(score_dict)
        #     if select_ret:
        #         matched_result.append((select_ret, 'INACCURATE-MATCH'))
        #         # elif score[0] > 0:
        #         #
        #         # else:
        #         #     matched_result.append((child_cell, 'INACCURATE'))
        # if not matched_result:
        #     for child_cell in filtered_candidates:
        #         if score_dict[child_cell][1] < 0.1:
        #             matched_result.append((child_cell, 'INACCURATE'))
        # if not matched_result:
        #     matched_result = [(child, 'PREDICTED') for child in filtered_candidates]
        # # print('matched single:', self.calc_similar(parent, filtered_candidates[0]))

    def is_new_cell(self):
        """判断是否为新出现的细胞"""

    def handle_duplicate_match(self):
        """解决一个细胞被多个细胞匹配"""

    def handle_loss_match(self):
        """解决细胞没有被匹配上"""

    def rematch(self):
        """对于待匹配帧的的细胞，如果没有被匹配上，或者发生多次匹配，则需要判断是否是新出现的细胞，或者是被遗漏的细胞
        如果是新出现的"""

    def match(self, before_frame: FeatureExtractor, current_frame: FeatureExtractor):
        """

        Args:
            current_frame: 当前帧的所有细胞实例的特征
            next_frame: 下一帧的细胞所有实例特征

        Returns:

        """
        matched = []
        for i in before_frame.cells:
            # print("current count: ", len(before_frame.cells))
            # print("next count: ", len(current_frame.cells))
            child = self._match(i, current_frame)  # [(cell, status)]
            if len(child) < 1:
                # print("child : ", child)
                pass
            # print("bbox: ", i.bbox)
            matched.append([(i, 'parent')] + child)
            # print('child: ', child)
            # print('matched: ', [i] + child)
        return matched


class Tracker(object):
    """Tracker对象，负责从头到尾扫描图像帧，进行匹配并分配track id，初始化并更新TrackingTree"""

    def __init__(self, mcy, dic, annotation):
        self.matcher = Matcher()
        self.trees: TrackingTree
        self.mcy = mcy
        self.dic = dic
        self.annotation = annotation
        self._exist_tree_id = []
        self._available_id = 0
        self.init_flag = False
        self.feature_ext = feature_extract(mcy, dic, annotation)
        self.tree_maps = {}
        self.init_tracking_tree(next(self.feature_ext)[0])

        self.nodes = set()
        self.count = 0

    def id_distributor(self):
        if self._available_id not in self._exist_tree_id:
            self._exist_tree_id.append(self._available_id)
            self._available_id += 1
            return self._available_id - 1
        else:
            i = 1
            while True:
                if self._available_id + i not in self._exist_tree_id:
                    self._available_id += (i + 1)
                    return self._available_id + i
                i += 1

    def init_tracking_tree(self, fe: FeatureExtractor):
        """初始化Tracking Tree"""
        trees = []
        for i in fe.cells:
            tree = TrackingTree(track_id=self.id_distributor())
            node = CellNode(i)
            node.branch_id = 0
            node.status = 'ACCURATE'
            tree.add_node(node)
            trees.append(tree)
            self.tree_maps[i] = tree
        self.init_flag = True
        self.trees = trees

    def draw_bbox(self, bg1, bbox, track_id):
        if len(bg1.shape) > 2:
            im_rgb1 = bg1
        else:
            im_rgb1 = cv2.cvtColor(convert_dtype(bg1), cv2.COLOR_GRAY2RGB)
        cv2.rectangle(im_rgb1, (bbox[2], bbox[0]), (bbox[3], bbox[1]),
                      [255, 0, 0], 2)
        cv2.putText(im_rgb1, str(track_id), (bbox[3], bbox[1]), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 255), 1)
        return im_rgb1

    def track_near_frame(self, fe1: FeatureExtractor, fe2: FeatureExtractor, trees: List[TrackingTree]):
        """跟踪相邻两帧"""
        for ret in self.matcher.match(fe1, fe2):
            parent_cell = ret[0]
            child_cells = ret[1:]
            current_tracking_tree = self.tree_maps.get(parent_cell)
            if current_tracking_tree is None:
                continue
            parent_node = CellNode(parent_cell)
            for cell in child_cells:
                child_node = CellNode(cell)
                if child_node not in current_tracking_tree:
                    current_tracking_tree.add_node(child_node, parent=parent_node.identifier)
                    self.tree_maps[cell] = current_tracking_tree
                else:
                    print(child_node, 'already exist')
            # if len(child_cells) > 1:
            #     print(child_cells, 'cell division', "###" * 100)
            #     print(current_tracking_tree)

    @staticmethod
    def update_speed(parent: Cell, child: Cell, default: Vector = None):
        """更新细胞的移动速度"""
        if default:
            child.update_speed(Vector(0, 0))
        else:
            speed = (child.vector - parent.vector)
            child.update_speed(speed)

    def update_tree_map(self, cell_key, tree_value):
        """更新tree_map"""
        # 废弃的参数，不再依赖tree_map

    def get_current_tree(self, parent_cell: Cell):
        """获取当前母细胞存在的TrackingTree"""
        exist_trees = []  # 细胞可能存在的tree
        for tree in self.trees:
            if parent_cell in tree.last_layer_cell:
                exist_trees.append(tree)
        return exist_trees

    def add_node(self, child_node, parent_node, tree):
        if child_node not in tree:
            tree.add_node(child_node, parent=parent_node.identifier)
        else:
            raise NodeExistError(child_node)

    def track_parent(self, fe1: FeatureExtractor, fe2: FeatureExtractor):
        for ret in self.matcher.match(fe1, fe2):
            parent_cell = ret[0]
            child_cells = ret[1:]
            # current_tracking_tree = self.tree_maps.get(parent_cell[0])  #
            current_tracking_trees = self.get_current_tree(
                parent_cell[0])  # TODO current_tracking_tree有可能为None，此时意味着上一帧的细胞没有被匹配上
            parent_node = CellNode(parent_cell[0])
            branch_id = 1
            for cell in child_cells:
                child_node = CellNode(cell[0])
                if cell[1] != 'INACCURATE':
                    child_node.set_branch_id(0)
                    # self.update_speed(parent_cell[0], cell[0])
                else:
                    child_node.set_branch_id(branch_id)
                    # self.update_speed(parent_cell[0], cell[0], default=Vector(0, 0))
                child_node.set_status(cell[1])
                try:
                    for tree in current_tracking_trees:
                        self.add_node(child_node, parent_node=parent_node, tree=tree)
                    # if child_node not in current_tracking_tree:
                    #     current_tracking_tree.add_node(child_node, parent=parent_node.identifier)
                    #     self.tree_maps[cell[0]] = current_tracking_tree
                    # else:
                    #     # print(child_node, 'already exist')
                    #     new_child_node = CellNode(cell=cell[0], node_type='duplicate cell')
                    #     new_child_node.set_branch_id(0)
                    #     new_child_node.set_status(cell[1])
                    #     current_tracking_tree.add_node(new_child_node, parent=parent_node.identifier)
                    # branch_id += 1
                except TypeError as E:
                    print(E)
                    # pass
                except NodeExistError as E2:
                    print(E2)
                    # pass

    def check_track(self, fe1: FeatureExtractor, fe2: FeatureExtractor, fe3: FeatureExtractor):
        """检查track结果，查看是否有错误匹配和遗漏， 同时更新匹配状态"""
        pass

    def fe_cache(self, reset_flag):
        """缓存已经匹配过的帧，用来做check"""

    def track(self):
        """从头到尾读取图像帧，开始追踪"""
        # matcher = Matcher()
        index = 0

        for fe_before, fe_current, fe_next in self.feature_ext:
            # print(fe_before, fe_current, fe_next)
            print((str(index) + '-') * 100)
            self.track_parent(fe_before, fe_current)
            # self.track_parent(fe_current, fe_next)
            self.check_track(fe_before, fe_current, fe_next)
            # print(fe_before, len(fe_before.cells), fe_current, len(fe_current.cells), fe_next, len(fe_next.cells))
            # result = self.matcher.match(fe_before, fe_current)
            # print(result)
            # self.matcher.match(fe_before.cells, fe_current.cells
            # bg_fname = fr'G:\20x_dataset\copy_of_xy_01\tif\mcy\copy_of_1_xy01-{index:0>4d}.tif'
            # bkground = cv2.imread(bg_fname, -1)
            # for c in fe_before.cells:
            #     bkground = Rectangle(*c.bbox).draw(bkground)
            #     bkground = c.available_range.draw(bkground, color=(255, 255, 255))
            # Rectangle(0,0,0,0).draw(bkground, isShow=True)
            # cv2.imwrite(fr'G:\20x_dataset\copy_of_xy_01\development-dir\test-filter\-test-filter{index}.png', bkground)

            # self.track_near_frame(fe_before, fe_current, self.trees)
            # for c1 in fe_before.cells:
            #     for c2 in fe_current.cells:
            #         print(hash(c1)==hash(c2))
            # for i in self.matcher.match(fe_before.cells, fe_current.cells):
            #     key = list(i.keys())[0]
            #     values = list(i.values())[0]
            #     tree = self.tree_maps.get(key)
            #     # for v in values:
            #         # try:
            #         # tree.add_node(CellNode(v), parent=CellNode(key))
            #         # except:
            #         #     print('error !!!!!!!!!!!!!!!')
            #         # self.tree_maps[v] = tree

            # TODO 处理有丝分裂，检测细胞一分为二，
            #  如果细胞存在一分为二，则一个父节点存在多个子节点，同时不对错误的阳性进行筛查，
            #  如果存在将背景误判为细胞，该分支会很短
            index += 1
            # break
            # if index > 181:
            #     break

        fi = 0
        for i in self.trees:
            # if fi != 5:
            #     fi += 1
            #     continue
            jsf = rf'G:\20x_dataset\copy_of_xy_01\development-dir\track_tree\tree3\tree{fi}.json'
            # print('leaves: ', '###'*20)
            print(i.leaves())
            print(i)
            if os.path.exists(jsf):
                os.remove(jsf)
            i.save2file(jsf)
            fi += 1

        def save_visualization(rg=369):
            bg_fname = [fr'G:\20x_dataset\copy_of_xy_01\tif\sub_mcy\copy_of_1_xy01-{n:0>4d}.tif' for n in range(rg)]
            print(bg_fname)
            images = list(map(lambda x: cv2.imread(x, -1), bg_fname))
            images_dict = dict(zip(list(range(rg)), images))
            print(images_dict.keys())
            tree_index = 0
            for i in self.trees:
                if tree_index == -12:
                    tree_index += 1
                    continue
                else:
                    # print(tree_index)
                    # print(i)
                    for node in i.expand_tree():
                        frame = i.nodes.get(node).cell.frame
                        bbox = i.nodes.get(node).cell.bbox
                        img_bg = images_dict[frame]
                        images_dict[frame] = self.draw_bbox(img_bg, bbox, i.track_id)
                # break

            for i in zip(bg_fname, list(images_dict.values())):
                fname = os.path.join(r'G:\20x_dataset\copy_of_xy_01\development-dir\track_example\t12',
                                     os.path.basename(i[0]).replace('.tif', '.png'))
                # fname = os.path.join(r'G:\20x_dataset\copy_of_xy_01\development-dir\track_example\t5',
                #                      os.path.basename(i[0]))
                print(fname)
                cv2.imwrite(fname, i[1])

        save_visualization()
        # pass
        # im1, im2 = self.draw_bbox(img1, img2, self.trees)
        # cv2.imwrite(r'G:\20x_dataset\copy_of_xy_01\development-dir\track11.png', im1)
        # cv2.imwrite(r'G:\20x_dataset\copy_of_xy_01\development-dir\track22.png', im2)

        # 已实现遍历读取


def test():
    annotation = r'G:\20x_dataset\copy_of_xy_01\copy_of_1_xy01-sub.json'
    mcy_img = r'G:\20x_dataset\copy_of_xy_01\raw\sub_raw\mcy\copy_of_1_xy01.tif'
    dic_img = r'G:\20x_dataset\copy_of_xy_01\raw\sub_raw\dic\copy_of_1_xy01.tif'
    fe = feature_extract(mcy_img, dic_img, annotation)
    ind = 0
    fe0 = next(fe)[1]  # 001
    fe1 = next(fe)[1]  # 012
    fe2 = next(fe)[1]  # 123
    fe3 = next(fe)[1]  # 234
    fe4 = next(fe)[1]  # 345
    # print(fe0)
    # print(fe0.cells)
    # bg = fe0.cells[15].draw(isShow=False)
    # speed = Vector(5, 10)
    # plt.imshow(bg)
    # plt.show()
    # new_cell = fe0.cells[15].move(speed)
    # new_cell.draw(background=bg)
    # plt.imshow(bg)
    # plt.show()

    c1 = fe0.cells[14]
    c2 = fe4.cells[13]

    speed = (c2.vector - c1.vector) * 0.8
    print(speed)

    bg1 = c1.draw()
    bg2 = c2.draw()
    plt.imshow(bg1)
    plt.show()
    plt.imshow(bg2)
    plt.show()

    new_cell = c1.move(speed=speed)
    new_cell.draw(background=bg1, color=(0, 255, 0))
    plt.imshow(bg1)
    plt.show()

    c2.draw(background=bg1, color=(255, 255, 255))
    plt.imshow(bg1)
    plt.show()
    plt.imsave(r'C:\Users\91481\Desktop\move.png', bg1)


if __name__ == '__main__':
    annotation = r'G:\20x_dataset\copy_of_xy_01\copy_of_1_xy01-sub.json'
    mcy_img = r'G:\20x_dataset\copy_of_xy_01\raw\sub_raw\mcy\copy_of_1_xy01.tif'
    dic_img = r'G:\20x_dataset\copy_of_xy_01\raw\sub_raw\dic\copy_of_1_xy01.tif'
    tracker = Tracker(mcy_img, dic_img, annotation)
    tracker.track()

    # test()

    # c1 = CellNode(Cell(position=([1, 2], [3, 4])))
    # c2 = CellNode(Cell(position=([1, 3], [3, 4])))
    # c3 = CellNode(Cell(position=([1, 1], [3, 4])))
    # tree = TrackingTree()
    # tree.add_node(c1)
    # tree.add_node(c2, parent=c1)
    # tree.add_node(c3, parent=c1)
    # print(tree)
    # print(len(tree))
    # print(tree.nodes)
