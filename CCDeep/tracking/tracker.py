"""
定义追踪树，追踪节点
每个追踪树root节点起始于第一帧的细胞
起始追踪初始化第一帧识别的细胞数量个树实例

定义树节点， 每个节点包含细胞的详细信息。
以及追踪信息，帧数，分裂信息等


"""
from __future__ import annotations
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from libtiff import TIFF

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')

import dataclasses
import enum
from copy import deepcopy
from functools import wraps, lru_cache

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
from tqdm import tqdm

from CCDeep.utils import convert_dtype, readTif
from CCDeep.tracking.base import Cell, Rectangle, Vector, SingleInstance, CacheData, MatchStatus, TreeStatus
from CCDeep.tracking.t_error import InsertError, MitosisError, NodeExistError, ErrorMatchMitosis, StatusError
from CCDeep.tracking.feature import FeatureExtractor, feature_extract

TEST = False
TEST_INDEX = None
CELL_NUM = 0


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
            cls._instance_[key].status = None
            cls._instance_[key].track_id = None
            cls._instance_[key].__branch_id = None
            cls._instance_[key].parent = None
            cls._instance_[key].childs = []
            cls._instance_[key].add_tree = False  # 如果被添加到TrackingTree中，设置为True
            cls._instance_[key].life = 5  # 每个分支初始生命值为5，如果匹配成功则+1，如果没有匹配上，或者利用缺省值填充匹配，则-1，如如果生命值为0，则该分支不再参与匹配
            cls._instance_[key]._init_flag = False
        return cls._instance_[key]

    def __init__(self, cell: Cell, node_type='cell', fill_gap_index=None):
        if not self._init_flag:
            self.cell = cell
            if node_type == 'gap':
                assert fill_gap_index is not None
            super().__init__(cell)
            self._init_flag = True

    @property
    def identifier(self):
        return str(id(self.cell))

    @property
    def nid(self):
        return self.identifier

    def _set_identifier(self, nid):
        if nid is None:
            self._identifier = str(id(self.cell))
        else:
            self._identifier = nid

    def get_status(self):
        return self.status

    def set_parent(self, parent: CellNode):
        self.parent = parent

    def get_parent(self):
        return self.parent

    def set_childs(self, child: CellNode):
        self.childs.append(child)

    def get_childs(self):
        return self.childs

    def set_tree_status(self, status: TreeStatus):
        self.tree_status = status
        self.add_tree = True

    def get_tree_status(self):
        if self.add_tree:
            return self.tree_status
        return None

    def set_status(self, status):
        if status in self.STATUS:
            self.status = status
        else:
            raise ValueError(f"set error status: {status}")

    def get_branch_id(self):
        return self.__branch_id

    def set_branch_id(self, branch_id):
        self.__branch_id = branch_id
        self.cell.set_branch_id(branch_id)

    def get_track_id(self):
        if self.track_id is None:
            raise ValueError("Don't set the track_id")
        else:
            return self.track_id

    def set_track_id(self, track_id):
        self.track_id = track_id

    def __repr__(self):
        if self.add_tree:
            return f"Cell Node of {self.cell}, status: {self.get_tree_status()}"
        else:
            return f"Cell Node of {self.cell}"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return int(id(self))


class TrackingTree(Tree):
    """追踪树，起始的时候初始化根节点，逐帧扫描的时候更新left子节点，发生分裂的时候添加right子节点"""

    def __init__(self, root: CellNode = None, track_id=None):
        super().__init__()
        self.root = root
        self.track_id = track_id
        self.mitosis_start_flag = False
        self.status = TreeStatus(self)
        self.__last_layer = []
        self._exist_branch_id = []
        self._available_branch_id = 1

    def __contains__(self, item):
        return item.identifier in self._nodes

    def change_mitosis_flag(self, flag: bool):
        """当细胞首次进入mitosis的时候，self.mitosis_start_flag设置为True， 当细胞完成分裂的时候，重新设置为false"""
        self.mitosis_start_flag = flag

    def add_node(self, node: CellNode, parent: CellNode = None):
        node.set_parent(parent)
        if parent != None:
            parent.childs.append(node)
        super().add_node(node, parent)

    def get_parent(self, node: CellNode):
        return node.get_parent()

    def get_childs(self, node: CellNode):
        return node.get_childs()

    @property
    def last_layer(self):
        return self.__last_layer

    @property
    def last_layer_cell(self):
        """返回{叶节点：节点包含的细胞}字典"""
        cells = {}
        for node in self.leaves():
            cells[node.cell] = node
        return cells

    def update_last_layer(self, node_list: List[CellNode]):
        self.__last_layer = node_list

    def auto_update_last_layer(self):
        self.__last_layer = self.leaves()

    def branch_id_distributor(self):
        if self._available_branch_id not in self._exist_branch_id:
            self._exist_branch_id.append(self._available_branch_id)
            self._available_branch_id += 1
            return self._available_branch_id - 1
        else:
            i = 1
            while True:
                if self._available_branch_id + i not in self._exist_branch_id:
                    self._available_branch_id += (i + 1)
                    return self._available_branch_id + i
                i += 1


class MatchRecorder(object):
    """
    匹配记录器，每匹配一帧，更新一下匹配记录。
    记录内容包括：已经完成匹配的细胞，没有匹配的细胞，匹配的精确度
    """

    def __init__(self):
        pass


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


class Match(object):
    """
    匹配器，根据前后帧及当前帧来匹配目标并分配ID
    主要用来进行特征比对，计算出相似性
    操作单位：帧
    """
    _instances = {}

    def __new__(cls, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

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
        if dic1 and dic2:
            return np.sin(self.normalize(dic1.cosSimilar(dic2)))
        else:
            return 0

    def compareMcySimilar(self, cell_1: Cell, cell_2: Cell):
        """
        比较mcy相似度
        返回值范围: float(0, 1)
        """
        mcy1 = Vector(cell_1.feature.mcy_intensity, cell_1.feature.mcy_variance)
        mcy2 = Vector(cell_2.feature.mcy_intensity, cell_2.feature.mcy_variance)
        if mcy1 and mcy2:
            return np.sin(self.normalize(mcy1.cosSimilar(mcy2)))
        else:
            return 0

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

    def __str__(self):
        return f"Match object at {id(self)}"


class Matcher(object):
    """拿子细胞找母细胞，根据下一帧匹配上一帧"""

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
        # new_cell = parent.move(parent.move_speed, 1)
        new_cell = parent
        return new_cell

    def _filter(self, child: Cell, cells: List[Cell]):
        """

        :param current: 待匹配的细胞
        :param cells: 候选匹配项
        :return: 筛选过的候选匹配项
        """
        return [cell for cell in cells if cell in child]

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
        一旦调用这个函数，意味着候选项中不止一个，此方法计算每个候选项的匹配度
        返回值为{Cell: similar_dict}形式的字典
        """
        matched = {}
        for i in unmatched_child_list:
            similar = self.match_similar(parent, i)
            matched[i] = similar
        return matched

    def is_mitosis_start(self, pre_parent: Cell, last_leaves: List[Cell], area_size_t=1.8, iou_t=0.5):
        """判断细胞是否进入M期，核心依据是细胞进入Mitosis的时候，体积会变大
        :returns 如果成功进入M期，返回包含最后一帧的G2和第一帧M的字典信息， 否则，返回False
        """
        match_score = {}
        for i in last_leaves:
            if self.match_similar(pre_parent, i).get('IoU') >= iou_t:
                match_score[i] = self.match_similar(pre_parent, i)
        for child_cell in match_score:
            # if Rectangle(*parent.bbox).isInclude(Rectangle(*child_cell.bbox)) or (
            #         (child_cell.area / parent.area) >= area_size_t):
            if (child_cell.area / pre_parent.area) >= area_size_t:
                return {'last_G2': pre_parent, 'first_M': child_cell}
        return False

    def get_similar_sister(self, parent: Cell, matched_cells_dict: dict, area_t=0.7, shape_t=0.03, area_size_t=1.3,
                           iou_t=0.1):
        """在多个候选项中找到最相似的两个细胞作为子细胞"""
        cell_dict_keys = list(matched_cells_dict.keys())
        cell_dict_keys.sort(key=lambda cell: cell.area, reverse=True)

        for cell in cell_dict_keys:
            if matched_cells_dict[cell].get('IoU') < iou_t:
                cell_dict_keys.remove(cell)
        if len(cell_dict_keys) > 2:
            if cell_dict_keys[0].area + cell_dict_keys[1].area > parent.area * area_size_t:
                cell_dict_keys.remove(cell_dict_keys[0])
        length = len(cell_dict_keys)
        match_result = {}
        for i in range(length - 1):
            cell_1 = cell_dict_keys.pop(0)
            for j in range(len(cell_dict_keys)):
                cell_2 = cell_dict_keys[j]
                score = self.match_similar(cell_1, cell_2)
                if score.get('area') >= area_t and score.get('shape') <= shape_t:
                    match_result[(cell_1, cell_2)] = score.get('area') + score.get('IoU')
        if match_result:
            max_score_cells = max(match_result, key=match_result.get)
            return max_score_cells
        else:
            raise MitosisError('cannot match the suitable daughter cells')

    def select_mitosis_cells(self, parent: Cell, candidates_child_list: List[Cell], area_t=0.7, shape_t=0.05,
                             area_size_t=1.3):
        """
        如果发生有丝分裂，选择两个子细胞， 调用这个方法的时候，确保大概率发生了分裂事件
        如果返回值为有丝分裂，则需要检查两个子细胞的面积是否正常，如果某一个子细胞的面积过大，则认为是误判

        :return ([cell_1, cell2], 'match status')
        """

        matched_candidates = self.match_duplicate_child(parent, candidates_child_list)
        matched_cells_dict = self.check_iou(matched_candidates)
        if not matched_cells_dict:
            raise MitosisError('not enough candidates')
        else:
            if len(matched_cells_dict) == 2:
                cells = list(matched_cells_dict.keys())
                if max([i.area for i in
                        list(matched_cells_dict.keys())]) * area_size_t > parent.area:  # 如果母细胞太小了，不认为会发生有丝分裂，转向单项判断
                    raise MitosisError('The cell is too small to have cell division !')
                if self.matcher.calcAreaSimilar(cells[0], cells[1]) > area_t and self.matcher.compareShapeSimilar(
                        cells[0], cells[1]) < shape_t:
                    return cells, 'ACCURATE'
                else:
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

    def select_single_child(self, score_dict):
        """
        对于有多个IOU匹配的选项，选择相似度更大的那一个, 此处是为了区分发生重叠的细胞，而非发生有丝分裂.
        注意：这个方法匹配出来的结果不一定是准确的，有可能因为细胞交叉导致发生错配，需要在后面的流程中解决
        另外，如果一个细胞被精确匹配后，另一个细胞在没有匹配项的时候（即在识别过程中，下一帧没有识别上，可能会出现重复匹配）
        这种情况原本应该填充预测细胞。

        """
        candidates = {}
        for cell in score_dict:
            if score_dict[cell].get('IoU') > 0.5:
                candidates[cell] = sum(score_dict[cell].values())
        if not candidates:
            # print("第二分支，重新选择")
            for cell in score_dict:
                if score_dict[cell].get('IoU') > 0.1:
                    candidates[cell] = sum(score_dict[cell].values())
        if not candidates:
            # print("第三分支，重新选择")
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

    def check_iou(self, similar_dict):
        """检查IoU，为判断进入有丝分裂提供依据，如果拥有IoU>0的匹配项少于2，返回False，否则，返回这两个细胞"""
        matched = {}
        for cell in similar_dict:
            score = similar_dict[cell]
            if score.get('IoU') > 0.1:
                matched[cell] = score
        if len(matched) < 2:
            return False
        else:
            return matched

    def _match(self, parent: Cell, filter_candidates_cells: List[Cell], cell_track_status: TreeStatus):
        """比较两个细胞的综合相似度
        is_new: 是否为新出现在视野中的细胞
        """
        # predict_child = self.predict_next_position(parent)
        predict_child = parent
        # filtered_candidates = self.match_candidates(predict_child, no_filter_candidates_cells)
        filtered_candidates = filter_candidates_cells
        # print(filtered_candidates)
        if len(filtered_candidates) > 0:
            if not self.match_one(predict_child, filtered_candidates):  # 不只有一个选项
                matched_candidates = self.match_duplicate_child(predict_child, filtered_candidates)
                if not cell_track_status.status.get('enter_mitosis'):
                    # if parent.phase != 'M':
                    return {'matched_cell': [(self.select_single_child(matched_candidates), 'ACCURATE')],
                            'status': cell_track_status}
                # elif not self.check_iou(matched_candidates):
                #     return {'matched_cell': [(self.select_single_child(matched_candidates), 'ACCURATE')],
                #             'status': cell_track_status}
                else:
                    matched_result = []
                    try:
                        sisters, status = self.select_mitosis_cells(predict_child, filtered_candidates)  # 此时细胞一分为二
                        cell_track_status.exit_mitosis(parent.frame + 1)
                        for sister in sisters:
                            matched_result.append((sister, status))

                    except MitosisError as M:  # 细胞可能仍然处于M期，但是已经完成分开，或者只是被误判为M期
                        matched_result.append((self.select_single_child(matched_candidates), 'INACCURATE'))
                        # print(M)
                    except ErrorMatchMitosis as M2:
                        # 细胞可能不均等分裂
                        matched_result.append((self.select_single_child(matched_candidates), 'INACCURATE'))
                        # print(M2)
                    finally:
                        return {'matched_cell': matched_result, 'status': cell_track_status}
            else:
                return {'matched_cell': self.match_one(predict_child, filtered_candidates), 'status': cell_track_status}
        else:
            # 此处视为没有匹配项，填充预测细胞，
            if predict_child.phase.startswith('predict'):
                predict_child_phase = predict_child.phase
            else:
                predict_child_phase = 'predict_' + predict_child.phase
            predict_child_cell = Cell(position=predict_child.position, mcy=predict_child.mcy, dic=predict_child.dic,
                                      phase=predict_child_phase, frame_index=predict_child.frame + 1,
                                      flag='gap')
            predict_child_cell.set_feature(predict_child.feature)
            predict_child_cell.set_region(predict_child.region)
            # no_filter_candidates.add_cell(predict_child_cell)
            return {'matched_cell': [(predict_child_cell, 'PREDICTED')], 'status': cell_track_status}

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

    def add_child_node(self, tree, child_node: CellNode, parent_node: CellNode):
        try:
            tree.add_node(child_node, parent=parent_node)
            child_node.set_tree_status(tree.status)
        # except TypeError as E:
        #     print(E)
        # except NodeExistError as E2:
        #     print(E2)
        except treelib.exceptions.DuplicatedNodeIdError:
            pass

    def match_single_cell(self, tree: TrackingTree, current_frame: FeatureExtractor):
        """追踪一个细胞的变化情况"""
        cells = current_frame.cells
        parents = tree.last_layer_cell
        for parent in parents:
            # print(f'\nparent cell math status: {parent.is_be_matched}')
            if parent.phase == 'M':
                tree.status.add_M_count()
            if tree.status.predict_M_len >= 3:
                tree.status.enter_mitosis(parent.frame - 2)

            # predict_child = self.predict_next_position(parent)
            predict_child = parent

            filtered_candidates = self.match_candidates(predict_child, cells)
            before_parent = tree.get_parent(parents[parent])
            if before_parent:
                if self.is_mitosis_start(before_parent.cell, [predict_child]):
                    # if self.is_mitosis_start(predict_child, filtered_candidates):
                    tree.status.enter_mitosis(parent.frame)
            match_result = self._match(predict_child, filtered_candidates, tree.status)
            child_cells = match_result.get('matched_cell')
            parent_node = CellNode(parent)
            if len(child_cells) == 1:
                if child_cells[0][1] == 'PREDICTED':
                    current_frame.add_cell(child_cells[0][0])
                    child_node = CellNode(child_cells[0][0])
                    child_node.life -= 1
                else:
                    child_node = CellNode(child_cells[0][0])
                # child_node.set_branch_id(parent_node.get_branch_id())
                child_node.cell.set_branch_id(parent_node.cell.branch_id)
                child_node.cell.set_status(tree.status)
                child_node.cell.update_region(track_id=tree.track_id)
                child_node.cell.update_region(branch_id=parent_node.cell.branch_id)
                child_node.cell.set_match_status(child_cells[0][1])
                if child_node.life > 0:
                    self.add_child_node(tree, child_node, parent_node)
                # child_node.branch_id = parent_node.branch_id
            else:
                try:
                    assert len(child_cells) == 2
                except AssertionError:
                    # self._match(predict_child, filtered_candidates, tree.status)
                    pass

                for cell in child_cells:
                    new_branch_id = tree.branch_id_distributor()
                    cell[0].set_branch_id(new_branch_id)
                    cell[0].update_region(track_id=tree.track_id)
                    cell[0].update_region(branch_id=new_branch_id)
                    cell[0].set_match_status(True)
                    child_node = CellNode(cell[0])
                    self.add_child_node(tree, child_node, parent_node)
        tree.status.add_exist_time()

    def match(self, before_frame: FeatureExtractor, current_frame: FeatureExtractor):
        """

        Args:
            current_frame: 当前帧的所有细胞实例的特征
            next_frame: 下一帧的细胞所有实例特征

        Returns:

        """
        pass
        # matched = []
        # for i in before_frame.cells:
        #     # print("current count: ", len(before_frame.cells))
        #     # print("next count: ", len(current_frame.cells))
        #     cells = current_frame.cells
        #     predict_child = self.predict_next_position(i)
        #     filtered_candidates = self.match_candidates(predict_child, cells)
        #     if self.is_mitosis_start(predict_child, cells):
        #         pass
        #     child = self._match(i, filtered_candidates)  # [(cell, status)]
        #     for c in child:
        #         if c[1] == 'PREDICTED':
        #             current_frame.add_cell(c[0])
        #     # print("bbox: ", i.bbox)
        #     matched.append([(i, 'parent')] + child)
        #     # print('child: ', child)
        #     # print('matched: ', [i] + child)
        # return matched


class Tracker(object):
    """Tracker对象，负责从头到尾扫描图像帧，进行匹配并分配track id，初始化并更新TrackingTree"""

    def __init__(self, annotation, mcy=None, dic=None):
        self.matcher = Matcher()
        self.trees: TrackingTree
        self.mcy = mcy
        self.dic = dic
        self.annotation = annotation
        self._exist_tree_id = []
        self._available_id = 0
        self.init_flag = False
        self.feature_ext = feature_extract(mcy=self.mcy, dic=self.dic, jsonfile=self.annotation)
        self.tree_maps = {}
        self.init_tracking_tree(next(self.feature_ext)[0])

        self.nodes = set()
        self.count = 0
        self.parser_dict = None

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
            i.set_match_status('ACCURATE')
            i.set_track_id(tree.track_id, 1)
            i.set_branch_id(0)
            i.set_cell_id(str(i.track_id) + '-' + str(i.branch_id))
            i.update_region(track_id=tree.track_id)
            i.update_region(branch_id=0)
            node = CellNode(i)
            node.set_branch_id(0)
            node.set_track_id(tree.track_id)
            node.status = 'ACCURATE'
            tree.add_node(node)
            trees.append(tree)
            self.tree_maps[i] = tree
        self.init_flag = True
        self.trees = trees

    def draw_bbox(self, bg1, cell: Cell, track_id, branch_id=None, phase=None):
        bbox = cell.bbox
        if len(bg1.shape) > 2:
            im_rgb1 = bg1
        else:
            im_rgb1 = cv2.cvtColor(convert_dtype(bg1), cv2.COLOR_GRAY2RGB)
        cv2.rectangle(im_rgb1, (bbox[2], bbox[0]), (bbox[3], bbox[1]),
                      [0, 255, 0], 2)

        def get_str():
            raw = str(track_id)
            if branch_id is not None:
                raw += '-'
                raw += str(branch_id)
            if phase is not None:
                raw += '-'
                raw += str(phase)
            return raw

        text = get_str()
        cv2.putText(im_rgb1, text, (bbox[3], bbox[1]), cv2.FONT_HERSHEY_COMPLEX,
                    0.75, (255, 255, 255), 2)
        return im_rgb1

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
            # tree.add_node(child_node, parent=parent_node.identifier)
            tree.add_node(child_node, parent=parent_node)
        else:
            raise NodeExistError(child_node)

    def track_near_frame(self, fe1: FeatureExtractor, fe2: FeatureExtractor):
        """匹配临近帧"""
        for parent in fe1.cells:
            # print(parent)
            trees = self.get_current_tree(parent)
            for tree in trees:
                self.matcher.match_single_cell(tree, fe2)

    def track_near_frame_mult_thread(self, fe1: FeatureExtractor, fe2: FeatureExtractor):
        """多线程测试版"""

        def work(__parent: Cell):
            trees = self.get_current_tree(__parent)
            for tree in trees:
                self.matcher.match_single_cell(tree, fe2)

        thread_pool_executor = ThreadPoolExecutor(max_workers=50, thread_name_prefix="track_")
        for parent in fe1.cells:
            thread_pool_executor.submit(work, parent)
        thread_pool_executor.shutdown(wait=True)

    def is_new_cell(self):
        """判断是否为新出现的细胞"""

    def handle_duplicate_match(self, duplicate_match_cell):
        """解决一个细胞被多个细胞匹配"""
        child_node = CellNode(duplicate_match_cell)
        tmp = self.get_current_tree(duplicate_match_cell)
        parent0 = tmp[0].parent(child_node.nid)
        parent1 = tmp[1].parent(child_node.nid)
        tree_dict = {parent0: tmp[0], parent1: tmp[1]}
        sm0 = self.matcher.match_similar(duplicate_match_cell, parent0.cell)
        sm1 = self.matcher.match_similar(duplicate_match_cell, parent1.cell)
        match_score = {parent0: sm0['IoU'] + sm0['area'] + (1 - sm0['shape']),
                       parent1: sm1['IoU'] + sm1['area'] + (1 - sm1['shape'])}
        truth_parent = min(match_score)
        error_parent = max(match_score)
        if len(tree_dict[truth_parent].nodes) < 3:
            error_parent = truth_parent
        tree_dict[error_parent].remove_node(child_node.nid)
        return {error_parent: tree_dict[error_parent]}

    def handle_loss_match(self):
        """解决细胞没有被匹配上"""

    def rematch(self, error_match_parent: CellNode, error_match_parent_tree: TrackingTree, exclusion_candidate: Cell,
                fe2: FeatureExtractor):
        """对于发生错配的细胞，重新匹配，候选项去掉了正确匹配的"""
        tree = error_match_parent_tree
        parent = error_match_parent
        # candidate = self.matcher.match_candidates()
        fc = [cell for cell in fe2.cells if cell in parent.cell]
        # print(fc)
        if exclusion_candidate in fc:
            # print('exclusion_candidate: ', exclusion_candidate)
            fc.remove(exclusion_candidate)
        if fc:
            try:
                self.add_node(CellNode(fc[0]), parent_node=parent, tree=tree)
                fc[0].set_match_status('INACCURATE')
            except NodeExistError:
                pass

    def check_track(self, fe1: FeatureExtractor, fe2: FeatureExtractor, fe3: FeatureExtractor):
        """检查track结果，查看是否有错误匹配和遗漏， 同时更新匹配状态"""
        for cell in fe2.cells:
            tmp = self.get_current_tree(cell)
            if len(tmp) > 1:
                # print(list(i.track_id for i in tmp))
                err_match_dict = self.handle_duplicate_match(duplicate_match_cell=cell)
                err_parent, err_tree = err_match_dict.popitem()
                # self.rematch(err_parent, err_tree, cell, fe2)
        for cell in fe1.cells:
            if cell.is_be_matched is False:
                # print(cell)
                tree = TrackingTree(track_id=self.id_distributor())
                cell.set_track_id(tree.track_id, 1)
                cell.set_branch_id(0)
                cell.set_cell_id(str(cell.track_id) + '-' + str(cell.branch_id))
                cell.update_region(track_id=tree.track_id)
                cell.update_region(branch_id=0)
                node = CellNode(cell)
                node.set_branch_id(0)
                node.set_track_id(tree.track_id)
                node.status = 'ACCURATE'
                tree.add_node(node)
                self.trees.append(tree)
                self.tree_maps[cell] = tree
                cell.set_match_status('INACCURATE')
                self.matcher.match_single_cell(tree, fe2)

    def fe_cache(self, reset_flag):
        """缓存已经匹配过的帧，用来做check"""

    def track(self, range=None):
        """顺序读取图像帧，开始追踪"""
        index = 0
        for fe_before, fe_current, fe_next in tqdm(self.feature_ext, total=range, desc='tracking process'):
            # self.track_near_frame(fe_before, fe_current)
            self.track_near_frame_mult_thread(fe_before, fe_current)
            self.check_track(fe_before, fe_current, fe_next)
            if range:
                index += 1
                if index > range:
                    break

    def track_tree_to_json(self, filepath):
        fi = 0
        for i in self.trees:
            # jsf = rf'G:\20x_dataset\copy_of_xy_01\development-dir\track_tree\tree4\tree{fi}.json'
            jsf = os.path.join(filepath, f'tree-{fi}.json')
            if os.path.exists(jsf):
                os.remove(jsf)
            i.save2file(jsf)
            fi += 1

    def visualize(self, background_filename_list, save_dir, tree_list):

        # bg_fname = [fr'G:\20x_dataset\copy_of_xy_01\tif\sub_mcy\copy_of_1_xy01-{n:0>4d}.tif' for n in range(export_range)]
        bg_fname = background_filename_list
        # bg_fname = [fr'F:\wangjiaqi\src\s6\tif-seq\mcy-{n:0>4d}.tif' for n in range(rg)]
        images = list(map(lambda x: cv2.imread(x, -1), bg_fname))
        images_dict = dict(zip(list(range(len(bg_fname))), images))
        print(images_dict.keys())
        for i in tree_list:
            for node in i.expand_tree():
                frame = i.nodes.get(node).cell.frame
                bbox = i.nodes.get(node).cell.bbox
                img_bg = images_dict[frame]
                images_dict[frame] = self.draw_bbox(img_bg, i.nodes.get(node).cell, i.track_id,
                                                    i.get_node(node).cell.branch_id,
                                                    phase=i.get_node(node).cell.phase)

        for i in zip(bg_fname, list(images_dict.values())):
            fname = os.path.join(save_dir, os.path.basename(i[0]).replace('.tif', '.png'))
            # fname = os.path.join(r'F:\wangjiaqi\src\s6\track-png\t0',os.path.basename(i[0]).replace('.tif', '.png'))
            # fname = os.path.join(r'G:\20x_dataset\copy_of_xy_01\development-dir\track_example\t5',
            #                      os.path.basename(i[0]))
            # print(fname)
            cv2.imwrite(fname, i[1])

    def visualize_single_tree(self, tree, background_filename_list, save_dir):
        bg_fname = background_filename_list
        images = list(map(lambda x: cv2.imread(x, -1), bg_fname))
        images_dict = dict(zip(list(range(len(bg_fname))), images))
        # print(images_dict.keys())
        for node in tree.expand_tree():
            frame = tree.nodes.get(node).cell.frame
            bbox = tree.nodes.get(node).cell.bbox
            img_bg = images_dict[frame]
            phase = tree.nodes.get(node).cell.phase
            images_dict[frame] = self.draw_bbox(img_bg, tree.nodes.get(node).cell, tree.track_id,
                                                tree.get_node(node).cell.branch_id, phase)
        for i in zip(bg_fname, list(images_dict.values())):
            fname = os.path.join(save_dir, os.path.basename(i[0]).replace('.tif', '.png'))
            print(fname)
            cv2.imwrite(fname, i[1])

    def save_visualize(self, track_range, tree=None):
        export_range = track_range + 2
        if export_range:
            bg_fname = [fr'G:\20x_dataset\copy_of_xy_01\tif\sub_mcy\copy_of_1_xy01-{n:0>4d}.tif' for n in
                        range(export_range)]
        else:
            bg_fname = [os.path.join(r'G:\20x_dataset\copy_of_xy_01\tif\sub_mcy') for i in
                        os.listdir(r'G:\20x_dataset\copy_of_xy_01\tif\sub_mcy')]
        save_dir = r'G:\20x_dataset\copy_of_xy_01\development-dir\track_example\t20'
        if type(tree) is TrackingTree:
            self.visualize_single_tree(tree, bg_fname, save_dir)
        elif type(tree) is list:
            self.visualize(bg_fname, save_dir, tree_list=tree)
        else:
            self.visualize(bg_fname, save_dir, tree_list=self.trees)

    def visualize_to_tif(self, background_mcy_image: str, output_tif_path, tree_list, xrange=None, single=False):
        def adjust_gamma(__image, gamma=1.0):
            image = convert_dtype(__image)
            brighter_image = np.array(np.power((image / 255), 1/gamma) * 255, dtype=np.uint8)
            return brighter_image

        tif = readTif(background_mcy_image)
        images_dict = {}
        index = 0

        if xrange is not None:
            for img, _ in tif:
                if index >= xrange:
                    break
                alpha = 1.5  # 对比度因子
                beta = 50  # 亮度因子
                img= adjust_gamma(img, gamma=1.5)
                images_dict[index] = img
                index += 1
        else:
            for img, _ in tif:
                images_dict[index] = img
                index += 1
        for i in tree_list:
            for node in i.expand_tree():
                frame = i.nodes.get(node).cell.frame
                if xrange:
                    if frame > xrange:
                        continue
                # bbox = i.nodes.get(node).cell.bbox
                img_bg = images_dict[frame]
                images_dict[frame] = self.draw_bbox(img_bg, i.nodes.get(node).cell, i.track_id,
                                                    i.get_node(node).cell.branch_id,
                                                    phase=i.get_node(node).cell.phase)
        if not single:
            if not (os.path.exists(output_tif_path) and os.path.isdir(output_tif_path)):
                os.mkdir(output_tif_path)
            for i in tqdm(range(index), desc="save tracking visualization"):
                fname = os.path.join(output_tif_path, f'{os.path.basename(output_tif_path)[:-4]}-{i:0>4d}.tif')
                tifffile.imwrite(fname, images_dict[i])
        else:
            with tifffile.TiffWriter(output_tif_path) as tif:
                for i in tqdm(range(index)):
                    if i > 300:
                        warnings.warn(
                            "the image is to big to save, and the tifffile cannot save the size >4GB tifffile, "
                            "so this image will be cut down.")
                        break
                    tif.write(images_dict[i])


def get_cell_line_from_tree(tree: TrackingTree, dic_path: str, mcy_path: str, savepath):
    """从track tree中获取完整的细胞序列，包括细胞图像，dic和mcy双通道，以及周期，生成的文件名以track_id-branch_id-frame-phase.tif命名"""
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    save_mcy = os.path.join(savepath, 'mcy')
    save_dic = os.path.join(savepath, 'dic')
    if not os.path.exists(save_mcy):
        os.mkdir(save_mcy)
    if not os.path.exists(save_dic):
        os.mkdir(save_dic)
    mcy = tifffile.imread(mcy_path)
    dic = tifffile.imread(dic_path)
    for nid in tree.expand_tree():
        cell = tree.get_node(nid).cell
        y0, y1, x0, x1 = cell.bbox
        mcy_img = mcy[cell.frame][y0: y1, x0: x1]
        dic_img = dic[cell.frame][y0: y1, x0: x1]
        fname = str(tree.track_id) + '-' + str(cell.branch_id) + '-' + str(cell.frame) + '-' + str(cell.phase[0]) + '.tif'
        tifffile.imwrite(os.path.join(save_mcy, fname), convert_dtype(mcy_img))
        tifffile.imwrite(os.path.join(save_dic, fname), convert_dtype(dic_img))


        # break


if __name__ == '__main__':
    annotation = r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\result-GT.json'
    mcy_img = r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\mcy.tif'
    dic_img = r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\dic.tif'
    tracker = Tracker(annotation)
    # tracker = Tracker(r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\result-GT.json')
    tracker.track(300)
    for i in enumerate(tracker.trees):
        get_cell_line_from_tree(i[1], dic_img, mcy_img,
                                fr'G:\20x_dataset\evaluate_data\copy_of_1_xy19\cell_lines\{i[0]}')
    # tracker.track_tree_to_json(r'G:\20x_dataset\copy_of_xy_01\development-dir\track_tree\tree5')
    # tracker.save_visualize(200)
    # for i in tracker.trees:
    #     print(i)
    #     print(i.nodes)
    #     node_r = i.nodes[list(i.nodes.keys())[0]]
    #     print(node_r)
    #     node_n = CellNode(i.nodes[list(i.nodes.keys())[0]].cell)
    #     print(node_n)
    #     break

    # track_jiaqi()
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
