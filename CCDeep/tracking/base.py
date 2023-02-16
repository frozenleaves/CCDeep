from __future__ import annotations

import math
from abc import ABC, abstractmethod
import warnings
from typing import List, Tuple

import numpy as np
from functools import lru_cache
from matplotlib import pyplot as plt

def convert_dtype(__image: np.ndarray) -> np.ndarray:
    """将图像从uint16转化为uint8"""
    min_16bit = np.min(__image)
    max_16bit = np.max(__image)
    image_8bit = np.array(np.rint(255 * ((__image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit


def NoneTypeFileter(func):
    def _filter(self, *args, **kwargs):
        ret = func(self, *args, **kwargs)
        cells = []
        for i in ret:
            if i.mcy.size != 0:
                cells.append(i)
        return cells

    return _filter


def warningFilter(func):
    warnings.filterwarnings("ignore")

    def _warn_filter(*args, **kwargs):
        return func(*args, **kwargs)

    return _warn_filter


class SingleInstance(object):
    """单例模式基类， 如果参数相同，则只会实例化一个对象"""
    _instances = {}
    init_flag = False
    def __new__(cls, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
            SingleInstance.__init__(SingleInstance._instances[key], *args, **kwargs)
        return cls._instances[key]

    def __init__(self, *args, **kwargs):
        pass


class Rectangle(object):
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    @property
    def area(self):
        return abs((self.x_max - self.x_min) * (self.y_max - self.y_min))

    def _intersectX(self, other):
        """判断两个矩形在X轴是否相交"""
        if max(self.x_min, other.x_min) - min(self.x_max, other.x_max) >= 0:
            return False
        else:
            return True

    def _intersectY(self, other):
        """判断两个矩形在Y轴是否相交"""
        if max(self.y_min, other.y_min) - min(self.y_max, other.y_max) >= 0:
            return False
        else:
            return True

    def _include(self, other, self_max, other_max, self_min, other_min):
        """判断是否包含，传入x值则表示在x轴方向上，传入y值表示在y轴方向上"""
        if self_min >= other_min:
            if self_max <= other_max:
                flag = other
                return True, flag      # indicate longer instance
            else:
                return False
        else:
            if self_min <= other_min:
                if self_max >= other_max:
                    flag = self
                    return True, flag
                else:
                    return False

    def _includeX(self, other):
        """判断两个矩形的X轴是否存在包含关系"""
        return self._include(other, self.x_max, other.x_max, self.x_min, other.x_min)

    def _includeY(self, other):
        """判断两个矩形的Y轴是否存在包含关系"""
        return self._include(other, self.y_max, other.y_max, self.y_min, other.y_min)

    def isIntersect(self, other):
        """判断两个矩形是否相交"""
        if self._intersectX(other) and self._intersectY(other):
            return True
        else:
            return False

    def isInclude(self, other):
        """判断两个矩形是否包含"""
        if not self._includeX(other):
            return False
        else:
            if not self._includeY(other):
                return False
            else:
                if self._includeX(other)[1] != self._includeY(other)[1]:
                    return False
                else:
                    return True

    def draw(self, background=None, isShow=False, color=(255, 0, 0)):
        import cv2
        if background is not None:
            _background = background
        else:
            _background = np.zeros(shape=(100, 100))

        if len(_background.shape) == 2:
            _background = cv2.cvtColor(convert_dtype(_background), cv2.COLOR_GRAY2RGB)
            # (bbox[2], bbox[0]), (bbox[3], bbox[1])
        cv2.rectangle(_background, (self.y_min, self.x_min), (self.y_max, self.x_max), color, 2)
        if isShow:
            cv2.imshow('rec', _background)
            cv2.waitKey()
        return _background


class Vector(np.ndarray):
    """二维平面向量类"""

    def __new__(cls, x=None, y=None, shape=(2,), dtype=float, buffer=None, offset=0,
                strides=None, order=None):
        obj = super().__new__(cls, shape, dtype,
                              buffer, offset, strides, order)
        if x is None and y is None:
            obj.xy = None
            obj.x = x
            obj.y = y
        elif x is None and y:
            obj.xy = [0, y]
            obj.x = 0
            obj.y = y
        elif x and y is None:
            obj.xy = [x, 0]
            obj.x = x
            obj.y = 0
        else:
            obj.xy = [x, y]
            obj.x = x
            obj.y = y
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.info = getattr(obj, 'xy', None)
        self.x = getattr(obj, 'x', None)
        self.y = getattr(obj, 'y', None)

    @property
    def module(self):
        """返回向量的模"""
        if self.xy is None:
            return 0
        return np.linalg.norm([self.x, self.y])

    @property
    def cos(self):
        """返回向量的余弦值"""
        if not self.xy:
            return None
        return self.x / self.module

    def cosSimilar(self, other):
        """比较两个向量的余弦相似度, 取值范围[-1, 1]"""
        if self.xy is None:
            if other.xy is None:
                return None
            elif other == Vector(0, 0):
                return 0
            else:
                return other.cos
        if self == other:
            return 1
        if other == Vector(0, 0):
            return 0
        if self == Vector(0, 0):
            if other.yx is None:
                return 0
            elif other == Vector(0, 0):
                return 1
            else:
                return other.cos
        return np.dot(self.xy, other.xy) / (self.module * other.module)

    def cosDistance(self, other):
        """self和other的余弦距离，数值上等于1减去余弦相似度, 取值范围[0, 2]"""
        return 1 - self.cosSimilar(other)

    def EuclideanDistance(self, other):
        """返回两个向量的欧氏距离"""
        return np.sqrt((abs(self.x - other.x) ** 2 + abs(self.y - other.y) ** 2))

    def __len__(self):
        if self.xy is None:
            return 0
        return len(self.xy)

    def __str__(self):
        if self.xy is None:
            return "None Vector"
        return str(self.xy)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.module < other.module

    def __bool__(self):
        if not self.xy or self == Vector(0, 0):
            return False
        return True

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __neg__(self):
        return Vector(-self.x, -self.y)

    def __mul__(self, other):
        if hasattr(other, '__float__'):
            return Vector(self.x * other, self.y * other)
        elif isinstance(other, Vector):
            return self.x * other.x + self.y * other.y
        else:
            raise TypeError(f'{type(other)} are not support the Multiplication operation with type Vector!')


class Cell(object):
    """
    定义细胞实例
    """
    _instances = {}
    # init_flag = False

    def __new__(cls, *args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cls._instances:
            cls._instances[key] = super(Cell, cls).__new__(cls)
            cls._instances[key].__feature = None
            cls._instances[key].__feature_flag = False
        return cls._instances[key]

    def __init__(self, position=None, mcy=None, dic=None, phase=None, frame_index=None, flag=None):
            # if  Cell.init_flag is False:
            self.position = position  # [(x1, x2 ... xn), (y1, y2 ... yn)]
            self.mcy = mcy
            self.dic = dic
            self.phase = phase
            self.__id = None
            self.frame = frame_index
            self.__track_id = None
            self.__inaccurate_track_id = None
            self.__is_track_id_changeable = True
            self.__parent = None  # 如果细胞发生分裂，则记录该细胞的父细胞的__id
            self.__move_speed = Vector(0, 0)
            if flag is None:
                self.flag = 'cell'
            else:
                self.flag = 'gap'
            Cell.init_flag = True
            # else:
            #     return

    @property
    @lru_cache(maxsize=None)
    def contours(self):
        points = []
        if self.position:
            for j in range(len(self.position[0])):
                x = int(self.position[0][j])
                y = int(self.position[1][j])
                points.append((x, y))
            contours = np.array(points)
            return contours
        else:
            return None

    @property
    def move_speed(self) -> Vector:
        return self.__move_speed

    def update_speed(self, speed: Vector):
        self.__move_speed = speed

    @property
    @lru_cache(maxsize=None)
    def center(self):
        return np.mean(self.position[0]), np.mean(self.position[1])

    @property
    @lru_cache(maxsize=None)
    def available_range(self):
        """指定前后两帧的可匹配范围，默认为细胞横纵坐标的两倍"""
        mult = 1

        x_len = self.bbox[3] - self.bbox[2]
        y_len = self.bbox[1] - self.bbox[0]
        x_min_expand = self.bbox[2] - mult * x_len
        x_max_expand = self.bbox[3] + mult* x_len
        y_min_expand = self.bbox[0] - mult * y_len
        y_max_expand = self.bbox[1] + mult * y_len
        return Rectangle(y_min_expand, y_max_expand, x_min_expand, x_max_expand)

    @staticmethod
    @lru_cache(maxsize=None)
    def polygon_area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    @property
    @lru_cache(maxsize=None)
    def vector(self):
        return Vector(*self.center)

    @property
    def area(self):
        return self.polygon_area(tuple(self.position[0]), tuple(self.position[1]))

    @property
    @lru_cache(maxsize=None)
    def bbox(self):
        """bounding box坐标"""
        x0 = math.floor(np.min(self.position[0])) if math.floor(np.min(self.position[0])) > 0 else 0
        x1 = math.ceil(np.max(self.position[0]))
        y0 = math.floor(np.min(self.position[1]))if math.floor(np.min(self.position[1])) > 0 else 0
        y1 = math.ceil(np.max(self.position[1]))
        return y0, y1, x0, x1

    def move(self, speed: Vector, time: int=1):
        """
        :param speed: 移动速度， Vector对象实例
        :param time: 移动时间，单位为帧
        :return: 移动后新的Cell实例
        """
        new_position = [tuple([i + speed.x * time for i in self.position[0]]), tuple([j + speed.y * time for j in self.position[1]])]
        new_cell =  Cell(position=new_position, mcy=self.mcy, dic=self.dic, phase=self.phase, frame_index=self.frame)
        return new_cell

    def set_feature(self, feature):
        """设置Feature对象"""
        self.__feature = feature
        self.__feature_flag = True

    @property
    def feature(self):
        if self.__feature_flag:
            return self.__feature
        else:
            raise ValueError("No available feature! ")

    def set_track_id(self, __track_id, status: 0|1):
        """为细胞设置track id， 如果status为0表示追踪不精确，可以被修改，如果status为1，表示追踪精确，不允许被修改"""
        if self.__is_track_id_changeable:
            if status == 1:
                self.__track_id = __track_id
                self.__is_track_id_changeable = False
            elif status == 0:
                self.__inaccurate_track_id = __track_id
            else:
                raise ValueError(f'status {status} is invalid!')
        else:
            warnings.warn('cannot change the accurate track_id')

    def set_parent_id(self, __parent_id):
        self.__parent = __parent_id

    def set_id(self, cell_id):
        self.__id = cell_id

    @property
    def cell_id(self):
        """细胞id， 母细胞和子细胞拥有不同的id"""
        return self.__id

    @property
    def parent(self):
        return self.__parent

    @property
    def track_id(self):
        return self.__track_id

    def draw(self, background=None, isShow=False, color=(255, 0, 0)):
        import cv2
        if background is not None:
            _background = background
        else:
            _background = np.ones(shape=(2048, 2048, 3), dtype=np.uint8)

        if len(_background.shape) == 2:
            _background = cv2.cvtColor(convert_dtype(_background), cv2.COLOR_GRAY2RGB)
        # cv2.drawContours(_background, self.contours, -1, color, 3)
        cv2.rectangle(_background, (self.bbox[2], self.bbox[0]), (self.bbox[3], self.bbox[1]), color, 5)
        if isShow:
            cv2.imshow('rec', _background)
            cv2.resizeWindow('rec', 500, 500)
            cv2.waitKey()
        return _background

    def __contains__(self, item):
        return True if self.available_range.isIntersect(Rectangle(*item.bbox)) else False

    def __str__(self):
        if self.position:
            if self.__id:
                return f"Cell {self.__id} at ({self.center[0]: .2f},{self.center[1]: .2f}), frame {self.frame}, {self.phase}"
            else:
                return f" Cell at ({self.center[0]: .2f},{self.center[1]: .2f}), frame {self.frame}, {self.phase}"
        else:
            return "Object Cell"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        if self.position and other.position:
            self_module = np.linalg.norm([np.mean(self.position[0]), np.mean(self.position[1])])
            other_module = np.linalg.norm([np.mean(other.position[0]), np.mean(other.position[1])])
            return self_module < other_module
        else:
            raise ValueError("exist None object of ")

    def __eq__(self, other):
        return self.position == other.position and self.frame == other.frame

    def __hash__(self):
        return id(self)


# class Base(ABC):
#     """
#     定义一些图像的基本操作，包括读写图像，显示图像等
#     """
#
#     @staticmethod
#     @abstractmethod
#     def show(image):
#         plt.imshow(image, cmap='gray')
#         plt.show()
#
#     @abstractmethod
#     def convert_dtype(self, __image: np.ndarray) -> np.ndarray:
#         """将图像从uint16转化为uint8"""
#         min_16bit = np.min(__image)
#         max_16bit = np.max(__image)
#         image_8bit = np.array(np.rint(255 * ((__image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
#         return image_8bit
#
#     def save(self, cell: Cell):
#         """保存图像"""
#         pass
#
#     def read(self, file):
#         pass






