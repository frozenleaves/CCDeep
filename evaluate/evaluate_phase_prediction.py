import hashlib
import json
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from CCDeep import utils, config
from sklearn.preprocessing import LabelEncoder, label_binarize

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report

# {
#             filename: {
#                 "filename": self.imageName,
#                 "size": 4194304,
#                 "regions": [region],
#                 "file_attributes": {}
#             }
#         }

# region:
# {
#             "shape_attributes":
#                 {
#                     "name": "polygon",
#                     "all_points_x": [],
#                     "all_points_y": []
#                 },
#             "region_attributes":
#                 {
#                     "phase": None
#                 }
#         }
from CCDeep.train_classify import get_model


class Predictor:
    """
    一个预测器，用于预测每个细胞所处的细胞周期，默认不需要提供任何参数，如需改变，请修改config.py相应的值
    """

    def __init__(self):
        self.model = get_model()
        self.model.load_weights(filepath=r'E:\PycharmProjects\CCDeep\models\classify\20x\best\model')


    def predict(self, images):
        """
        :param images: 一个包含多张图片的数组或列表，其形状为[image_count, image_width, image_height, image_channels]
        :return: 每个细胞的预测周期组成的列表
        """
        phaseMap = {0: 'G1/G2', 1: 'M', 2: 'S'}
        img = images
        # img = cv2.resize(img, (128, 128)) / 255.0
        tensor = tf.convert_to_tensor(img, dtype=tf.float64)

        prediction = self.model(tensor, training=False)
        # print(prediction)
        phases = []
        prob = []
        for i in prediction:
            prob.append(i.numpy())
            phase = np.argwhere(i == np.max(i))[0][0]
            # print(phase)
            # print(phaseMap[phase])
            phases.append(phaseMap.get(phase))
        return phases, prob


def __convert_dtype(__image: np.ndarray) -> np.ndarray:
    """将图像从uint16转化为uint8"""
    min_16bit = np.min(__image)
    max_16bit = np.max(__image)
    image_8bit = np.array(np.rint(255 * ((__image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
    return image_8bit


def ellipse_points(center, rx, ry, num_points=100, theta=0):
    all_x = []
    all_y = []
    for i in range(num_points):
        t = i * 2 * np.pi / num_points
        x = center[0] + rx * np.cos(t) * np.cos(theta) - ry * np.sin(t) * np.sin(theta)
        y = center[1] + rx * np.cos(t) * np.sin(theta) + ry * np.sin(t) * np.cos(theta)
        all_x.append(x)
        all_y.append(y)
    return all_x, all_y

def get_cells(annotation: dict, dic: np.ndarray, mcy: np.ndarray):
    cells_data = []
    for region in annotation.get("regions"):
        if region.get("shape_attributes").get("name") == "polygon":
            all_x = region.get("shape_attributes").get("all_points_x")
            all_y = region.get("shape_attributes").get("all_points_y")
            phase = region['region_attributes']['phase']
        elif region.get("shape_attributes").get("name") == 'ellipse':
            rx = region['shape_attributes'].get('rx')
            ry = region['shape_attributes'].get('ry')
            cx = region['shape_attributes'].get('cx')
            cy = region['shape_attributes'].get('cy')
            theta = region['shape_attributes'].get('theta')
            phase = region['region_attributes']['phase']
            all_x, all_y = ellipse_points((cx, cy), rx, ry, num_points=32, theta=theta)
        else:
            continue
        x0 = math.floor(np.min(all_y))
        x1 = math.ceil(np.max(all_y))
        y0 = math.floor(np.min(all_x))
        y1 = math.ceil(np.max(all_x))
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        __mcy = mcy[x0:x1, y0:y1]
        __dic = dic[x0:x1, y0:y1]
        mcy8 = __convert_dtype(__mcy)
        dic8 = __convert_dtype(__dic)
        instance_id = hashlib.md5(str(region).encode()).hexdigest()
        data = utils.Data()
        data.image_dic = cv2.resize(dic8, (config.image_width, config.image_height)) / 255
        data.image_mcy = cv2.resize(mcy8, (config.image_width, config.image_height)) / 255
        data.image_id = instance_id
        data.phase = phase
        cells_data.append(data)
    return cells_data


def predict(cells_data):
    predictor = Predictor()
    image_data = []
    id_data = []
    for cell in cells_data:
        data = np.dstack([cell.image_dic, cell.image_mcy])
        image_data.append(data)
        id_data.append(cell.image_id)
    pred, prob = predictor.predict(image_data)
    truth = [i.phase for i in cells_data]
    # print('predict: ', pred)
    # print('truth: ', [i.phase for i in cells_data])
    return {'predict': pred, 'prob': prob, 'truth': truth}


def evaluate(true_labels, pred_labels, average='weighted'):
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average=average)
    recall = recall_score(true_labels, pred_labels, average=average)
    f1 = f1_score(true_labels, pred_labels, average=average)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print('---'*10)


def confusion_matrix_ratio_draw(truth_labels, predict_labels, filepath):
    cm = confusion_matrix(truth_labels, predict_labels)
    conf_mat_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    conf_mat_percent = np.round(conf_mat_percent, decimals=4)
    conf_matrix_ratio = pd.DataFrame(conf_mat_percent, index=['G1/G2', 'M', 'S'], columns=['G1/G2', 'M', 'S'])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set(yticklabels=[])  # remove the tick labels
    ax.set(xticklabels=[])  # remove the tick labels
    ax.tick_params(left=False)  # remove the ticks
    ax.tick_params(bottom=False)  # remove the ticks
    sns.heatmap(conf_matrix_ratio, annot=True, annot_kws={"size": 14}, cmap="Blues", fmt='g')
    plt.ylabel('Truth', fontsize=14)
    plt.xlabel('Prediction', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig(filepath)

def confusion_matrix_draw(truth_labels, predict_labels, filepath):
    cm = confusion_matrix(truth_labels, predict_labels)
    conf_matrix = pd.DataFrame(cm, index=['G1/G2', 'M', 'S'], columns=['G1/G2', 'M', 'S'])
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set(yticklabels=[])  # remove the tick labels
    ax.set(xticklabels=[])  # remove the tick labels
    ax.tick_params(left=False)  # remove the ticks
    ax.tick_params(bottom=False)  # remove the ticks
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 14}, cmap="Blues", fmt='g')
    plt.ylabel('Truth', fontsize=14)
    plt.xlabel('Prediction', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.savefig(r'H:\CCDeep-data\figure\confusion_matrix.pdf')
    plt.savefig(filepath)

    # plt.show()


def pr(true_label, predict_label):
    predict_label = label_binarize(predict_label, classes=['G1/G2', 'M', 'S'])
    true_label = label_binarize(true_label, classes=['G1/G2', 'M', 'S'])

    n_classes = predict_label.shape[1]
    lb = ['G1/G2', 'M', 'S']
    # 计算精度和召回率
    precision = dict()
    recall = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(true_label[:, i],
                                                            predict_label[:, i])

    # 绘制PR曲线
    plt.figure(figsize=(10, 8))
    colors = ['navy', 'turquoise', 'darkorange']
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='{}'.format(lb[i]))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve ')
    plt.legend(loc="lower left")
    plt.savefig(r'H:\CCDeep-data\figure\pr-curve.pdf')
    plt.show()

def evaluate_timelapse():
    # annotation = r'G:\20x_dataset\copy_of_xy_01\copy_of_1_xy01-sub-id-center.json'
    # mcy_img = r'G:\20x_dataset\copy_of_xy_01\raw\sub_raw\mcy\copy_of_1_xy01.tif'
    # dic_img = r'G:\20x_dataset\copy_of_xy_01\raw\sub_raw\dic\copy_of_1_xy01.tif'

    annotation = r'E:\paper\evaluate_data\src01\result-GT-modify.json'
    mcy_img = r'E:\paper\evaluate_data\src01\mcy.tif'
    dic_img = r'E:\paper\evaluate_data\src01\dic.tif'

    dic = tifffile.imread(dic_img)
    mcy = tifffile.imread(mcy_img)
    with open(annotation, encoding='utf-8') as f:
        ann = json.load(f)

    index = 0
    truth_labels = []
    predict_labels = []

    result_data = []

    for i in  ann:
        # print(ann[i])
        cells = get_cells(ann[i], dic[index], mcy[index])
        result = predict(cells)
        truth_labels.extend(result.get('truth'))
        predict_labels.extend( result.get('predict'))
        index += 1
        # evaluate(result.get('truth'), result.get('predict'))
        if index > 50:
            break

        for i in zip(result.get('truth'), result.get('predict'), result.get('prob')):
            r = [i[0], i[1], i[2][0], i[2][1], i[2][2]]
            result_data.append(r)
    # evaluate(truth_labels, predict_labels)

    # target_names = ['G1/G2', 'M', 'S']
    print(classification_report(truth_labels, predict_labels))

    # pr(truth_labels, predict_labels)
    # confusion_matrix_draw(truth_labels, predict_labels)


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
    return G, M, S


def get_test_data():
    """生成测试数据集"""
    # base_dir = r'G:\20x_dataset\copy_of_xy_01\development-dir\cell_lines'
    base_dir = r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\cell_lines'
    all_G = []
    all_M = []
    all_S = []
    for line in os.listdir(base_dir):
        g, m, s = group_cell_line(os.path.join(base_dir, line))
        all_G.extend(g)
        all_S.extend(s)
        all_M.extend(m)
        # break
    return all_G, all_M, all_S

def generate_data(path, phase):
    data = utils.Data()
    data.image_dic = cv2.resize(tifffile.imread(path[0]), (config.image_width, config.image_height)) / 255
    data.image_mcy = cv2.resize(tifffile.imread(path[1]), (config.image_width, config.image_height)) / 255
    data.phase = phase
    return data

def get_predict_data():
    g, m, s = get_test_data()
    g = g[: len(s) * 2]
    data_g_list = []
    data_m_list = []
    data_s_list = []
    for i in g:
        data_g_list.append(generate_data(i, 'G1/G2'))
    for j in m:
        data_m_list.append(generate_data(j, 'M'))
    for k in s:
        data_s_list.append(generate_data(k, 'S'))
    return data_g_list, data_m_list, data_s_list

evaluate_timelapse()

# data = get_predict_data()
#
# truth_labels = []
# predict_labels = []
#
# for i in data:
#     result = predict(i)
#     truth_labels.extend(result.get('truth'))
#     predict_labels.extend(result.get('predict'))
#
# confusion_matrix_draw(truth_labels, predict_labels, r'H:\CCDeep-data\figure\confusion_matrix\copy19-cm.pdf')
# confusion_matrix_ratio_draw(truth_labels, predict_labels, r'H:\CCDeep-data\figure\confusion_matrix\copy19-ratio-cm.pdf')
#
# with open(r'H:\CCDeep-data\figure\confusion_matrix\copy19_classification_report-dev-model.txt', 'w') as f:
#     print(classification_report(truth_labels, predict_labels),file=f)
#
# with open(r'H:\CCDeep-data\figure\confusion_matrix\copy19_truth_predict_label.txt', 'w') as f2:
#     print('truth labels:\n', truth_labels, '\npredict labels:\n', predict_labels, file=f2)
