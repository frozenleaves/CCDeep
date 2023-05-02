# 评估模型分割性能
# 采用如下指标衡量分割性能

# Similarity:
# DICE    = 1.000000      Dice Coefficient (F1-Measure)
# JACRD   = 1.000000      Jaccard Coefficient (IoU)
# AUC     = 1.000000      Area under ROC Curve
# KAPPA   = 1.000000      Cohen Kappa
# RNDIND  = 1.000000      Rand Index
# ADJRIND = 1.000000      Adjusted Rand Index
# ICCORR  = 1.000000      Interclass Correlation
# VOLSMTY = 1.000000      Volumetric Similarity Coefficient
# MUTINF  = 0.183272      Mutual Information
#
# Distance:
# HDRFDST = 0.000000      Hausdorff Distance (in voxel)
# AVGDIST = 0.000000      Average Hausdorff Distance (in voxel)
# MAHLNBS = 0.000000      Mahanabolis Distance
# VARINFO = 0.000000      Variation of Information
# GCOERR  = 0.000000      Global Consistency Error
# PROBDST = 0.000000      Probabilistic Distance
#
# Classic Measures:
# SNSVTY  = 1.000000      Sensitivity (Recall, true positive rate)
# SPCFTY  = 1.000000      Specificity (true negative rate)
# PRCISON = 1.000000      Precision (Confidence)
# FMEASR  = 1.000000      F-Measure
# ACURCY  = 1.000000      Accuracy
# FALLOUT = 0.000000      Fallout (false positive rate)
# TP      = 116631        true positive (in voxel)
# FP      = 0     false positive (in voxel)
# TN      = 4077673       true negative (in voxel)
# FN      = 0     false negative (in voxel)
# REFVOL  = 116631        reference volume (in voxel)
# SEGVOL  = 116631        segmented volume (in voxel)





import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm
import cv2
import subprocess
import os
import pyximport

pyximport.install(language_level=3)
import calculate
import csv


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


def json2mask(JsonFilePath, img, mask):
    """
    :param JsonFilePath:  需要转化的json文件路径
    :param img: json文件指向的tif图片所在目录
    :param mask: 生成的掩膜文件保存目录
    :return: 如果转化成功，返回True
    """
    annotation = json.load(open(JsonFilePath, 'r'))
    for i in tqdm(annotation):
        filename = i.replace('.png', '.tif')
        # filename = i.replace('.tif', '.png')
        regions = annotation[i].get('regions')
        # image_path = os.path.join(img, filename)
        # image = cv2.imread(image_path, -1)  # image = skimage.io.imread(image_path)
        # print(image.shape)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # height, width = image.shape[:2]
        height, width = 2048, 2048
        mask_arr = np.zeros((height, width), dtype=np.uint8)
        for region in regions:
            if region['shape_attributes'].get('name') == 'ellipse':
                rx = region['shape_attributes'].get('rx')
                ry = region['shape_attributes'].get('ry')
                cx = region['shape_attributes'].get('cx')
                cy = region['shape_attributes'].get('cy')
                theta = region['shape_attributes'].get('theta')
                all_x, all_y = ellipse_points((cx, cy), rx, ry, num_points=32, theta=theta)
                polygons = {'all_points_x': all_x, 'all_points_y': all_y}
            else:
                polygons = region.get('shape_attributes')
            points = []
            for j in range(len(polygons['all_points_x'])):
                x = int(polygons['all_points_x'][j])
                y = int(polygons['all_points_y'][j])
                points.append((x, y))
            contours = np.array(points)
            cv2.fillConvexPoly(mask_arr, contours, (255, 255, 255))
        save_path = os.path.join(mask, filename)
        cv2.imwrite(save_path, mask_arr)


def c_calc_single_frame(SEG_mask, GT_mask):
    start = time.time()
    img_GT = cv2.imread(GT_mask, -1)
    img_R = cv2.imread(SEG_mask, -1)
    ret_GT, binary_GT = cv2.threshold(img_GT, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret_R, binary_R = cv2.threshold(img_R, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    DICE = calculate.calDSI(binary_GT, binary_R)
    VOE = calculate.calVOE(binary_GT, binary_R)
    RVD = calculate.calRVD(binary_GT, binary_R)
    Precision = calculate.calPrecision(binary_GT, binary_R)
    Recall = calculate.calRecall(binary_GT, binary_R)
    end = time.time()
    print(f'cost time: {end - start:.2f}')
    # step 4：计算DSI
    print('（1）DICE计算结果，      DSI       = {0:.4}'.format(DICE))  # 保留四位有效数字

    # step 5：计算VOE
    print('（2）VOE计算结果，       VOE       = {0:.4}'.format(VOE))

    # step 6：计算RVD
    print('（3）RVD计算结果，       RVD       = {0:.4}'.format(RVD))

    # step 7：计算Precision
    print('（4）Precision计算结果， Precision = {0:.4}'.format(Precision))

    # step 8：计算Recall
    print('（5）Recall计算结果，    Recall    = {0:.4}'.format(Recall))


def Exec_EvaluateSegmentation(SEG_mask, GT_mask):
    # main = fr"H:\software\EvaluateSegmentation-2017.04.25-win-xp-and-later\EvaluateSegmentation.exe"
    cmd = fr'I:\CCDeep\CCDeep-data\software\EvaluateSegmentation-2017.04.25-win-xp-and-later\EvaluateSegmentation.exe {GT_mask} {SEG_mask}'
    print(GT_mask)
    print(SEG_mask)
    out = subprocess.getoutput(cmd)
    split_out = out.split(sep='\n\n')
    Similarity = split_out[0]
    Distance = split_out[1]
    Classic_Measures = split_out[2]
    sim_dict = {}
    dis_dict = {}
    cm_dict = {}
    for j in Similarity.split(sep='\n')[1:]:
        tmp = j.split()
        sim_dict[tmp[0]] = tmp[2]
    for k in Distance.split(sep='\n')[1:]:
        tmp = k.split()
        dis_dict[tmp[0]] = tmp[2]
    for i in Classic_Measures.split(sep='\n')[1:]:
        tmp = i.split()
        cm_dict[tmp[0]] = tmp[2]
    return sim_dict, dis_dict, cm_dict


def save(result: list, filename: str):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        header = list(result[0].keys())
        writer.writerow(header)
        for i in result:
            tmp = [i.get(key) for key in header]
            writer.writerow(tmp)


def calc(SEG_mask_dir, GT_mask_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    GT_masks = os.listdir(GT_mask_dir)
    SEG_masks = os.listdir(SEG_mask_dir)
    Similarity = []
    Distance = []
    Classic_Measures = []

    # thread_pool_executor = ThreadPoolExecutor(max_workers=50, thread_name_prefix="evaluate_")
    for i in tqdm(zip(GT_masks, SEG_masks), total=len(GT_masks)):
        GT_mask_path = os.path.join(GT_mask_dir, i[0])
        SEG_mask_path = os.path.join(SEG_mask_dir, i[1])
        sim_dict, dis_dict, cm_dict = Exec_EvaluateSegmentation(SEG_mask_path, GT_mask_path)
        Similarity.append(sim_dict)
        Distance.append(dis_dict)
        Classic_Measures.append(cm_dict)
        # break
    fname_sim = os.path.join(out_dir, 'Similarity.csv')
    fname_dis = os.path.join(out_dir, 'Distance.csv')
    fname_cm = os.path.join(out_dir, 'Classic_Measures.csv')
    save(Similarity, fname_sim)
    save(Distance, fname_dis)
    save(Classic_Measures, fname_cm)


def get_SEG_GT_mask():
    json_file = r'I:\paper\evaluate_data\evaluation_for_segmentation\evaluate.json'
    tif_dir = r'I:\paper\evaluate_data\evaluation_for_segmentation\image'
    out = r'I:\paper\evaluate_data\evaluation_for_segmentation\mask-CCDeep'
    json2mask(json_file, tif_dir, out)


if __name__ == '__main__':
    # get_SEG_GT_mask()

    # GT = r'F:\wangjiaqi\src\s6\mask\GT\mcy-0007.tif'
    # SEG = r'F:\wangjiaqi\src\s6\mask\SEG\mcy-0007.tif'
    # Exec_EvaluateSegmentation(SEG, GT)
    #
    # GT_dir = r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\mask\GT'
    # SEG_dir = r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\mask\deepcell'
    # out_dir = r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\measure\deepcell_segmentation'
    # calc(SEG_dir, GT_dir, out_dir)

    GT_dir = r'I:\paper\evaluate_data\evaluation_for_segmentation\mask-GT'
    SEG_dir = r'I:\paper\evaluate_data\evaluation_for_segmentation\mask-cellpose'
    out_dir = r'I:\paper\evaluate_data\evaluation_for_segmentation\measure\cellpose_segmentation'
    calc(SEG_dir, GT_dir, out_dir)

    # SEG_CCDeep_dir = r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\mask\CCDeep'
    # out_dir = r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\measure\CCDeep_segmentation'
    # calc(SEG_CCDeep_dir, GT_dir, out_dir)
    # i = ''
    # json2mask(rf'H:\CCDeep-data\raw-data\train\raw\dataset{i}\dataset{i}.json',
    #           rf'H:\CCDeep-data\raw-data\train\raw\dataset{i}\tif\mcy',
    #           rf'H:\CCDeep-data\raw-data\train\raw\dataset{i}\tif\masks')
