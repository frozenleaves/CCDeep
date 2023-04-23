import os.path

import tifffile
import numpy as np

import json
import os
import shutil


def sample_frames(json_file, output_dir, interval):
    # 打开原始 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建新的字典来保存间隔采样帧的信息
    sampled_frames = {}

    # 遍历所有帧，根据所需的间隔采样帧
    index = 0
    for filename, frame_data in data.items():
        if index % interval == 0:
            # 将文件名从 mcy-xxxx.png 更改为 0001, 0002, 0003 等
            print(filename)
            new_filename = filename[:-8] + "{:04d}.png".format(index // interval)
            print(new_filename)
            frame_data['filename'] = new_filename
            # 将帧信息添加到新的字典中
            sampled_frames[new_filename] = frame_data
        index += 1

    # 将新的字典保存到新的 JSON 文件中
    output_file = os.path.join(output_dir, str(interval * 5) + '-' + os.path.basename(json_file))
    with open(output_file, 'w') as f:
        json.dump(sampled_frames, f, indent=4)


def sample_files(input_dir, output_dir, interval):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历所有 PNG 文件，根据所需的间隔采样文件
    index = 0
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.png'):
            if index % interval == 0:
                # 构建新的文件名，如 "0000.png", "0001.png" 等
                new_filename = filename[:-8] + "{:04d}.png".format(index // interval)
                # 拷贝文件到新的文件夹中，并重新命名
                src_path = os.path.join(input_dir, filename)
                dst_path = os.path.join(output_dir, new_filename)
                shutil.copyfile(src_path, dst_path)
            index += 1


# 将间隔设置为 2，每两帧采样一次
interval = 2
# 原始 JSON 文件路径
json_file = r"E:\paper\evaluate_data\copy_of_1_xy01\result-GT.json"
# 间隔采样后的 JSON 文件
sampled_json_file = r"E:\paper\evaluate_data\copy_of_1_xy01\json_file.json"


# 执行间隔采样操作
# sample_frames(json_file, interval)
# input_dir = r"E:\paper\evaluate_data\copy_of_1_xy01\png"
# 间隔采样后的文件夹路径
# output_dir = r"E:\paper\evaluate_data\10min\copy_of_1_xy01_10min\png"

# 执行间隔采样操作
# sample_files(input_dir, output_dir, interval)

# print("间隔采样完成！请检查文件夹：", output_dir)
# print("间隔采样完成！请检查文件：", sampled_json_file)


def select_frame_for_annotation(gap):
    base_save_dir = rf'E:\paper\evaluate_data\{gap*5}min'
    # dirs = ['copy_of_1_xy01', 'copy_of_1_xy19', 'MCF10A_copy02', 'MCF10A_copy11', 'split-copy19', 'src01', 'src06']
    dirs = ['copy_of_1_xy10']
    base_file_dir = r'E:\paper\evaluate_data'
    for i in dirs:
        file = os.path.join(base_file_dir, i) + '\\result-GT.json'
        pngs = os.path.join(base_file_dir, i) + '\\png'
        out_dir = os.path.join(base_save_dir, i + f'_{5*gap}min')
        if os.path.exists(file):
            sample_frames(file, out_dir, gap)
            sample_files(pngs, out_dir + '\\png', gap)


gap_map = {
    5: 1,
    10: 2,
    15: 3,
    20: 4,
    25: 5,
    30: 6
}


def select(dic_tiff, mcy_tiff, gap=10):
    dic_tmp = []
    mcy_tmp = []
    step = gap_map[gap]
    print(dic_tiff.shape)
    for i in range(0, len(dic_tiff), step):
        dic_tmp.append(dic_tiff[i])
        mcy_tmp.append(mcy_tiff[i])

    return np.array(dic_tmp), np.array(mcy_tmp)


def save(dic_tif, mcy_tif, gap=10):
    save_base_path = rf'J:\paper\evaluate_data\src01_{gap}min'
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)
    dic, mcy = select(dic_tif, mcy_tif, gap)
    tifffile.imwrite(save_base_path + '\\dic.tif', dic)
    tifffile.imwrite(save_base_path + '\\mcy.tif', mcy)






if __name__ == '__main__':
    for i in range(2, 7):
        print(i)
        select_frame_for_annotation(i)
    dic_tiff = r'J:\paper\evaluate_data\src01\dic.tif'
    mcy_tiff = r'J:\paper\evaluate_data\src01\mcy.tif'

    dic = tifffile.imread(dic_tiff)
    mcy = tifffile.imread(mcy_tiff)

    save(dic, mcy, gap=10)
    save(dic, mcy, gap=15)
    save(dic, mcy, gap=20)
    save(dic, mcy, gap=25)
    save(dic, mcy, gap=30)
