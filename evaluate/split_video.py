import os
import shutil

# 定义文件夹路径和每组文件的数量
folder_path = r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\png'
group_size = 100

# 创建保存新文件夹的路径
new_folder_path = r'G:\20x_dataset\evaluate_data\split-copy19\png'
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

# 获取文件名列表，并按照文件名排序
file_names = os.listdir(folder_path)
file_names.sort()
#
# # 分组保存文件
# for i in range(0, len(file_names), group_size):
#     # 构造新文件夹名称
#     new_folder_name = 'group{:03d}'.format(i // group_size )
#     new_folder_path_i = os.path.join(new_folder_path, new_folder_name)
#     if not os.path.exists(new_folder_path_i):
#         os.makedirs(new_folder_path_i)
#
#     # 复制文件到新文件夹，并修改文件名
#     for j in range(i, min(i + group_size, len(file_names))):
#         file_path = os.path.join(folder_path, file_names[j])
#         file_prefix, file_suffix = os.path.splitext(file_names[j])
#         new_file_prefix = file_prefix.split('-')[0] + '-{:04d}'.format(j - i)
#         new_file_name = new_file_prefix + file_suffix
#         new_file_path = os.path.join(new_folder_path_i, new_file_name)
#         shutil.copy(file_path, new_file_path)

import json
with open(r'G:\20x_dataset\evaluate_data\copy_of_1_xy19\result-GT.json') as f:
    data = json.load(f)

new_group =  {}

keys = list(data.keys())
for i in range(0, len(data), group_size):
    new_group[i] = {}
    for j in range(i, min(i + group_size, len(data))):
        file_prefix, file_suffix = os.path.splitext(keys[j])
        new_file_prefix = file_prefix.split('-')[0] + '-{:04d}'.format(j - i)
        new_file_name = new_file_prefix + file_suffix
        print(keys[j])
        print(new_file_name)
        region = data[keys[j]]
        region['filename'] = new_file_name
        print(region)
        new_group[i][new_file_name] = region
    print('-----------------')
    # break
# print(new_group[0].keys())
for k in new_group:
    fname = fr'G:\20x_dataset\evaluate_data\split-copy19\group-{k:04d}.json'
    with open(fname, 'w') as f:
        json.dump(new_group[k], f)
