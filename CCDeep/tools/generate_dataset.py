from __future__ import annotations

import glob
from CCDeep.utils import *

root_dir = r'G:\20x_dataset'


def get_path():
    dataset_dir = []
    for i in os.listdir(root_dir):
        path = os.path.join(root_dir, i)
        if os.path.isdir(path) and (i.startswith('dataset') or i.startswith('copy_of')):
            dataset_dir.append(path)
    return dataset_dir


def generate(dataset_dir, cellfilter: int = 10):
    dic_dir = os.path.join(dataset_dir, 'tif\\dic')
    mcy_dir = os.path.join(dataset_dir, 'tif\\mcy')
    json_path = glob.glob(dataset_dir + '\\*.json')[0]
    assert len(os.listdir(dic_dir)) == len(os.listdir(mcy_dir))
    save_dir = os.path.join(dataset_dir, 'train_data_dev1.3.2')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir_mcy = os.path.join(save_dir, 'mcy')
    save_dir_dic = os.path.join(save_dir, 'dic')

    generator = DataGenerator(training_data_mcy=mcy_dir, train_data_dic=dic_dir, training_label=json_path)
    for data in generator.generate():
        if max(data.image_mcy.shape) < cellfilter:
            print(data.image_mcy.shape)
            continue
        else:
            if data.phase == 0:
                phase = 'G'
            elif data.phase == 1:
                phase = 'M'
            elif data.phase == 2:
                phase = 'S'
            else:
                continue
            dic = os.path.join(save_dir_dic, phase + '\\' + data.image_id + '.tif')
            mcy = os.path.join(save_dir_mcy, phase + '\\' + data.image_id + '.tif')
            if not os.path.exists(os.path.dirname(dic)):
                os.makedirs(os.path.dirname(dic))
            if not os.path.exists(os.path.dirname(mcy)):
                os.makedirs(os.path.dirname(mcy))
            if 0 in data.image_mcy.shape:
                print('filter')
                continue
            cv2.imwrite(dic, data.image_dic)
            cv2.imwrite(mcy, data.image_mcy)
            # plt.imshow(data.image_mcy, cmap='gray')
            # plt.show()
            # plt.imshow(data.image_dic, cmap='gray')
            # plt.show()
    print(f'{os.path.basename(dataset_dir)} ok')


def get_path_60x():
    root_60x = r'E:\60x_dataset'
    paths = []
    labels = []
    for i in glob.glob(root_60x + '\\*.json'):
        labels.append(i)
    img_dir = [x.replace('.json', '') for x in labels]
    dic_dir = [os.path.join(x, 'dic') for x in img_dir]
    mcy_dir = [os.path.join(x, 'mcy') for x in img_dir]
    for n in range(len(labels)):
        # yield labels[n], dic_dir[n], mcy_dir[n]
        paths.append((labels[n], dic_dir[n], mcy_dir[n]))
    return paths


# def generate_60x(path, cellfilter=10):
#     label = path[0]
#     dic = path[1]
#     mcy = path[2]
#     save_root = r'E:\60x_train_data-dev1.3.1'
#     dataset_name = os.path.join(save_root, os.path.basename(os.path.dirname(dic)))
#     dataset_dic = os.path.join(dataset_name, 'dic')
#     dataset_mcy = os.path.join(dataset_name, 'mcy')
#     generator = DataGenerator(training_data_mcy=mcy, train_data_dic=dic, training_label=label)
#     for data in generator.generate():
#         if max(data.image_mcy.shape) < cellfilter:
#             # print(data.image_mcy.shape)
#             continue
#         else:
#             if data.phase == 0:
#                 phase = 'G'
#             elif data.phase == 1:
#                 phase = 'M'
#             elif data.phase == 2:
#                 phase = 'S'
#             else:
#                 continue
#             dic = os.path.join(dataset_dic, phase + '\\' + data.image_id + '.tif')
#             mcy = os.path.join(dataset_mcy, phase + '\\' + data.image_id + '.tif')
#             if not os.path.exists(os.path.dirname(dic)):
#                 os.makedirs(os.path.dirname(dic))
#             if not os.path.exists(os.path.dirname(mcy)):
#                 os.makedirs(os.path.dirname(mcy))
#             if 0 in data.image_mcy.shape:
#                 print('filter')
#                 continue
#             cv2.imwrite(dic, data.image_dic)
#             cv2.imwrite(mcy, data.image_mcy)
#     print(f'{os.path.basename(os.path.dirname(dic))} ok')


if __name__ == '__main__':
    # for p in get_path_60x()[4:]:
    #     generate_60x(p)
    for p in get_path():
        print(p)
        generate(p)
        break
    # generate(r'E:\20x_dataset\test')
