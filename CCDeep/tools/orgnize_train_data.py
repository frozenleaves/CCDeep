import os
import shutil


from CCDeep.tools import augmentation, generate_dataset
from CCDeep.tools import split_dataset as split_dataset
from CCDeep import config

# train_classification_mcy_data_save_path = config.dataset_dir_mcy_20x
# train_classification_dic_data_save_path = config.dataset_dir_dic_20x

# train_classification_mcy_data_save_path = r'H:\CCDeep-data\raw-data\train\classification'
train_classification_mcy_data_save_path = r'H:\CCDeep-data\raw-data\train\debug_dataset'


def copy_to_tmp(__src_mcy, tmp, flag:['G', 'S', 'M']):
    base_dir = os.path.join(__src_mcy, flag)
    for i in os.listdir(base_dir):
        src_mcy = os.path.join(base_dir , i)
        src_dic = src_mcy.replace('mcy', 'dic')
        dst_mcy = tmp + f'\\mcy\\{flag}'
        dst_dic = tmp + f'\\dic\\{flag}'
        # print(dst_dic_g)
        # print(f'copy {src_dic} to {dst_dic}')
        # print(f'copy {src_mcy} to {dst_mcy}')
        shutil.copy(src_dic, dst_dic)
        shutil.copy(src_mcy, dst_mcy)


def generate_classification_dataset():
    root_data_dir = config.raw
    tmp = os.path.join(root_data_dir, 'tmp\\train_classification_dataset')
    generate_dataset.generate_data(root_data_dir)
    #
    if not os.path.exists(tmp):
        os.makedirs(tmp + '\\mcy\\G')
        os.makedirs(tmp + '\\mcy\\M')
        os.makedirs(tmp + '\\mcy\\S')
        os.makedirs(tmp + '\\dic\\G')
        os.makedirs(tmp + '\\dic\\M')
        os.makedirs(tmp + '\\dic\\S')
    for dataset in os.listdir(root_data_dir):
        if 'tmp' in dataset:
            continue
        dataset_path = os.path.join(root_data_dir, dataset)
        print('dataset_path: ', dataset_path)
        un_aug_dic = os.path.join(dataset_path, 'train_classification_dataset\\dic')
        un_aug_mcy = os.path.join(dataset_path, 'train_classification_dataset\\mcy')
        src_dic = os.path.join(dataset_path, 'train_classification_dataset\\aug_dic')
        src_mcy = os.path.join(dataset_path, 'train_classification_dataset\\aug_mcy')
        augmentation.augment(un_aug_dic, un_aug_mcy)
        if (os.path.exists(src_dic) and os.path.exists(src_mcy)):
            copy_to_tmp(src_mcy, tmp, 'S')
            copy_to_tmp(src_mcy, tmp, 'G')
            copy_to_tmp(src_mcy, tmp, 'M')
            # copy_to_tmp(un_aug_mcy, tmp, 'S')
            # copy_to_tmp(un_aug_mcy, tmp, 'G')
            # copy_to_tmp(un_aug_mcy, tmp, 'M')
        # break

    split = split_dataset.SplitDataset(dataset_dir=os.path.join(tmp,'mcy'),
                                 saved_dataset_dir=os.path.join(train_classification_mcy_data_save_path, 'mcy'),
                                 show_progress=True)
    split.start_splitting()

    shutil.rmtree(tmp)



if __name__ == '__main__':
    generate_classification_dataset()
