import os
import random
import shutil
from tqdm import tqdm


class SplitDataset(object):
    def __init__(self, dataset_dir, saved_dataset_dir, train_ratio=0.6, test_ratio=0.1, show_progress=False):
        self.dataset_dir = dataset_dir
        self.saved_dataset_dir = saved_dataset_dir
        self.saved_train_dir = os.path.join(saved_dataset_dir, "train")
        self.saved_valid_dir = os.path.join(saved_dataset_dir, "valid")
        self.saved_test_dir = os.path.join(saved_dataset_dir, "test")

        self.train_ratio = train_ratio
        self.test_radio = test_ratio
        self.valid_ratio = 1 - train_ratio - test_ratio

        self.train_file_path = []
        self.valid_file_path = []
        self.test_file_path = []

        self.index_label_dict = {}

        self.show_progress = show_progress
        if not os.path.exists(self.saved_dataset_dir):
            os.mkdir(self.saved_dataset_dir)
        if not os.path.exists(self.saved_train_dir):
            os.mkdir(self.saved_train_dir)
        if not os.path.exists(self.saved_test_dir):
            os.mkdir(self.saved_test_dir)
        if not os.path.exists(self.saved_valid_dir):
            os.mkdir(self.saved_valid_dir)

    def __get_label_names(self):
        label_names = []
        for item in os.listdir(self.dataset_dir):
            item_path = os.path.join(self.dataset_dir, item)
            if os.path.isdir(item_path):
                label_names.append(item)
        return label_names

    def __get_all_file_path(self):
        all_file_path = []
        index = 0
        for file_type in self.__get_label_names():
            self.index_label_dict[index] = file_type
            index += 1
            type_file_path = os.path.join(self.dataset_dir, file_type)
            file_path = []
            for file in os.listdir(type_file_path):
                single_file_path = os.path.join(type_file_path, file)
                file_path.append(single_file_path)
            all_file_path.append(file_path)
        return all_file_path

    def __copy_files(self, type_path, type_saved_dir):
        for item in type_path:
            src_path_list = item[1]
            dst_path = os.path.join(type_saved_dir, item[0])
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            if not os.path.exists(dst_path.replace('mcy', 'dic')):
                os.makedirs(dst_path.replace('mcy', 'dic'))
            for src_path in tqdm(src_path_list):
                shutil.copy(src_path, dst_path)
                shutil.copy(src_path.replace('mcy', 'dic'), dst_path.replace('mcy', 'dic'))
                if self.show_progress:
                    print("Copying file " + src_path + " to " + dst_path)
                    print("Copying file " + src_path.replace('mcy', 'dic') + " to " + dst_path.replace('mcy', 'dic'))

    def __split_dataset(self):
        all_file_paths = self.__get_all_file_path()
        for index in range(len(all_file_paths)):
            file_path_list = all_file_paths[index]
            file_path_list_length = len(file_path_list)
            random.shuffle(file_path_list)

            train_num = int(file_path_list_length * self.train_ratio)
            test_num = int(file_path_list_length * self.test_radio)

            self.train_file_path.append([self.index_label_dict[index], file_path_list[: train_num]])
            self.test_file_path.append([self.index_label_dict[index], file_path_list[train_num:train_num + test_num]])
            self.valid_file_path.append([self.index_label_dict[index], file_path_list[train_num + test_num:]])

    def start_splitting(self):
        self.__split_dataset()
        self.__copy_files(type_path=self.train_file_path, type_saved_dir=self.saved_train_dir)
        self.__copy_files(type_path=self.train_file_path, type_saved_dir=self.saved_train_dir)

        self.__copy_files(type_path=self.valid_file_path, type_saved_dir=self.saved_valid_dir)
        self.__copy_files(type_path=self.test_file_path, type_saved_dir=self.saved_test_dir)


def copy_file(dataset_type, src_mcy_dir):
    # name_src = '/home/zje/CellClassify/train_dataset/train_data_20x_new2/train_mcy'
    name_src = src_mcy_dir
    train = os.path.join(name_src, 'train')
    valid = os.path.join(name_src, 'valid')
    test = os.path.join(name_src, 'test')
    if dataset_type == 'train':
        folder = train
        ss = 'train/'
    elif dataset_type == 'test':
        folder = test
        ss = 'test/'
    elif dataset_type == 'valid':
        folder = valid
        ss = 'valid/'
    else:
        raise ValueError("no support dataset type")
    for fd in os.listdir(folder):
        f = os.path.join(folder, fd)
        files = os.listdir(f)
        file_src = [os.path.join(f.replace('train_mcy', 'dic'), x) for x in files]
        src = [s.replace(ss, '') for s in file_src]

        file_dst = [os.path.join(f.replace('train_mcy', 'train_dic'), y) for y in files]
        for z in zip(src, file_dst):
            if not os.path.exists(os.path.dirname(z[1])):
                os.makedirs(os.path.dirname(z[1]))
            shutil.copy(z[0], z[1])
            print(f'copy f{z[0]} to f{z[1]}')


if __name__ == '__main__':
    # split_dataset = SplitDataset(dataset_dir="original_dataset",
    #                              saved_dataset_dir="dataset",
    #                              show_progress=True)

    split_dataset = SplitDataset(dataset_dir=r'G:\20x_dataset\train_dataset-dev1.3.2\augment_mcy',
                                 saved_dataset_dir=r'G:\20x_dataset\train_dataset-dev1.3.2\train_mcy',
                                 show_progress=False)
    split_dataset.start_splitting()
    # copy_file(dataset_type='valid', src_mcy_dir='/home/zje/CellClassify/train_dataset/20x_exp_bbox/20x_augment/train_mcy')
