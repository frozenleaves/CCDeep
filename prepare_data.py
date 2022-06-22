import cv2
import numpy as np
import skimage
from skimage import io
import tensorflow as tf
import pathlib
import config


def get_images_and_labels(data_root_dir):
    # get all images' paths (format: string)
    data_root = pathlib.Path(data_root_dir)
    all_image_path = [str(path) for path in list(data_root.glob('*/*'))]
    label_names = sorted(item.name for item in data_root.glob('*/'))
    label_to_index = dict((label, index) for index, label in enumerate(label_names))
    all_image_label = [label_to_index[pathlib.Path(single_image_path).parent.name] for single_image_path in
                       all_image_path]
    return all_image_path, all_image_label


def read_img(img_path, img_label):
    mcy_path = img_path.numpy().decode()
    dic_path = mcy_path.replace('mcy', 'dic')
    dic_img = skimage.io.imread(dic_path)
    mcy_img = skimage.io.imread(mcy_path)
    img = np.dstack([cv2.resize(dic_img, (config.image_width, config.image_height)),
                     cv2.resize(mcy_img, (config.image_width, config.image_height))])
    img = tf.convert_to_tensor(img/255, dtype=tf.float64)
    return img, img_label.numpy()


def wrap_function(mcy, y):
    x_img, y_label = tf.py_function(read_img, inp=[mcy, y], Tout=[tf.float64, tf.int32])
    return x_img, y_label


def get_dataset(dataset_root_dir_mcy):
    mcy_img_path, img_label = get_images_and_labels(dataset_root_dir_mcy)
    X_mcy = np.array(mcy_img_path)
    y = np.array(img_label)
    dataset = tf.data.Dataset.from_tensor_slices((X_mcy, tf.cast(y, tf.int32))).map(wrap_function)
    count = len(dataset)
    return dataset, count


def generate_datasets(train_mcy, valid_mcy, test_mcy):
    train_dataset, train_count = get_dataset(dataset_root_dir_mcy=train_mcy)
    valid_dataset, valid_count = get_dataset(dataset_root_dir_mcy=valid_mcy)
    test_dataset, test_count = get_dataset(dataset_root_dir_mcy=test_mcy)
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
    test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE)
    return train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count


def generate_datasets_20x():
    return generate_datasets(train_mcy=config.train_dir_mcy_20x,
                             valid_mcy=config.valid_dir_mcy_20x,
                             test_mcy=config.test_dir_mcy_20x)


def generate_datasets_60x():
    return generate_datasets(train_mcy=config.train_dir_mcy_60x,
                             valid_mcy=config.valid_dir_mcy_60x,
                             test_mcy=config.test_dir_mcy_60x)


if __name__ == '__main__':
    pass
