import cv2
import numpy as np
import skimage
from skimage import io
import tensorflow as tf
import pathlib

if __name__ != '__main__':
    from . import config
else:
    import config
    from tools.augmentation import augment_in_train


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
    """read image data from path dataset, this function's return will be organized for real training dataset."""
    mcy_path = img_path.numpy().decode()
    dic_path = mcy_path.replace('mcy', 'dic')
    dic_img = skimage.io.imread(dic_path)
    mcy_img = skimage.io.imread(mcy_path)
    img = np.dstack([cv2.resize(dic_img, (config.image_width, config.image_height)),
                     cv2.resize(mcy_img, (config.image_width, config.image_height))])
    if config.AUGMENTATION_IN_TRAINING:
        img = tf.convert_to_tensor(img, dtype=tf.uint16)
    else:
        img = tf.convert_to_tensor(img / 255, dtype=tf.float64)
    return img, img_label.numpy()


def read_img_dev(img_path, img_label):
    """Develop features, embed data augmentation, based on the image path and label dataset,
       do the augmentation for each batch during read image, and then return the augmented dataset"""
    mcy_path = img_path.numpy().decode()
    dic_path = mcy_path.replace('mcy', 'dic')
    dic_img = skimage.io.imread(dic_path)
    mcy_img = skimage.io.imread(mcy_path)
    aug_dic = augment_in_train(dic_img, img_label.numpy())
    aug_mcy = augment_in_train(mcy_img, img_label.numpy())
    images = []
    for i in range(len(aug_dic)):
        img_stack = np.dstack([cv2.resize(aug_dic[i], (config.image_width, config.image_height)),
                               cv2.resize(aug_mcy[i], (config.image_width, config.image_height))])
        images.append(img_stack / 255)

    images = np.array(images)
    images = tf.convert_to_tensor(images, dtype=tf.float64)
    labels = tf.repeat(img_label, images.shape[0]).numpy()

    img = np.dstack([cv2.resize(dic_img, (config.image_width, config.image_height)),
                     cv2.resize(mcy_img, (config.image_width, config.image_height))])
    img = tf.convert_to_tensor(img / 255, dtype=tf.float64)
    return img, img_label.numpy()
    # return images, labels


def wrap_function(mcy, y):
    """wrap function for tensor and numpy transform """
    if config.AUGMENTATION_IN_TRAINING:
        x_img, y_label = tf.py_function(read_img, inp=[mcy, y], Tout=[tf.uint16, tf.int32])
    else:
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
    test, count = get_dataset(r'G:\20x_dataset\train_dataset-dev1.3.2\train_mcy\t')
    t = test.shuffle(buffer_size=count).batch(batch_size=4)
    for i in t:
        print(i[0].shape)
        train_datas = []
        train_labels = []
        for j in range(i[1].shape[0]):
            aug_images_dic = augment_in_train(i[0][j][:, :, 0].numpy(), i[1][j].numpy())
            aug_images_mcy = augment_in_train(i[0][j][:, :, 1].numpy(), i[1][j].numpy())
            aug_labels = tf.repeat(i[1][j], len(aug_images_dic))
            for k in range(len(aug_images_dic)):
                img = np.dstack([cv2.resize(aug_images_dic[k], (config.image_width, config.image_height))/255,
                                 cv2.resize(aug_images_mcy[k], (config.image_width, config.image_height))/255])
                train_datas.append(img)
            train_labels.append(aug_labels)
        print(np.array(train_datas).shape)
        print(i[1])
        print(tf.concat(train_labels, axis=0))

        print(tf.convert_to_tensor(np.array(train_datas), dtype=tf.float64)[0])
        break


