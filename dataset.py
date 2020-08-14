import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


class DataSet(object):
    def __init__(self, images, labels, img_names, cls):
        self.num_examples = images.shape[0]
        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        """ 返回下一个batch_size大小的数据集"""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_examples:
            """已经完成一个epoch，开始一个新的epoch"""
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self.num_examples  # 断言 如果条件错误，终止程序
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]


def read_train_sets(train_path, img_size, classes, test_size):
    class DataSets(object):
        pass
    data_sets = DataSets()
    count = 0
    images = []
    labels = []
    img_names = []
    cls = []
    print('Going to read training images')
    path = os.path.join(train_path, '*g')
    files = glob.glob(path)
    for file in files:
        image = cv2.imread(file)
        image = cv2.resize(image, (img_size, img_size), 0, 0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        images.append(image)
        label = np.zeros(2)
        if count < 12500:
            label[0] = 1.0
            cls.append(classes[0])
        else:
            label[1] = 1.0
            cls.append(classes[1])
        labels.append(label)
        file_base = os.path.basename(file)
        img_names.append(file_base)
        count = count + 1
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)
    # print(images, labels, img_names, cls)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  # 打乱图像的顺序，利于后面划分数据集
    if isinstance(test_size, float):
        test_size = int(test_size * images.shape[0])  # 计算测试集样本数量 数据集划分为8：2

    test_images = images[:test_size]
    test_labels = labels[:test_size]
    test_img_names = img_names[:test_size]
    test_cls = cls[:test_size]

    train_images = images[test_size:]
    train_labels = labels[test_size:]
    train_img_names = img_names[test_size:]
    train_cls = cls[test_size:]
    data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.test = DataSet(test_images, test_labels, test_img_names, test_cls)
    return data_sets

# 验证数据集是否读取 分别计算并输出训练集和测试集的数量

# train_path = 'data/train'
# classes = ['cat', 'dog']
# test_size = 0.2
# data= read_train_sets(train_path, 64, classes, test_size)
# print("数据读取完毕")
# print("训练集的数量为：{}".format(len(data.train._images)))
# print("测试集的数量为：{}".format(len(data.test._images)))