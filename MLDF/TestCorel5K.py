# -*- coding=utf-8 -*-
import csv
import numpy as np
import pandas as pd

from sklearn.utils import shuffle

from learner.cascade import Cascade
from learner.measure import *


# 随机排列实例数，将实例划分为训练集和测试集
def shuffle_index(num_samples):
    # a = range(0, 502),502是实例数
    a = range(0, num_samples)

    # 利用shuffle函数将序列a中的元素重新随机排列
    a = shuffle(a)

    # 去实例数的一半，上取整
    length = int((num_samples + 1) / 2)
    # 上半做训练集
    train_index = a[:length]
    # 下半做测试集
    test_index = a[length:]
    return [train_index, test_index]


def load_csv(dataset_name, label_name):
    data_split = 0.5
    # 数据集目录和文件
    data_dir = 'D:/Pycharm2020.1.3/WorkSpace/HMLDSC/ML_Dataset'
    # low_memory：分块加载到内存，再低内存消耗中解析。但是可能出现类型混淆。确保类型不被混淆需要设置为False。
    data = pd.read_csv(data_dir + dataset_name, low_memory=False)
    label = pd.read_csv(data_dir + label_name, low_memory=False)
    n_samples = data.shape[0]
    n_training = int(data_split * n_samples)
    # 训练集
    train = data[:n_training]  # type(train) = <class 'pandas.core.frame.DataFrame'>
    train = np.array(train)
    train_label = label[:n_training]
    train_label = np.array(train_label)  # 只有np类型才能每次取一列，list类型不行。
    # 测试集
    test = data[n_training:]
    test = np.array(test)
    test_label = label[n_training:]
    test_label = np.array(test_label)
    if train is not None:
        print(dataset_name[1:-9], "数据集预处理完成！！")
    print("train_data、train_label shape：", train.shape, train_label.shape)
    print("test_data、test_label shape：", test.shape, test_label.shape)
    print("\n+++++++++++++++++开始训练++++++++++++++++")

    # 获取当前数据集的特征数、标签数
    return [train, train_label, test, test_label]


if __name__ == '__main__':
    dataset_name = '/Corel5k_data.csv'
    label_name = '/Corel5k_label.csv'
    # 初始化数据集、测试数据集、标签集
    train_data, train_label, test_data, test_label = load_csv(dataset_name, label_name)

    # 构造森林，将另个森林级联，最大层数设为10，5折交叉验证，step用来计算每一层中森林的最大深度（按层递增）【max_depth = step * layer_index + step】
    model = Cascade(dataset_name, max_layer=20, num_forests=4, n_fold=3, step=3)

    # 训练森林，传入训练集、训练标签、指标名称、每个森林中的树的数量设为40
    model.train(train_data, train_label, "hamming loss", n_estimators=40)

    test_prob = model.predict(test_data, "hamming loss")

    value = do_metric(test_prob, test_label, 0.5)

    meatures = ["hamming loss", "one-error", "coverage", "ranking loss", "average precision", "macro-auc"]
    res = zip(meatures, value)
    for item in res:
        print(item)
