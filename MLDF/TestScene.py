# -*- coding=utf-8 -*-
import csv
import numpy as np

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


# 加载数据和标签
def load_csv():
    """
    从CSV文件中读取数据信息
    :param csv_file_name: CSV文件名
    :return: Data：二维数组
    """
    data_csv = r'D:\Pycharm2020.1.3\WorkSpace\MLDF\dataset\scene_data.csv'
    label_csv = r'D:\Pycharm2020.1.3\WorkSpace\MLDF\dataset\scene_label.csv'

    with open(data_csv, encoding='utf-8') as f:
        data = np.loadtxt(f, str, delimiter=",")
    with open(label_csv, encoding='utf-8') as f:
        label = np.loadtxt(f, str, delimiter=",")

    # 将数据label强制转换为指定的类型，astype函数是在副本上进行，并非修改原数组。
    # 从文件中load出来的数据类型是“class 'numpy.int16'”类型，需要进行类型转化
    # data = data.astype("float")
    label = label.astype("int")

    # 取数据集的行数，即是实例数
    num_samples = len(data)

    # 用shuffle_index函数将502这个整数随机划分成两个长为251的list，list中的元素是502以内的整数
    # data是<class 'numpy.ndarray'>的二维矩阵，将上一步的list传入，会将data中按list中的元素按行取出
    # 这两步就是将（502，68）的data二维矩阵划分成了两个（251，68）的二维矩阵，分别代表训练集和测试集
    # 针对label这个（502, 174）的二维矩阵也是这么操作，而且采集时用的是同一组list，保证实例和标签对应
    train_index, test_index = shuffle_index(num_samples)
    """
    划分结果如下：
    train_data <class 'numpy.ndarray'> (1204, 294) 1204
    train_label <class 'numpy.ndarray'> (1204, 6) 1204
    test_data <class 'numpy.ndarray'> (1203, 294) 1203
    test_label <class 'numpy.ndarray'> (1203, 6) 1203
    """
    train_data = data[train_index]
    train_label = label[train_index]
    test_data = data[test_index]
    test_label = label[test_index]


    print("加载scene数据集完成！！!")

    # 返回值是训练数据、测试数据、标签数
    return [train_data, train_label, test_data, test_label]


if __name__ == '__main__':
    dataset = "scene"
    # 初始化数据集、测试数据集、标签集
    train_data, train_label, test_data, test_label = load_csv()

    # 构造森林，将4个森林级联，最大层数设为10，5折交叉验证
    model = Cascade(dataset, max_layer=20, num_forests=4, n_fold=6, step=3)

    # 训练森林，传入训练集、训练标签、指标名称、每个森林中的树的数量设为40
    model.train(train_data, train_label, "hamming loss", n_estimators=40)

    test_prob = model.predict(test_data, "hamming loss")

    value = do_metric(test_prob, test_label, 0.5)

    meatures = ["hamming loss", "one-error", "coverage", "ranking loss", "average precision",  "macro-auc"]
    res = zip(meatures, value)
    for item in res:
        print(item)

"""
参数配置：model = Cascade(dataset, max_layer=10, num_forests=2, n_fold=5, step=3)
('hamming loss', 0.08922139096702686)
('one-error', 0.2053200332502078)
('coverage', 0.07204211692989748)
('ranking loss', 0.06783042394014982)
('average precision', 0.8785882515932402)
('macro-auc', 0.9476817170863981)
"""
