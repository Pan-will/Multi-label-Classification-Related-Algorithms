from sklearn.utils import shuffle

from learner.cascade import Cascade
from learner.measure import *


# 随机排列实例数，将实例划分为训练集和测试集
def shuffle_index(num_samples):
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
# image数据集一共2000条数据，每条数据5个标签
# data矩阵shape是2000，294；label矩阵shape是2000，5
def make_data(dataset):
    data = np.load("dataset/{}_data.npy".format(dataset))
    label = np.load("dataset/{}_label.npy".format(dataset))

    # 将数据label强制转换为指定的类型，astype函数是在副本上进行，并非修改原数组。
    # 从文件中load出来的数据类型是“class 'numpy.int16'”类型，需要进行类型转化
    label = label.astype("int")
    # 取数据集的行数，即是实例数：2000个实例
    num_samples = data.shape[0]

    # 用shuffle_index函数将2000这个整数随机划分成两个长为1000的list，list中的元素是2000以内的整数
    # data是<class 'numpy.ndarray'>的二维矩阵，将上一步的list传入，会将data中按list中的元素按行取出
    # 这两步就是将（2000，294）的data二维矩阵划分成了两个（1000，294）的二维矩阵，分别代表训练集和测试集
    # 针对label这个（2000，5）的二维矩阵也是这么操作，而且采集时用的是同一组list，保证实例和标签对应
    train_index, test_index = shuffle_index(num_samples)
    train_data = data[train_index]
    train_label = label[train_index]
    test_data = data[test_index]
    test_label = label[test_index]
    # print("train_data", train_data)
    # print("train_label", train_label)
    # print("test_data", test_data)
    # print("test_label", test_label)
    return [train_data, train_label, test_data, test_label]


"""
数据集文件：data、label文件
"""
if __name__ == '__main__':
    dataset = "image"
    # 初始化数据集、标签集、测试数据标签集
    train_data, train_label, test_data, test_label = make_data(dataset)

    # 构造森林，将另个森林级联，最大层数设为10，5重交叉验证
    model = Cascade(dataset, max_layer=10, num_forests=4, n_fold=5, step=3)
    # 训练森林，传入训练集、训练标签、指标名称、每个森林中的树的数量设为40
    model.train(train_data, train_label, "hamming loss", n_estimators=40)

    test_prob = model.predict(test_data, "hamming loss")

    value = do_metric(test_prob, test_label, 0.5)
    meatures = ["hamming loss", "one-error", "coverage", "ranking loss", "average precision", "macro-auc"]
    res = zip(meatures, value)
    for item in res:
        print(item)
