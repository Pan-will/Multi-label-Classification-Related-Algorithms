"""
能跑。增量式。
训练集批量训练；测试集增量更新。
预测结果比更新前好。
"""
from sklearn.utils import shuffle
from Measure import *
import pandas as pd

from Batch_MultiLabelHT.MultiLabelHoeffdingTree import MultiLabelHoeffdingTreeClassifier


def load_csv():
    data_split = 0.5
    # 数据集目录和文件
    data_dir = 'D:/Pycharm2020.1.3/WorkSpace/HMLDSC/ML_Dataset'
    # dataset_name = '/yeast_data.csv'
    # label_name = '/yeast_label.csv'
    # dataset_name = '/enron_data.csv'
    # label_name = '/enron_label.csv'
    # dataset_name = '/image_data.csv'
    # label_name = '/image_label.csv'
    dataset_name = '/scene_data.csv'
    label_name = '/scene_label.csv'
    # dataset_name = '/CAL500_data.csv'
    # label_name = '/CAL500_label.csv'

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
    print("数据集加载正确！！!")

    # 返回值是训练数据、测试数据、标签数
    return [train, train_label, test, test_label]


# 计算弱学习器的数量，根据公式，当公式所得概率值大于参数中的概率阈值时的m，此m即为弱学习器的个数
def get_baseLearner_number(n_features, prob=0.9):
    m = 1
    print("每条实例特征个数：", n_features)
    while True:
        p = (1 - (((n_features - 1) / n_features) ** (n_features * m)))
        print("计算基学习器的数量m=", m, "；p=", p, p > prob)
        if p > prob:
            break
        m += 1
    return m


# 计算弱学习器特征序号,参数是弱学习器的数量，特征向量list类型。
# 从参数传来的特征向量（长度为n）中随机选根号n个作为基学习器的特征。
# 返回值是一个list
def get_baseLearner_feature_indexes(n_trees, feature_vector):
    base_trees_features = []
    for i in range(n_trees):
        wl_features = np.random.choice(np.arange(feature_vector.__len__()), int(np.sqrt(feature_vector.__len__())))
        base_trees_features.append(wl_features)
    return base_trees_features


# 随机排列实例数，将实例划分为训练集和测试集
def shuffle_index(num_samples):
    a = range(0, num_samples)

    # 利用shuffle函数将序列a中的元素重新随机排列
    a = shuffle(a)

    # 取实例数的一半，上取整
    length = int((num_samples + 1) / 2)
    # 上半做训练集
    train_index = a[:length]
    # 下半做测试集
    test_index = a[length:]
    return [train_index, test_index]


if __name__ == '__main__':
    # 初始化数据集、测试数据集、标签集
    train_data, train_label, test_data, test_label = load_csv()

    # 查看train_data、train_label、test_data、test_label的shape
    print("训练集实例数量：", train_data.shape[0], "; 测试集实例数量：", test_data.shape[0])
    print("每条实例特征数量：", train_data.shape[1])
    print("train_data、train_label shape：", train_data.shape, train_label.shape)
    print("test_data、test_label shape：", test_data.shape, test_label.shape)

    # 获取测试集的实例数、特征数、标签数
    n_instance, n_feature = test_data.shape
    n_label = test_label.shape[1]

    # 实例化一个多标签霍夫丁树分类器
    clf = MultiLabelHoeffdingTreeClassifier()
    # 拟合训练集
    clf.fit(train_data, train_label)
    result1 = clf.predict(test_data)

    temp_inst = []
    temp_label = []
    for index, instance in enumerate(test_data):
        temp_inst.append(instance)
        temp_label.append(test_label[index])
        print("打印缓存区的长度：", len(temp_inst))
        # 若缓冲区满了
        if len(temp_inst) == 20:
            clf.partial_fit(np.array(temp_inst), np.array(temp_label))
            temp_inst = []
            temp_label = []


    # 预测结果
    result2 = clf.predict(test_data)

    # 将预测结果转型成array类型
    res1 = np.array(result1)
    res2 = np.array(result2)
    print(res1.shape, res2.shape, test_label.shape)
    value1 = do_metric(res1.transpose(), test_label, 0.5)
    value2 = do_metric(res2.transpose(), test_label, 0.5)
    meatures = ["hamming loss", "one-error", "coverage", "ranking loss", "average precision", "macro-auc"]
    for item in zip(meatures, value1):
        print(item)
    print("\n")
    for item in zip(meatures, value2):
        print(item)
