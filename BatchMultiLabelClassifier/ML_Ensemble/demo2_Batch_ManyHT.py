"""
批处理
多个MultiLabelHoeffdingTreeClassifier()
没有划分特征，没有使用多个基分类器投票。
多个基分类器的结果相同。
"""
from sklearn.utils import shuffle
from Measure import *
import pandas as pd

from sklearn.metrics import accuracy_score
from MultiLabelHoeffdingTree import MultiLabelHoeffdingTreeClassifier


def load_csv():
    data_split = 0.6
    # 数据集目录和文件
    data_dir = 'D:/Pycharm2020.1.3/WorkSpace/BatchMultiLabelClassifier/ML_Dataset'
    dataset_name = '/scene_data.csv'
    label_name = '/scene_label.csv'

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

    # 获取每个实例的特征个数
    n_features = train_data.shape[1]
    # 阈值，用来计算需要多少个基分类器
    prob_threshold = 0.99
    # 根据每条实例的特征数量和实现设置的阈值，计算需要多少个基分类器
    n_base_trees = get_baseLearner_number(n_features, prob_threshold)

    # 存放构造的n_base_trees个基分类器
    multi_base_trees = []
    # 一组多标签霍夫丁树
    for _ in range(n_base_trees):
        multi_base_trees.append(MultiLabelHoeffdingTreeClassifier())

    # 遍历n_base_trees个分类器，逐个拟合
    for baseTree in multi_base_trees:
        baseTree.fit(train_data, train_label)
    # 预测
    multi_pre = []
    for baseTree in multi_base_trees:
        print("\n")
        # 每个分类器分别预测
        res = baseTree.predict(test_data)
        # 将预测结果转型成array类型
        res = np.array(res)
        # 将当前分类器的预测值求指标，并存放到外层list中，取每个指标的最优值
        value = do_metric(res.transpose(), test_label, 0.38)
        meatures = ["hamming loss", "one-error", "coverage", "ranking loss", "precision", "macro-auc"]
        res = zip(meatures, value)
        for item in res:
            print(item)
