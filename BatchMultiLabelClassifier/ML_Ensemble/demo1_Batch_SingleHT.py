"""
能跑。
批处理。
仅用一个 MultiLabelHoeffdingTreeClassifier()做的分类。
没有划分特征，没有使用多个基分类器投票。
"""
from sklearn.utils import shuffle
import pandas as pd
from Measure import *
from MultiLabelHoeffdingTree import MultiLabelHoeffdingTreeClassifier


def load_csv():
    data_split = 0.6
    # 数据集目录和文件
    data_dir = 'F:/GitHub/Multi-label-Classification-Related-Algorithms/BatchMultiLabelClassifier/ML_Dataset'
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
    print("train_data、train_label shape：", train_data.shape, train_label.shape)
    print("test_data、test_label shape：", test_data.shape, test_label.shape)

    # 实例化一个多标签霍夫丁树分类器
    clf = MultiLabelHoeffdingTreeClassifier()
    # 拟合数据、训练模型
    clf.fit(train_data, train_label)
    # 用模型做做测试数据的预测
    res = clf.predict(test_data)  # len(res)=174, len(res[0])=251, type(res)=<class 'list'>
    # 将预测结果转型成array类型
    res = np.array(res)
    print(res.shape)  # (174, 251), test_label.shape=(251, 174)
    # value:list，存放了根据预测值、真实标签、阈值而求得的各个指标的值
    value = do_metric(res.transpose(), test_label, 0.38)
    meatures = ["hamming loss", "one-error", "coverage", "ranking loss", "average precision", "macro-auc"]
    res = zip(meatures, value)
    for item in res:
        print(item)
