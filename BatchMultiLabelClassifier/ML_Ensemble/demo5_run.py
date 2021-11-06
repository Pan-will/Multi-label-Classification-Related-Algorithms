"""
运行demo5_IncreaseMLHT_ManyForest。
"""
import pandas as pd
from operator import itemgetter
from MyUtils import *
from demo5_IncreaseMLHT_ManyForest import Cascade
from Measure import *

# 各个超参数
prob_threshold = 0.99  # 阈值，用来确定基分类器个数
num_layer = 2  # 模型层数


def load_csv(dataset_name, label_name):
    data_split = 0.5
    # 数据集目录和文件
    data_dir = 'D:/Pycharm2020.1.3/WorkSpace/BatchMultiLabelClassifier/ML_Dataset'

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
    # 获取当前数据集的特征数、标签数
    n_features = train.shape[1]
    n_labels = train_label.shape[1]

    # 返回值是训练数据、测试数据、标签数
    return [train, train_label, test, test_label, n_samples, n_labels, n_features]


if __name__ == '__main__':
    dataset_name = '/scene_data.csv'
    label_name = '/scene_label.csv'
    # 初始化数据集、测试数据集、标签集，并获取实例数、特征数、标签数
    train_data, train_label, test_data, test_label, n_samples, n_labels, n_features = load_csv(dataset_name, label_name)
    if train_data.any():
        print(dataset_name[1:-9], "数据集加载正确！！!")

    # 查看train_data、train_label、test_data、test_label的shape
    print("训练集实例数量：", train_data.shape[0], "; 测试集实例数量：", test_data.shape[0])
    print("每条实例特征数量：", train_data.shape[1])
    print("train_data、train_label shape：", train_data.shape, train_label.shape)
    print("test_data、test_label shape：", test_data.shape, test_label.shape)

    # 根据特征数量和定好的阈值，确定要构建的基分类器数量
    n_base_learners = get_baseLearner_number(n_features, prob_threshold)
    print("每层需要构建", n_base_learners, "个基学习器。")
    print("\n+++++++++++++++++开始训练++++++++++++++++")

    model = Cascade(n_labels, n_base_learners, n_features, num_layer)
    model.train(train_data, train_label)
    test_prob = model.predict(test_data, test_label)
    test_prob = np.array(test_prob)
    # 此时test_prob的形状：(80, 15, 6)，下面需要reshape成（15*80，6）
    final_res = np.reshape(test_prob, (-1, test_prob.shape[2]))
    # 结果的形状： (80, 15, 6)
    print("结果的形状：", final_res.shape, "测试标签集的形状：", test_label[:len(final_res)].shape)
    value = do_metric(test_prob, test_label[:len(test_prob)], 0.38)
    meatures = ["hamming loss", "one-error", "coverage", "ranking loss", "macro_f1", "average precision", "macro-auc"]
    res = zip(meatures, value)
    for item in res:
        print(item)
