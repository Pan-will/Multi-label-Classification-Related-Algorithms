import mondrianforest
from sklearn import datasets, cross_validation
import pandas as pd


def load_data():
    # 数据集目录和文件
    data_dir = 'dataset'
    dataset_name = '/waveform.data'
    data_split = 0.6
    data = pd.read_csv(data_dir + dataset_name, low_memory=False)
    # 获取数据集中的实例数量
    n_samples = data.shape[0]
    # 根据开始设定的划分比例参数，将加载的数据集划分成训练集合测试集
    n_training = int(data_split * n_samples)
    train = data[:n_training]
    test = data[n_training:]
    # 取data的第一行为title
    title = list(data.columns.values)
    # 特征数量
    features = title[:-1]
    labels = title[-1]
    n_features = features.__len__()
    n_labels = labels.__len__()
    print("特征数量n_features = ", n_features, "标签数量 = ", n_labels)
    print("整个数据、训练集、测试集规模：", data.shape, train.shape, test.shape)
    return [train, test, n_samples, n_features]

if __name__ == '__main__':
    train, test, n_samples, n_features = load_data()
    print("实例数量、特征数量：", n_samples, n_features)
    print("train_data、test.shape：", train.shape, test.shape)
    forest = mondrianforest.MondrianForestClassifier(n_tree=10)






    # iris = datasets.load_iris()
    # forest = mondrianforest.MondrianForestClassifier(n_tree=10)
    # cv = cross_validation.ShuffleSplit(len(iris.data), n_iter=20, test_size=0.10)
    # scores = cross_validation.cross_val_score(forest, iris.data, iris.target, cv=cv)
    # print(scores.mean(), scores.std())
