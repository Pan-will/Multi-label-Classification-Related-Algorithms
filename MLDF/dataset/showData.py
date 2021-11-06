import numpy as np
from sklearn.utils import shuffle
# from sklearn.cross_validation import KFold
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from scipy import io


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


mat = io.loadmat('miml data.mat')
# print(type(mat), len(mat))
# print(type(mat.keys()))
bags = mat["bags"]
print(bags)
print(type(bags), len(bags), len(bags[0]))
print(type(bags[0][0]), len(bags[0][0]))
print(bags.shape)
# for k, v in mat.items():
#     print("key是：", k, "；value是：", v)
# 可以用values方法查看各个cell的信息


# data_csv = r'D:\Pycharm2020.1.3\WorkSpace\MLDF\dataset\CAL500_data.csv'
# label_csv = r'D:\Pycharm2020.1.3\WorkSpace\MLDF\dataset\CAL500_label.csv'

# with open(data_csv, encoding='utf-8') as f:
#     data = np.loadtxt(f, str, delimiter=",")
# with open(label_csv, encoding='utf-8') as f:
#     label = np.loadtxt(f, str, delimiter=",")

# 将数据label强制转换为指定的类型，astype函数是在副本上进行，并非修改原数组。
# 从文件中load出来的数据类型是“class 'numpy.int16'”类型，需要进行类型转化
# label = label.astype("int")
#
# print("data矩阵信息：", type(data[0]), data[0].shape, data.shape)
# print("label矩阵信息：", type(label[0]), label[0].shape, label.shape)

# a = np.zeros([2, 5, 5])
# b = np.zeros([2, 5, 5])
# print(a)
# print(b)
# print(a == b)
# temp = sum(a == b)
# print(temp)
# acc = sum(a == b) * 1.0 / len(a)
# print(1-acc.mean())

# a = np.random.randint(0, 10, (2, 3, 4, 5))
# print(a)
# print("原数组形状：", a.shape)
# print("transpose:", np.transpose(a, (1, 2, 0, 3)).shape)  # 重新指定轴0到3的顺序
# print("transpose2:", np.transpose(a, (2, 0, 1, 3)).shape)  # 重新指定轴0到3的顺序

# from sklearn.metrics import hamming_loss
#
# y_pred = [1, 2, 3, 4]
# y_true = [2, 2, 3, 4]
# print(hamming_loss(y_true, y_pred))

# num_samples = len(data)
# num_labels = len(label)
# print("实例数：", len(data))
# train_index, test_index = shuffle_index(num_samples)
# train_data = data[train_index]
# print("train_data的shape:", train_data.shape)
# train_label = label[train_index]
# test_data = data[test_index]
# test_label = label[test_index]
# print("train_data", type(train_data), train_data.shape, len(train_data))
# print("train_label", type(train_label), train_label.shape, len(train_label))
# print("test_data", type(test_data), test_data.shape, len(test_data))
# print("test_label", type(test_label), test_label.shape, len(test_label))
# testset = train_data.copy()
# print("testset", type(testset), testset.shape, len(testset))
# best_train_prob = np.empty(train_label.shape)
# print("best_train_prob", type(best_train_prob), best_train_prob.shape, len(best_train_prob))
# print(train_label[0])
# print(best_train_prob[0])
# best_concatenate_prob = np.empty([2, train_data.shape[0], train_label.shape[1]])
# print("best_concatenate_prob", type(best_concatenate_prob), best_concatenate_prob.shape, len(best_concatenate_prob))
# kf = KFold(len(train_label), n_folds=5, random_state=0)
# kf = KFold(n_splits=3, shuffle=False, random_state=None)
# print(type(kf), kf)
# 构造一个二维矩阵，规模是（实例数，标签数），构造的矩阵不为空
# prob = np.empty([num_samples, num_labels])
# 构造一个二维矩阵，规模是（森林数，实例数，标签数）=（2，实例数，标签数），构造的矩阵不为空
# prob_concatenate = np.empty([2, num_samples, num_labels])
# for train_i, test_i in kf.split(train_index):
#     print(type(train_i), train_i.shape, len(train_i))
#     print(type(test_i), test_i.shape, len(test_i))
#     clf = RandomForestClassifier(n_estimators=40, criterion="gini", max_depth=20, n_jobs=-1)
#     clf.fit(train_data, train_label)
#     predict_p = clf.predict_proba(test_data)

# X_train = train_data[train_i, :]
# print("X_train:", type(X_train), X_train.shape, len(X_train))
# X_val = train_data[test_i, :]
# print("X_val", type(X_val), X_val.shape, len(X_val))
# y_train = train_label[train_i, :]
# print("y_train", type(y_train), y_train.shape, len(y_train))
# print("*" * 80)


# print(type(predict_p[1]), len(predict_p[1]))
#
# yuan = predict_p[1][:, 0]
# yuanzhuan = predict_p[1][:, 0].T
# jianyuan = 1 - yuan
# jianyuanzhuan = 1 - yuanzhuan
#
# print(yuan == yuanzhuan)
# print(yuan == jianyuan)
# print(yuanzhuan == jianyuanzhuan)
#
# print(type(predict_p[1][:, 0]), len(predict_p[1][:, 0]), predict_p[1][:, 0].shape)

# predict_prob = np.zeros([2, train_data.shape[0], 174])
# print(predict_prob.shape, type(predict_prob))
# print(type(kf), kf)
# prob = np.empty([num_samples, train_label.shape[1]])
# print("prob", type(prob), prob.shape, len(prob))


# data_csv = r'D:\Pycharm2020.1.3\WorkSpace\MLDF\dataset\CAL500_data.csv'
# label_csv = r'D:\Pycharm2020.1.3\WorkSpace\MLDF\dataset\CAL500_label.csv'
# with open(data_csv, encoding='utf-8') as f:
#     data = np.loadtxt(f, str, delimiter=",")
# with open(label_csv, encoding='utf-8') as f:
#     label = np.loadtxt(f, str, delimiter=",")
#
# print("data矩阵信息：", type(data[0]), data[0].shape, data.shape)
# label = label.astype("int")
# print("label矩阵信息：", type(label[0]), label[0].shape, label.shape)
# num_samples = len(data)
# train_index, test_index = shuffle_index(num_samples)
# train_data = data[train_index]
# train_label = label[train_index]
# test_data = data[test_index]
# test_label = label[test_index]
# print("train_data", type(train_data), train_data.shape, len(train_data))
# print("train_label", type(train_label), train_label.shape, len(train_label))
# print("test_data", type(test_data), test_data.shape, len(test_data))
# print("test_label", type(test_label), test_label.shape, len(test_label))


# dataset = "image"
# data = np.load("image_data.npy")
# print(type(data[0][0]))
# print("data矩阵信息：", data.shape, type(data[0]), data[0].shape)
# num_samples = data.shape[0]
# print("数据集的行数，即实例个数：", num_samples)
#
# trainData, testData = shuffle_index(num_samples)
# print(len(testData), len(trainData), type(trainData), trainData)
#
# train_data = data[trainData]
# test_data = data[testData]
# wangwen = data[[1, 2]]
# print(wangwen)
# print(len(wangwen), type(wangwen), wangwen.shape)
# print(len(train_data), type(train_data), train_data.shape)

# print("\n")
# label = np.load("image_label.npy")
# print(type(label[0][0]))
# print("label矩阵信息：", label.shape, type(label[0]), label[0].shape)
