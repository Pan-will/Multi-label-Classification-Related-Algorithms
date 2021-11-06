"""
waveform是单标签数据集。
Results for windwos size = 100 and hmldsc impact Threshold = 0.5
精度比较：
HMLDSC: % 77.02000000000001
VFDT: % 73.25
Random Forest: % 75.01500000000001
"""
import numpy as np
import time
from operator import itemgetter
import pandas as pd
from VFDT.VFDT import *
from VFDT.utils import *
import collections
from scipy import stats

# plot=1时结果可视化（画图表）；否则不画
plot = 1
# report=1则最后做结果统计；否则不做
report = 1

prob_threshold = 0.99
# 数据划分比例：60%用作训练集、40%用作测试集
data_split = 0.6
test_numbers = 10  # 测试次数
windows_size_array = [20, 30, 70, 100]  # 窗口大小
hmldsc_power_threshold_array = [0.2, 0.5, 0.7, 1]
# 3个5行5列的二维矩阵组成的三维矩阵
# results_acc、results_std分别存放不同条件下的训练精度、标准差
results_acc = np.zeros((3, len(windows_size_array), len(hmldsc_power_threshold_array)))
results_std = np.zeros((3, len(windows_size_array), len(hmldsc_power_threshold_array)))

# 数据集目录和文件
data_dir = 'dataset'
dataset_name = '/waveform.data'

# 加载并预处理数据集
# low_memory：分块加载到内存，再低内存消耗中解析。但是可能出现类型混淆。确保类型不被混淆需要设置为False。
data = pd.read_csv(data_dir + dataset_name, low_memory=False)
# 获取数据集中的实例数量
n_samples = data.shape[0]
# 根据开始设定的划分比例参数，将加载的数据集划分成训练集合测试集
n_training = int(data_split * n_samples)
train = data[:n_training]
test = data[n_training:]
# 取data的第一行为title
title = list(data.columns.values)
# print(title)
# 特征数量
features = title[:-1]
labels = title[-1]
n_features = features.__len__()
n_labels = labels.__len__()
print("特征数量n_features = ", n_features, "标签数量 = ", n_labels)
print("整个数据、训练集、测试集规模：", data.shape, train.shape, test.shape)
del data

print("\n*******************开始训练****************\n")

# 计算弱学习器的数量，根据公式，当公式所得概率值大于参数中的概率阈值时的m，此m即为弱学习器的个数
n_base_trees, _ = get_weaklearner_number(n_features, prob_threshold)
print("需要构建", n_base_trees, "个基学习器。\n")

# 外层循环是不同size的窗口
for windows_size_index, windows_size in enumerate(windows_size_array):
    # 遍历设置的不同的权重阈值
    for hmldsc_power_threshold_index, hmldsc_power_threshold in enumerate(hmldsc_power_threshold_array):
        hmldsc = []
        vfdt = []
        random_forest = []  # 随机森林
        for trn in range(test_numbers):
            print('正在进行第 {} 轮训练，windows size {} and HMLDSC power {}.'.format(trn + 1, windows_size,
                                                                             hmldsc_power_threshold))
            # 存放构造的n_base_trees个基分类器
            base_trees = []
            # 为n_base_trees个基学习器分配各自需要训练的特征
            base_trees_features = get_weaklearner_feature_indexes(n_base_trees, features)
            print("本轮对这", n_base_trees, "个基学习器随机划分的特征分别是：", base_trees_features)

            # 遍历上一步为每个基学习器划分的特征，调用VFDT类方法并设置参数，构造霍夫丁树，并添加到基学习器数组中
            for i in base_trees_features:
                base_trees.append(Vfdt(list(itemgetter(*i)(features)), delta=0.01, nmin=100, tau=0.5))
            # 再构造一棵临时用的霍夫丁树
            tree = Vfdt(features, delta=0.01, nmin=100, tau=0.5)

            # 以（index，Series）对的形式，按行遍历训练集，模拟流数据？按条读取实例
            for sample in train.iterrows():
                x = sample[1][:-1]  # 取本条实例的特征向量
                y = sample[1][-1]  # 取本条实例的标签，单标签
                # 取的x的type： <class 'pandas.core.series.Series'> 取的y的type： <class 'numpy.float64'>
                # print("取的x的type：", type(x), "取的y的type：", type(y))

                # 遍历基分类器，拟合划分的对应特征
                for base_tree, base_tree_feature in zip(base_trees, base_trees_features):
                    # print("更新树时，分配的x是：", itemgetter(*base_tree_feature)(x), "\n")
                    base_tree.update(itemgetter(*base_tree_feature)(x), y)

                tree.update(x, y)  # 拟合单棵树

            # 初始化一个hmldsc权值list，长度为当前窗口大小，初始都填充1
            hmldsc_power = collections.deque(maxlen=windows_size)
            for i in range(windows_size):
                hmldsc_power.append(1)

            # 基学习器的性能
            base_performance = np.zeros((n_base_trees, test.shape[0]))
            # 临时树的性能
            tree_performance = np.zeros((1, test.shape[0]))

            # 参数x:长度为特征数量（21）的list，y是概率值
            def update_n_test(x, y, counter):
                # 记录主树的预测值
                y_pred_tree = tree.predict([x])
                # 存放基学习器的预测值
                y_pred_base = []
                # 遍历基学习器及其对应的特征,记录每个基学习器的预测值
                for base_tree, base_tree_feature in zip(base_trees, base_trees_features):
                    y_pred_base.append(
                        base_tree.predict(np.asarray([itemgetter(*base_tree_feature)(item) for item in [x]])))
                # 将存放的每个基学习器的预测值格式转化成np.array数组格式
                y_pred_base = np.asarray(y_pred_base)

                # 若主树的预测值 == 真实标签值
                if y_pred_tree == y:
                    tree_performance[0, counter] = 1
                # 遍历每个基学习器的预测值
                for index_pre, base_pre in enumerate(y_pred_base):
                    # 若基学习器的预测值 == 参数中传来的值，更新base_performance
                    if base_pre == y:
                        base_performance[index_pre, counter] = 1

                # 处理并求得预测值y_pred
                # 若hmldsc权值list的和 > 当前窗口的阈值*窗口大小
                if sum(hmldsc_power) > (hmldsc_power_threshold * windows_size):
                    y_pred = get_predictions(np.asarray(y_pred_tree).reshape(-1, 1), y_pred_base)
                # 否则就用临时树的预测值
                else:
                    y_pred = y_pred_tree

                # 取占多数的预测值，参数是基学习器所有预测值矩阵的转置矩阵
                y_pred_base = get_majority(y_pred_base.T)
                if y_pred_base == y:
                    hmldsc_power.append(1)
                else:
                    hmldsc_power.append(0)

                # 用临时树（临时学习器）、拟合参数中数据
                tree.update(x, y)
                for base_tree, base_tree_feature in zip(base_trees, base_trees_features):
                    base_tree.update(itemgetter(*base_tree_feature)(x), y)
                # 返回值列表：预测值，临时树的预测值，基学习器预测值的最重要的值
                return y_pred, y_pred_tree, get_majority(y_pred_base)


            # 用right_count来计数预测正确的实例数
            right_count = 0
            y_pred_list = []
            tree_pred_list = []
            main_base_pred = []
            # 测试集。按行遍历，模拟流数据
            for row in test.iterrows():
                [a, b, c] = update_n_test(row[1][:-1].__array__(), row[1][-1], right_count)
                y_pred_list.append(a)
                tree_pred_list.append(b)
                main_base_pred.append(c)
                right_count += 1

            right_count = 0
            # test.values[:, -1]指：取test的最后一列，即是测试用例的真实值
            for i, j in zip(y_pred_list, test.values[:, -1]):
                if i[0] == j:
                    right_count += 1
            print("hmldsc本轮的预测正确率为：", right_count, "/", test.shape[0])
            hmldsc.append(100 * right_count / test.shape[0])

            right_count = 0
            for i, j in zip(tree_pred_list, test.values[:, -1]):
                if i[0] == j:
                    right_count += 1
            print("Tree本轮的预测正确率为：", right_count, "/", test.shape[0])
            vfdt.append(100 * right_count / test.shape[0])

            right_count = 0
            for i, j in zip(main_base_pred, test.values[:, -1]):
                if i[0] == j:
                    right_count += 1
            print("Random Forest本轮的预测正确率为：", right_count, "/", test.shape[0])
            random_forest.append(100 * right_count / test.shape[0])

            print("+" * 88)

        # 记录当前窗口和阈值下，10轮训练的精度均值
        results_acc[:, windows_size_index, hmldsc_power_threshold_index] = [np.mean(hmldsc), np.mean(vfdt),
                                                                            np.mean(random_forest)]
        # 记录当前窗口和阈值下，10轮训练的标准差均值
        results_std[:, windows_size_index, hmldsc_power_threshold_index] = [np.std(hmldsc), np.std(vfdt),
                                                                            np.std(random_forest)]

[_, i, j] = results_acc.shape
print("所求的准确率和标准差的规模分别是：", results_acc.shape, results_std.shape)

if report:
    for iidx in range(i):
        for jidx in range(j):
            print("*" * 20)
            print('Results for windwos size = {} and hmldsc impact Threshold = {}'
                  .format(windows_size_array[iidx], hmldsc_power_threshold_array[jidx]))
            acc = results_acc[:, iidx, jidx]
            print("精度比较：")
            print("HMLDSC: %", acc[0])
            print("VFDT: %", acc[1])
            print("Random Forest: %", acc[2])

            std = results_std[:, iidx, jidx]
            print("标准差比较：")
            print("HMLDSC: ", std[0])
            print("VFDT: ", std[1])
            print("Random Forest: ", std[2])
            print("*" * 20 + '\n\n')

# 将结果作图，并存放在设置的目录中
# 图存储目录
figure_save_dir = 'figures'
dataset_figure_dir = '/waveform'
if plot:
    import matplotlib.pyplot as plt

    for jidx in range(j):
        plt.figure(jidx)
        plt.grid('on')
        plt.title('Scores for HMLDSC Impact Threshold {}'.format(hmldsc_power_threshold_array[jidx]))
        ind = np.arange(i)
        width = 0.35
        p1 = plt.bar(ind, results_acc[1, :, jidx], width, yerr=results_std[1, :, jidx])
        p2 = plt.bar(ind, results_acc[0, :, jidx] - results_acc[1, :, jidx], width, yerr=results_std[0, :, jidx],
                     bottom=results_acc[1, :, jidx])
        plt.xticks(ind, windows_size_array)
        plt.ylim((60, 100))
        plt.xlabel('Windows Size')
        plt.ylabel('Accuracy')
        plt.legend((p1[0], p2[0]), ('VFDT', 'HMLDSC'))
        # plt.savefig(figure_save_dir + dataset_figure_dir + "/stackedbars_hyi-{}.svg".format(hmldsc_power_threshold_array[jidx]), format='svg')
        # plt.savefig(figure_save_dir + dataset_figure_dir + "/stackedbars_hyi-{}.png".format(hmldsc_power_threshold_array[jidx]), format='png')

    for iidx in range(i):
        plt.figure(j + iidx)
        plt.grid('on')
        plt.title('Scores for Windows Size {}'.format(windows_size_array[iidx]))
        ind = np.arange(j)
        width = 0.35
        p1 = plt.bar(ind, results_acc[1, iidx, :], width, yerr=results_std[1, iidx, :])
        p2 = plt.bar(ind, results_acc[0, iidx, :] - results_acc[1, iidx, :], width, yerr=results_std[0, iidx, :],
                     bottom=results_acc[1, iidx, :])
        plt.xticks(ind, hmldsc_power_threshold_array)
        plt.ylim((60, 100))
        plt.xlabel('HMLDSC Impact Threshold')
        plt.ylabel('Accuracy')
        plt.legend((p1[0], p2[0]), ('VFDT', 'HMLDSC'))
        # plt.savefig(figure_save_dir + dataset_figure_dir + "/stackedbars_wsz-{}.svg".format(windows_size_array[iidx]), format='svg')
        # plt.savefig(figure_save_dir + dataset_figure_dir + "/stackedbars_wsz-{}.png".format(windows_size_array[iidx]), format='png')

    for iidx in range(i):
        plt.figure(j + i + iidx)
        plt.grid('on')
        plt.title('Scores for Windows Size {}'.format(windows_size_array[iidx]))
        index = np.arange(j)
        bar_width = 0.25
        opacity = 0.8

        rects1 = plt.bar(index, results_acc[0, iidx, :], bar_width,
                         alpha=opacity,
                         label='HMLDSC')

        rects2 = plt.bar(index + bar_width, results_acc[1, iidx, :], bar_width,
                         alpha=opacity,
                         label='VFDT')

        rects3 = plt.bar(index + 2 * bar_width, results_acc[2, iidx, :], bar_width,
                         alpha=opacity,
                         label='Random Forest')

        plt.xlabel('Method')
        plt.ylabel('Accuracy')
        plt.xticks(index + 2 * bar_width, hmldsc_power_threshold_array)
        plt.legend()
        plt.ylim((60, 100))

        plt.tight_layout()
        # plt.savefig(figure_save_dir + dataset_figure_dir + "/comparison_wsz-{}.svg".format(windows_size_array[iidx]), format='svg')
        plt.savefig(figure_save_dir + dataset_figure_dir + "/comparison_wsz-{}.png".format(windows_size_array[iidx]),
                    format='png')

    for jidx in range(j):
        plt.figure(j + 2 * i + jidx)
        plt.grid('on')
        plt.title('Scores for HMLDSC Impact Threshold {}'.format(hmldsc_power_threshold_array[jidx]))
        index = np.arange(j)
        bar_width = 0.25
        opacity = 0.8

        rects1 = plt.bar(index, results_acc[0, :, jidx], bar_width, alpha=opacity, label='HMLDSC')

        rects2 = plt.bar(index + bar_width, results_acc[1, :, jidx], bar_width,
                         alpha=opacity,
                         label='VFDT')

        rects3 = plt.bar(index + 2 * bar_width, results_acc[2, :, jidx], bar_width,
                         alpha=opacity,
                         label='Random Forest')

        plt.xlabel('Method')
        plt.ylabel('Accuracy')
        plt.xticks(index + 2 * bar_width, windows_size_array)
        plt.legend()
        plt.ylim((60, 100))

        plt.tight_layout()
        plt.savefig(figure_save_dir + dataset_figure_dir + "/comparison_hyi-{}.svg".format(hmldsc_power_threshold_array[jidx]), format='svg')
        plt.savefig(figure_save_dir + dataset_figure_dir + "/comparison_hyi-{}.png".format(hmldsc_power_threshold_array[jidx]), format='png')
    plt.show()
