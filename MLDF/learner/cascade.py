"""
树级联成森林
森林最大层数：20
森林数：2
"""
from sklearn.cross_validation import KFold
# from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
from .measure import *
from .warpper import KfoldWarpper


class Cascade:
    # 本实验将最大层数（T）设置为20
    def __init__(self, dataname, max_layer=20, num_forests=4, n_fold=3, step=3):
        """
        :param dataname: 数据集名称
        :param max_layer: 森林最大层数，设为20
        :param num_forests: 每一层的森林数量，设为2
        :param n_fold: 每一层交叉验证倍数，设为5
        :param step: 迭代次数，设为3
        """
        self.dataname = dataname
        self.max_layer = max_layer
        self.n_fold = n_fold
        self.step = step
        self.layer_list = []
        self.num_forests = num_forests
        self.eta = []
        self.model = []

    # 针对六个多标签指标（用supervise表示），计算置信度，公式如论文中的表2所示，结果用alpha表示
    # P是预测值矩阵：准确的说是4个森林预测值的均值
    def compute_confidence(self, supervise, P):
        """
        :param supervise: string (e.g. "hamming loss", "one-error")，即指标
        :param P: array, whose shape is (num_samples, num_labels)
        :return alpha: array, whose shape is :
                        (num_samples, ) when supervise is instance-based measure,
                        and (num_labels, ) when supervise is label-based measure
        """
        # 获取实例数、标签数
        m, l = P.shape[0], P.shape[1]
        if supervise == "hamming loss":
            alpha = np.sum(np.abs(P - 0.5) + 0.5, axis=0) / m
        elif supervise == "one-error":
            alpha = np.max(P, axis=1)
        elif supervise == "ranking loss" or supervise == "average precision":
            forward_prod = np.sort(P, axis=1)
            backward_prod = 1 - forward_prod
            for j in range(1, l, 1):
                forward_prod[:, j] = forward_prod[:, j - 1] * P[:, j]
            for j in range(l - 2, -1, -1):
                backward_prod[:, j] = backward_prod[:, j + 1] * (1 - P[:, j])
            alpha = forward_prod[:, l - 1] + backward_prod[:, 0]
            for j in range(l - 1):
                alpha += forward_prod[:, j] * backward_prod[:, j + 1]
        elif supervise == "coverage":
            backward_prod = 1 - np.sort(P, axis=1)
            for j in range(l - 2, -1, -1):
                backward_prod[:, j] = backward_prod[:, j + 1] * (1 - P[:, j])
            alpha = backward_prod[:, 0]
            for j in range(l - 1):
                alpha += j * P[:, j] * backward_prod[:, j + 1]
            alpha = 1 - alpha / l
        elif supervise == "macro_auc":
            forward_prod = np.sort(P, axis=0)
            backward_prod = 1 - P.copy()
            for i in range(1, m, 1):
                forward_prod[i, :] = forward_prod[i - 1, :] * P[i, :]
            for i in range(m - 2, -1, -1):
                backward_prod[i, :] = backward_prod[i + 1, :] * (1 - P[i, :])
            alpha = forward_prod[m - 1, :] + backward_prod[0, :]
            for i in range(m - 1):
                alpha += forward_prod[i, :] * backward_prod[i + 1, :]
        return alpha

    # 在第一层中，每个森林中有40棵树，然后比上一层增加20棵树，直到树数达到100
    # 形参中指定了参数默认值，但是调用时以实参为准
    def train(self, train_data_raw, train_label_raw, supervise, n_estimators=40):
        """
        :param train_data_raw: array, whose shape is (num_samples, num_features)
        :param train_label_raw: array, whose shape is (num_samples, num_labels)
        :param supervise: string, (e.g. "hamming loss", "one-error")
        :param n_estimators: int, 每个森林块中树的数量，本实验中设为40
        """
        # 将参数中的训练集、对应的标签集复制一份
        train_data = train_data_raw.copy()
        train_label = train_label_raw.copy()
        # 标签数取的是训练标签集的列数
        self.num_labels = train_label.shape[1]
        # 初始化指标值，不同的指标初值不同
        best_value = init_supervise(supervise)
        # 统计为改进的层数，若近三层未改进，则停止训练
        bad = 0
        # 初始化一个和train_label矩阵一样规模的矩阵，但元素不是空
        best_train_prob = np.empty(train_label.shape)
        # 初始化一个三维矩阵：每层的森林数、实例数、标签数
        best_concatenate_prob = np.empty([self.num_forests, train_data.shape[0], self.num_labels])

        # max_layer = 20，遍历森林的每一层，逐层训练
        for layer_index in range(self.max_layer):
            # K折交叉验证：用sklearn.cross_validation 求kf，此包已经弃用，但有n_folds参数
            # 将训练数据集划分len(train_label)个互斥子集，
            #       每次用其中一个子集当作验证集，剩下的len(train_label)-1个作为训练集，
            #               进行len(train_label)次训练和测试，得到len(train_label)个结果
            # 为了防止过拟合，我们对森林的每一层都做了K折交叉验证
            # n_splits 表示划分为几块（至少是2）
            # shuffle 表示是否打乱划分，默认False，即不打乱
            # random_state 随机种子数,表示是否固定随机起点，Used when shuffle == True.
            # rkf = RepeatedKFold(n_splits=5, n_repeats=3)

            kf = KFold(len(train_label), n_folds=self.n_fold, shuffle=True, random_state=0)

            # print("cross_validation求得kf:", type(kf), kf)

            # 用from sklearn.model_selection 求kf
            # shuffle：在每次划分时，是否打乱
            #     ①若为Falses时，其效果等同于random_state等于整数，每次划分的结果相同
            #     ②若为True时，每次划分的结果都不一样，表示经过洗牌，随机取样的
            # kf = KFold(len(train_label), shuffle=True, random_state=0).split(train_data.shape[0])

            # 参数：森林数=2，每个森里中的树的数量=40，n_fold折交叉验证，层序号（1~20，for循环ing），步数=3
            kfoldwarpper = KfoldWarpper(self.num_forests, n_estimators, self.n_fold, kf, layer_index, self.step)
            # 参数：训练集、对应标签集；返回值是[预测值针对森林数取得均值， 4个森林的预测值]
            prob, prob_concatenate = kfoldwarpper.train(train_data, train_label)

            self.model.append(kfoldwarpper)
            # 第一层，没有拼接，4个森林的输出值即是第一层的输出值
            if layer_index == 0:
                best_train_prob = prob
                # 指标名称，训练标签集，阈值初值为0.5
                pre_metric = compute_supervise_vec(supervise, best_train_prob, train_label, 0.5)
            # 非第一层，可以做拼接了。
            else:
                # 先计算4个森林预测值的compute_supervise_vec
                now_metric = compute_supervise_vec(supervise, prob, train_label, 0.5)
                # precision和auc这两个指标的值越大越好
                if supervise == "average precision" or supervise == "macro_auc":
                    # indicator是一个非0即1的一维向量
                    indicator = now_metric < pre_metric
                else:
                    indicator = now_metric > pre_metric

                if np.sum(indicator) > 0:
                    # 计算置信度，参数是：当前指标，4个森林预测值的均值
                    confidence = self.compute_confidence(supervise, prob)
                    # 取置信度均值(算术平均值)作为阈值。
                    eta_t = np.mean(confidence[indicator])
                    train_indicator = confidence < eta_t
                    if supervise == "hamming loss" or supervise == "macro_auc":
                        prob[:, train_indicator] = best_train_prob[:, train_indicator]
                        prob_concatenate[:, :, train_indicator] = best_concatenate_prob[:, :, train_indicator]
                    else:
                        prob[train_indicator, :] = best_train_prob[train_indicator, :]
                        prob_concatenate[:, train_indicator, :] = best_concatenate_prob[:, train_indicator, :]
                else:
                    eta_t = 0

                self.eta.append(eta_t)

                best_train_prob = prob

                best_concatenate_prob = prob_concatenate

                pre_metric = compute_supervise_vec(supervise, best_train_prob, train_label, 0.5)

            value = compute_supervise(supervise, best_train_prob, train_label, 0.5)
            back = compare_supervise_value(supervise, best_value, value)
            if back:
                bad += 1
            else:
                bad = 0
                best_value = value

            # 若近3层没有更新，则舍弃当前层模型和阈值
            if bad >= 3:
                # print("cascade测试bad：", bad, "，近3层没有更新，则舍弃当前层模型和阈值")
                for i in range(bad):
                    self.model.pop()
                    self.eta.pop()
                break
            # 准备下一层数据
            # transpose函数：重新指定0，1，2三个轴的顺序
            prob_concatenate = best_concatenate_prob.transpose((1, 0, 2))
            prob_concatenate = prob_concatenate.reshape(prob_concatenate.shape[0], -1)
            print("第", layer_index + 1, "层交叉验证完成后做拼接，原矩阵、预测值矩阵的形状：", train_data_raw.shape, prob_concatenate.shape)
            # 将prob_concatenate拼接到train_data_raw下面，行数会改变，所以axis=1
            train_data = np.concatenate([train_data_raw.copy(), prob_concatenate], axis=1)
            print("下一层的输入特征矩阵的形状：", train_data.shape, "\n拼接阈值为：", self.eta)  # (322, 336)，0.989648033126294
        print("cascade层：深度森林模型训练完成！")

    # 针对不同指标，对原始测试数据做预测
    def predict(self, test_data_raw, supervise):
        """
        :param test_data_raw: array, whose shape is (num_test_samples, num_features)
        :return prob: array, whose shape is (num_test_samples, num_labels)
        """
        test_data = test_data_raw.copy()
        best_prob = np.empty([test_data.shape[0], self.num_labels])
        best_concatenate_prob = np.empty([self.num_forests, test_data.shape[0], self.num_labels])
        # 遍历每层的分类器和每层的阈值
        for clf, eta_t in zip(self.model, self.eta):
            # 分类器预测test_data，得到[预测值针对森林数取的均值， 按分类器存放的预测值]
            prob, prob_concatenate = clf.predict(test_data)
            confidence = self.compute_confidence(supervise, prob)
            indicator = confidence < eta_t
            if supervise == "hamming loss" or supervise == "macro_auc":
                prob[:, indicator] = best_prob[:, indicator]
                prob_concatenate[:, :, indicator] = best_concatenate_prob[:, :, indicator]
            else:
                prob[indicator, :] = best_prob[indicator, :]
                prob_concatenate[:, indicator, :] = best_concatenate_prob[:, indicator, :]
            best_concatenate_prob = prob_concatenate
            best_prob = prob
            prob_concatenate = best_concatenate_prob.transpose((1, 0, 2))
            prob_concatenate = prob_concatenate.reshape(prob_concatenate.shape[0], -1)
            test_data = np.concatenate([test_data_raw.copy(), prob_concatenate], axis=1)
        return best_prob
