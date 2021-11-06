"""
层类
"""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

class Layer:
    # 构建层类，参数列表：每个森林树的数量=40，森林数量=2，标签数=5（不同数据集，标签数不同），步数=3(用来求每一层中森林的最大深度，按层递增)，层序号（从0开始），交叉验证倍数
    def __init__(self, n_estimators, num_forests, num_labels, step=3, layer_index=0, fold=0):
        """
        :param n_estimators: 每个森林中树的数量=40
        :param num_forests: 森林数量=4
        :param num_labels: 标签数量
        :param step: 步数=3
        :param layer_index: 层序号
        :param fold:
        """
        self.n_estimators = n_estimators
        self.num_labels = num_labels
        self.num_forests = num_forests
        self.layer_index = layer_index
        self.fold = fold
        self.step = step
        self.model = []

    # 参数是原train_data（251，68），train_label(251,174)进一步划分过的：(167, 68)、(167, 174)
    # 逐个森林分别训练分类器，并挨个放到self.model中
    def train(self, train_data, train_label):
        """
        :param train_data: 训练数据集
        :param train_label: 训练数据对应的标签
        :return:
        """
        # 在第一层中，每个森林中有40棵树，然后比上一层增加20棵树，直到树数达到100，最多100棵树；
        n_estimators = min(20 * self.layer_index + self.n_estimators, 100)
        # 最大深度 = 步数*层序号 + 步数
        max_depth = self.step * self.layer_index + self.step

        # 遍历森林块，从cascade层传递过来num_forests=4
        for forest_index in range(self.num_forests):
            # 第偶数个森林，用随机森林分类器，bootstrap参数值默认True
            if forest_index % 2 == 0:
                """
                参考博文：https://blog.csdn.net/w952470866/article/details/78987265/
                随机森林分类器。随机森林是一种元估计量，它适合数据集各个子样本上的许多决策树分类器，
                并使用平均数来提高预测准确性和控制过度拟合。
                子样本大小始终与原始输入样本大小相同，但是如果bootstrap = True（默认值），则将替换绘制样本。
                为了降低内容消耗，决策树的复杂度和大小应该通过设置这些参数值来控制。
                参数列表：
                 n_estimators: Any = 10,森林里（决策）树的数目。
                 criterion: Any = "gini",衡量分裂质量的性能（函数）,Gini不纯度和Gini系数没有关系。
                 max_depth: Any = None,（决策）树的最大深度
                 min_samples_split: Any = 2,分割内部节点所需要的最小样本数量,默认值2
                 min_samples_leaf: Any = 1,需要在叶子结点上的最小样本数量，默认值1
                 min_weight_fraction_leaf: Any = 0.,一个叶子节点所需要的权重总和（所有的输入样本）的最小加权分数。当sample_weight没有提供时，样本具有相同的权重
                 max_features: Any = "auto",寻找最佳分割时需要考虑的特征数目
                 max_leaf_nodes: Any = None,以最优的方法使用max_leaf_nodes来生长树。最好的节点被定义为不纯度上的相对减少。如果为None,那么不限制叶子节点的数量。
                 min_impurity_decrease: Any = 0.,如果节点的分裂导致的不纯度的下降程度大于或者等于这个节点的值，那么这个节点将会被分裂。
                 min_impurity_split: Any = None,树早期生长的阈值。如果一个节点的不纯度超过阈值那么这个节点将会分裂，否则它还是一片叶子。
                 bootstrap: Any = True,建立决策树时，是否使用有放回抽样
                 oob_score: Any = False,是否使用袋外样本来估计泛化精度。
                 n_jobs: Any = 1,用于拟合和预测的并行运行的工作（作业）数量。如果值为-1，那么工作数量被设置为核的数量。
                 random_state: Any = None,RandomStateIf int，random_state是随机数生成器使用的种子; 如果是RandomState实例，random_state就是随机数生成器; 如果为None，则随机数生成器是np.random使用的RandomState实例。
                 verbose: Any = 0,控制决策树建立过程的冗余度 
                 warm_start: Any = False,当被设置为True时，重新使用之前呼叫的解决方案，用来给全体拟合和添加更多的估计器，反之，仅仅只是为了拟合一个全新的森林。
                 class_weight: Any = None) -> None
                """
                clf = RandomForestClassifier(n_estimators=n_estimators,
                                             criterion="gini",
                                             max_depth=max_depth,
                                             n_jobs=-1)
            # 第奇数个森林，用极端随机森林分类器
            # 一般情况下，极端随机森林分类器在分类精度和训练时间等方面都要优于随机森林分类器。
            else:
                clf = ExtraTreesClassifier(n_estimators=n_estimators,
                                           criterion="gini",
                                           max_depth=max_depth,
                                           n_jobs=-1)
            clf.fit(train_data, train_label)
            self.model.append(clf)
        self.layer_index += 1

    # 预测
    def predict(self, test_data):
        # 设置一个三维空矩阵，shape是（森林数，参数中test_data的行数，标签数）
        predict_prob = np.zeros([self.num_forests, test_data.shape[0], self.num_labels])
        # 遍历上一步逐个森林训练好的分类器
        for forest_index, clf in enumerate(self.model):
            # 每个分类器都做预测，单个分类器的预测结果predict_p的信息：嵌套list，维度（174，51）
            predict_p = clf.predict_proba(test_data)
            # print(type(predict_p), len(predict_p), len(predict_p[0]))
            # 遍历当前分类器的预测结果list
            for j in range(len(predict_p)):
                # 三维空矩阵[分类器序号，全部行，前j列] = 1-predict_p[j][:, 0]的转置矩阵
                predict_prob[forest_index, :, j] = 1 - predict_p[j][:, 0].T
        # 三维矩阵求和，axis取多少，就表明在哪个维度上求和；
        # axis=0表示矩阵内部对应元素之间求和，结果是一个矩阵，其维度与三维矩阵的第一个元素相同，只不过元素是求的和
        # 按列进行求和
        prob_avg = np.sum(predict_prob, axis=0)
        # 针对森林数取均值——注意不是针对分类器个数取均值
        # 求平均
        prob_avg /= self.num_forests
        prob_concatenate = predict_prob
        # 返回值是[预测值针对森林数取得均值， 当前层四个森林的预测值]
        return [prob_avg, prob_concatenate]

    def train_and_predict(self, train_data, train_label, val_data, test_data):
        self.train(train_data, train_label)
        val_avg, val_concatenate = self.predict(val_data)
        prob_avg, prob_concatenate = self.predict(test_data)

        return [val_avg, val_concatenate, prob_avg, prob_concatenate]
