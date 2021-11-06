"""
训练时批处理、每一层各基分类器划分特征。
测试数据分批、增量式到达。
分层式深度森林结构，除了第一层每层的输入由原始特征+前一层输出拼接而成。

__init__()
train()
predict()
第一层用当前实例的原始特征做输入，其它层用（增广向量*自适应因子+原始特征向量）做输入，其中增广向量即是上一层的输出；
所以对于每条实例，需要收集每层的输出；
欲收集每层的输出，对于当前层：先挨个收集没个基分类器的输出；
收集每个基分类器的输出：逐个标签预测，即收集每个标签的预测值；
"""
from MyUtils import *
from MultiLabelHoeffdingTree import *


class Cascade:
    def __init__(self, n_labels, n_base_learners, n_features, layer=6):
        self.n_labels = n_labels  # 当前数据集的标签数量
        self.n_base_learners = n_base_learners  # 每一层构建的基分类器数量
        self.n_features = n_features  # 原始特征数量
        self.layer = layer  # 层数
        self.buffer_size = 15  # 缓冲区大小，设置为5
        self.eta = np.round(n_features / n_labels, 2)  # 每层的自适应因子，每层相同。eta = len(原始特征向量) / len(增广特征向量)

        self.every_layer_pred = []  # 存放每一层的输出值
        self.vfdt_deep_forest = []  # 存放分层式模型
        self.layer_feature_index = []  # 每一层划分的特征下标，预测时要用
        self.old_feature_index = []  # 存放训练时划分的特征下标，预测后更新时要用
        self.main_HT = []  # 每层用完整特征构建一个分类器，用于择优

    # 训练：训练集整体训练，只存放每层的模型
    def train(self, train_data, train_label):
        for layer_index in range(self.layer):
            # 存放当前层的基分类器
            cur_layer_baser = []
            # 划分特征，参数是：要划分的份数，划分范围上界
            base_trees_features = get_baseLearner_feature_indexes(self.n_base_learners, self.n_features)
            self.layer_feature_index.append(base_trees_features)
            for index, column_list in enumerate(base_trees_features):
                print("构建第%d层第%d个分类器......" % (layer_index + 1, index + 1))
                clf = MultiLabelHoeffdingTreeClassifier()
                cur_train_data = train_data[:, column_list]
                # 从训练集中抽取划分给当前基分类器的特征
                clf.fit(cur_train_data, train_label)
                cur_layer_baser.append(clf)
            self.vfdt_deep_forest.append(cur_layer_baser)
        self.old_feature_index = self.layer_feature_index
        print("模型训练完成。")

    def predict(self, test_data, test_label):
        final_res = []  # 存放每条实例的预测标签值
        # 缓冲区，存放小批量数据
        temp_inst = []
        temp_label = []
        # 模拟流数据
        for index, instance in enumerate(test_data):
            temp_inst.append(instance)
            temp_label.append(test_label[index])
            # 若缓冲区满了
            if len(temp_inst) == self.buffer_size:
                # 新一批数据到达，清空缓存
                self.every_layer_pred = []
                print("第%d批数据到达，当前缓存区已满。" % (index / self.buffer_size + 1))
                # temp是np.array类型
                temp = self.do_predict(np.array(temp_inst), np.array(temp_label))
                # temp.shape = (6,15)
                final_res.append(temp.T)
                # 清空缓存区
                temp_inst = []
                temp_label = []
        return final_res

    """
    方法介绍：执行预测，每个小批量数据传入，经过每一层，先预测并收集预测结果（投票法前后都收集），再利用该批数据更新当前层模型。
    返回值：只返回最后一层的输出。
    """

    def do_predict(self, temp_inst, temp_label):
        # 逐层处理
        for layer_index in range(self.layer):
            # 第一层只能用原始特征
            if layer_index == 0:
                # 第一层所有基分类器的预测值,np.array()类型
                cur_layer_allBasePred = self.forcast_then_update_1(temp_inst, temp_label)
                # 存放经由投票法选出的当前层输出值
                # cur_layer_allBasePred:<class 'numpy.ndarray'> (3, 6, 15)
                self.every_layer_pred.append(get_manyArray_majory(cur_layer_allBasePred))
            # 非第一层，则输入特征矩阵需要经过拼接，所以预测、更新都要用新的特征矩阵
            else:
                cur_layer_allBasePred = self.forcast_then_update_N(temp_inst, temp_label, layer_index, self.every_layer_pred[layer_index - 1])
                # 存放经由投票法选出的当前层输出值
                self.every_layer_pred.append(get_manyArray_majory(cur_layer_allBasePred))
        return self.every_layer_pred[-1]  # np.array()类型

    """
    先预测然后更新分层式森林模型的第一层
    """

    def forcast_then_update_1(self, batch_data, batch_label):
        # 存放当前层的基分类器
        temp_layer_pred = []
        # 每一层随机划分特征
        base_trees_features = get_baseLearner_feature_indexes(self.n_base_learners, self.n_features)
        # 保存第一层为个基分类器划分的特征序号
        self.layer_feature_index[0] = base_trees_features
        for base_learner, column_list in zip(self.vfdt_deep_forest[0], self.layer_feature_index[0]):
            cur_train_data = batch_data[:, column_list]
            # cur_train_data：<class 'numpy.ndarray'>
            cur_pred = base_learner.predict(cur_train_data)
            temp_layer_pred.append(cur_pred)
            base_learner.partial_fit(cur_train_data, batch_label)
        return np.array(temp_layer_pred)

    """
    训练森林模型后N层,参数列表：当前到达的缓冲区数据、对应标签、当前层序号、上一层的输出
    除了第一层：后面每层都用拼接过后的新特征矩阵作预测、作更新
    """

    def forcast_then_update_N(self, batch_data, batch_label, layer_index, last_layer_pred):
        # 存放当前层的所有基分类器的预测值
        temp_layer_pred = []
        # 现将上一层的预测值乘以阈值
        last_layer_pred = last_layer_pred * self.eta
        # 上层输出和原特征横向拼接
        new_origin_feature = np.hstack((batch_data, last_layer_pred.T))
        base_trees_features = get_baseLearner_feature_indexes(self.n_base_learners, new_origin_feature.shape[1])
        # base_trees_features = get_baseLearner_feature_indexes(self.n_base_learners, self.n_features)
        # 保存当前层为各个基分类器划分的特征序号
        self.layer_feature_index[layer_index] = base_trees_features

        for base_learner, column_list in zip(self.vfdt_deep_forest[layer_index], self.layer_feature_index[layer_index]):
            # cur_train_data = batch_data[:, column_list]
            cur_train_data = new_origin_feature[:, column_list]
            # cur_train_data：<class 'numpy.ndarray'>
            cur_pred = base_learner.predict(cur_train_data)
            temp_layer_pred.append(cur_pred)
            base_learner.partial_fit(cur_train_data, batch_label)
        return np.array(temp_layer_pred)
