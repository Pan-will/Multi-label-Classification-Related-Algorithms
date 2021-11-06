"""
不划分特征，每层一个多标签分类器。
训练时批处理；
测试时分批增量式处理。
"""
from MultiLabelHoeffdingTree import *


class Cascade:
    def __init__(self, n_labels, n_features, layer=6):
        self.n_labels = n_labels  # 当前数据集的标签数量
        self.n_features = n_features  # 原始特征数量
        self.layer = layer  # 层数
        self.buffer_size = 15  # 缓冲区大小，设置为5
        self.eta = round(n_features / n_labels, 2)  # 每层的自适应因子，每层相同。eta = len(原始特征向量) / len(增广特征向量)

        self.every_layer_pred = []  # 存放每一层的输出值
        self.vfdt_deep_forest = []  # 存放分层式模型
        self.main_HT = []  # 每层用完整特征构建一个分类器，用于择优

    def train(self, train_data, train_label):
        for layer_index in range(self.layer):
            clf = MultiLabelHoeffdingTreeClassifier()
            clf.fit(train_data, train_label)
            self.vfdt_deep_forest.append(clf)
        print("共有%d层。" % (self.layer))

    def predict(self, test_data, test_label):
        final_res = []  # 存放每条实例的预测标签值
        # 缓冲区
        temp_inst = []
        temp_label = []
        for index, instance in enumerate(test_data):
            temp_inst.append(instance)
            temp_label.append(test_label[index])
            # 若缓冲区满了
            if len(temp_inst) == self.buffer_size:
                # 新一批数据到达，清空缓存。每一批都从第一层处理。
                self.every_layer_pred = []
                print("第%d批数据到达，当前缓存区已满。" % (index / self.buffer_size + 1))
                temp = self.do_predict(np.array(temp_inst), np.array(temp_label))  # 返回值temp是np.array类型
                final_res.append(temp.T)
                # 清空缓存区
                temp_inst = []
                temp_label = []
        return final_res

    def do_predict(self, temp_inst, temp_label):
        for layer_index in range(self.layer):
            if layer_index == 0:  # 第一层不拼接
                cur_layer_allBasePred = self.forcast_then_update_1(temp_inst, temp_label)  # 返回值是np.array类型
                self.every_layer_pred.append(cur_layer_allBasePred)
            else:
                cur_layer_allBasePred = self.forcast_then_update_N(temp_inst, temp_label, layer_index, self.every_layer_pred[layer_index - 1])
                self.every_layer_pred.append(cur_layer_allBasePred)
        return self.every_layer_pred[-1]  # np.array()类型

    def forcast_then_update_1(self, batch_data, batch_label):
        temp_layer_pred = self.vfdt_deep_forest[0].predict(batch_data)
        self.vfdt_deep_forest[0].partial_fit(batch_data, batch_label)
        return np.array(temp_layer_pred)

    def forcast_then_update_N(self, batch_data, batch_label, layer_index, last_layer_pred):
        last_layer_pred = last_layer_pred * self.eta  # np.array类型
        # print(batch_data.shape, last_layer_pred.shape)
        new_origin_feature = np.hstack((batch_data, last_layer_pred.T))  # 横向拼接
        # new_origin_feature.shape=(15, 300) 是符合要求的
        temp_layer_pred = self.vfdt_deep_forest[layer_index].predict(batch_data)
        self.vfdt_deep_forest[layer_index].partial_fit(batch_data, batch_label)
        return np.array(temp_layer_pred)