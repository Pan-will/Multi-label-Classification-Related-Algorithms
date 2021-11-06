from VFDT.VFDT_Node import *
import numpy as np

# very fast decision tree class, 即 hoeffding tree
class Vfdt:
    def __init__(self, features, delta=0.01, nmin=100, tau=0.1):
        """
        :features: list of data features；数据特征列表
        :delta: used to compute hoeffding bound, error rate；用于计算霍夫丁界，误差率
        :nmin: to limit the G computations
        :tau: to deal with ties
        """
        self.features = features
        self.delta = delta
        self.nmin = nmin
        self.tau = tau
        self.root = VfdtNode(features)
        self.n_examples_processed = 0

    # update the tree by adding training example
    # 通过添加训练实例来更新树
    def update(self, x, y):
        self.n_examples_processed += 1
        node = self.root.sort_example(x)
        node.update_stats(x, y)

        result = node.attempt_split(self.delta, self.nmin, self.tau)
        if result is not None:
            feature = result[0]
            value = result[1]
            self.node_split(node, feature, value)

    # split node, produce children
    # 节点分裂，产生子节点
    def node_split(self, node, split_feature, split_value):
        features = node.possible_split_features
        # print('节点分裂，产生子节点！')
        left = VfdtNode(features)
        right = VfdtNode(features)
        node.add_children(split_feature, split_value, left, right)

    # predict test example's classification
    # 预测测试实例的类别
    def predict(self, x_test):
        prediction = []
        if isinstance(x_test, np.ndarray) or isinstance(x_test, list):
            for x in x_test:
                leaf = self.root.sort_example(x)
                prediction.append(leaf.most_frequent())
            return prediction
        else:
            leaf = self.root.sort_example(x_test)
            return leaf.most_frequent()

    def print_tree(self, node):
        if node.is_leaf():
            print('Leaf')
        else:
            print(node.split_feature)
            self.print_tree(node.left_child)
            self.print_tree(node.right_child)
