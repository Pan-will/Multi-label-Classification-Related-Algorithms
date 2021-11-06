from sklearn.ensemble import BaseEnsemble
from sklearn import tree
import numpy as np
import pandas
from collections import Counter
from my_classifier import *
from src.HoeffdingTree import HoeffdingTree


class EnsembleClassifier(BaseEnsemble):
    """
    此类用于创建一个基于流数据打包的集成方法。
    This class is to create a an ensemble method based on bagging for streaming data
    这个实现基于Oza和Russell的在线打包模型。
    This implementation is based on Oza and Russell's Online Bagging Models.
    ----------------------
    Parameters:
    ------------
    n_classifier: Number of different classifiers used for bagging
    base_estimator: Base Estimator of the Ensemble Method, such as HoeffdingTree, DecisionTreeClassifier,...
    buffer_size: Number of streaming instances kept by the buffer as a mini-batch.缓冲区作为小批量保留的流式处理实例数。


    **** Methods:
    fit(self, x, Y, sample_weight = None): This method is to train the streaming data.

    predict (self, X): predict label of X. This predict function uses Majority Vote to emit the result.

    """

    def __init__(self, n_classifier=10, base_estimator="HoeffdingTree", buffer_size=1000):
        self.current_instance_number = 0
        self.n_classifier = n_classifier
        self.buffer_size = buffer_size
        self.instances_bag_X = []
        self.instances_bag_Y = []
        # self.L = -1  #number of labels (for multi-label compatibility)
        for i in range(0, self.n_classifier):
            self.instances_bag_X.append([])
            self.instances_bag_Y.append([])
        self.baseEns = BaseEnsemble.__init__(self, base_estimator=base_estimator)

    def predict(self, X):
        # pass
        predict = []
        for i in range(0, self.n_classifier):
            temp = self.models[i].predict(X)
            predict.append(temp)
        print("====== Predictions: ")
        print(np.array(predict))
        # find the most frequent predict
        most_predicted = Counter(np.array(predict).flatten())
        print("====== Count: ")
        print(most_predicted)
        label = most_predicted.most_common(1)[0][0]
        print("====== Final prediction:")
        print(X, label)
        return label

    def fit(self, x, Y, sample_weight=None):
        # pass

        if self.current_instance_number < self.buffer_size:
            # filing the batch
            # @nhatminh: for each classifier, draw a Poisson distribution of size(n_classifier)
            for i in range(0, self.n_classifier):
                n_poisson = np.random.poisson(1)
                # print "Poisson: ", n_poisson
                for j in range(0, n_poisson):
                    self.instances_bag_X[i].append(x)
                    self.instances_bag_Y[i].append(Y)
                    # print "Added ", x
                # print "Classifier[",i,"]: ",self.instances_bag_X[i], self.instances_bag_Y[i]
            self.current_instance_number += 1

        else:
            # training models
            # print "Full of batch, now we TRAIN!"
            self.models = []
            for i in range(0, self.n_classifier):
                # this line will be changed to use HoeffdingTree
                #    model = tree.DecisionTreeClassifier()
                model = HoeffdingTree()
                #    model = self.baseEns
                #    print len(self.instances_bag_X[i]), len(self.instances_bag_Y[i])
                if (len(self.instances_bag_X[i]) == 0):
                    break
                self.models.append(model.fit(np.array(self.instances_bag_X[i]).reshape(-1, 1),
                                             np.array(self.instances_bag_Y[i]).reshape(-1, 1)))
                #    print "Classifier[",i,"]: ",self.instances_bag_X[i], self.instances_bag_Y[i]
                #    print "Training..."
                self.instances_bag_X[i] = []
                self.instances_bag_Y[i] = []
            # after training, empty the array of X and Y
            self.current_instance_number = 0
        return self


# Test
# Uncomment those lines for testing

x = EnsembleClassifier(n_classifier=5, buffer_size=6)
x.fit(0, 1)
x.fit(1, 0)
x.fit(2, 1)
x.fit(3, 1)
x.fit(4, 0)
x.fit(5, 1)
x.fit(6, 0)
x.fit(7, 1)
x.fit(8, 0)
x.fit(3, 1)
x.fit(5, 1)
x.fit(6, 1)
x.fit(9, 1)

x.predict(6)
