from sklearn.ensemble import BaseEnsemble
from sklearn import tree
import numpy as np
import pandas
from collections import Counter


class EnsembleClassifier(BaseEnsemble):
    def __init__(self, n_classifier=10, base_estimator="HoeffdingTree", buffer_size=1000):
        self.current_instance_number = 0
        self.n_classifier = n_classifier
        self.buffer_size = buffer_size
        self.instances_bag_X = []
        self.instances_bag_Y = []
        # self.L = -1  # number of labels (for multi-label compatibility)
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
        print(np.array(predict))
        # find the most frequent predict
        most_predicted = Counter(np.array(predict).flatten())
        label = most_predicted.most_common(1)[0][0]
        print(X, label)
        return label

    def fit(self, x, Y, sample_weight=None):
        # pass
        # 当前实例数量小于缓冲区大小
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
                model = tree.DecisionTreeClassifier()
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
