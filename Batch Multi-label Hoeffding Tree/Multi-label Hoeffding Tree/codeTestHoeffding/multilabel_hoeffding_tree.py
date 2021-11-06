import numpy as np

from sklearn.tree import DecisionTreeClassifier

from infogainsplitmetric import InfoGainSplitMetric

from node import Node


class MultiLabelHoeffdingTreeClassifier:

    def __init__(self):
        self.hoeffdingTreesList = None
        self.predictionsList = None
        self.h1 = HoeffdingTreeClassifier()

    def fit(self, X, Y):
        self.h1.fit(X, Y)
        self.hoeffdingTreesList = []
        self.predictionsList = []
        for column in range(len(Y[0])):
            yvar = np.zeros((len(Y), 1))
            for row in range(len(Y)):
                yvar[row][0] = Y[row][column]
            self.hoeffdingTreesList.append(HoeffdingTreeClassifier())
            self.hoeffdingTreesList[column].fit(X, yvar)
        return self

    def partial_fit(self, X, Y=None):
        self.h1.partial_fit(X, Y)
        if self.hoeffdingTreesList is None:
            self.hoeffdingTreesList = list(range(len(Y[0])))
        if self.predictionsList is None:
            self.predictionsList = list(range(len(Y[0])))
        for column in range(len(Y[0])):
            yvar = np.zeros((len(Y), 1))
            for row in range(len(Y)):
                yvar[row][0] = Y[row][column]
            if self.hoeffdingTreesList[column]:
                self.hoeffdingTreesList[column].partial_fit(X, yvar)
            else:
                self.hoeffdingTreesList.append(HoeffdingTreeClassifier())
                self.hoeffdingTreesList[column].partial_fit(X, yvar)
        return self

    def predict(self, X):
        results = []
        c = None
        for i in range(len(self.hoeffdingTreesList)):
            c = self.hoeffdingTreesList[i].predict(X)
            results.append(c)
        return results


class HoeffdingTreeClassifier:

    def __init__(self):
        self.root = Node()
        j_max = 0

    def fit(self, X, Y):
        N, self.L = X.shape
        for i in range(N - 1):
            self.partial_fit([X[i]], [Y[i]])
        return self

    def partial_fit(self, x, y=None):
        if self.root is None:
            self.root = Node()
        self.root.update_statistics(x, y)
        return self

    def predict(self, X):
        c = self.root.predict(X)
        return c
