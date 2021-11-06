import numpy as np

from sklearn.tree import DecisionTreeClassifier

from InfoGainSplitMetric import InfoGainSplitMetric

from Node import Node


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

if __name__ == '__main__':
    # 测试脚本
    X = np.array([[8, 9, 3], [7, 15, -1], [5, 9, 3], [1, 5, 7], [2, 8, -3]]).reshape(5, 3)
    Y = np.array([[1, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]]).reshape(5, 4)
    print("X:\n", X)
    print("Y:\n", Y)
    # reshape(行,列)，若其中一个参数是-1，则代表该参数不作考虑。如(1,-1)表示：将array构造成1行若干列就行，而具体多少列不用管。
    x_new = np.array([5, 2, 3]).reshape(1, -1)
    print("新实例:\n", x_new)

    clf = MultiLabelHoeffdingTreeClassifier()
    clf.fit(X, Y)
    res = clf.predict(x_new)
    print("新实例和其预测值拼接后的果：\n", type(res), type(res[0]), res)
