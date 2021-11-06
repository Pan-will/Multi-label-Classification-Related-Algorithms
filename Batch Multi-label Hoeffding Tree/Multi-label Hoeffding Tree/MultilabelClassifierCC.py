from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import tree


class MultilabelDecisionTreeClassifier(DecisionTreeClassifier):

    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 class_weight=None,
                 presort=False):
        super().__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_split=min_impurity_split,
            presort=presort)

    def fit(self, X, Y, sample_weight=None, check_input=True, X_idx_sorted=None):
        self.n_labels = len(np.transpose(Y))
        self.model = list(range(self.n_labels))
        self.prediction = list(range(self.n_labels))
        for i in range(self.n_labels):
            self.model[i] = DecisionTreeClassifier().fit(X, Y.T[i], sample_weight=sample_weight,
                                                         check_input=check_input, X_idx_sorted=X_idx_sorted)
            self.prediction[i] = self.model[i].predict(X, check_input=check_input)
            X = np.concatenate((X, self.prediction[i].reshape(len(self.prediction[i]), 1)), axis=1)
            print(X)
            print('------')
            print(self.prediction)
            # To visualize the Classifier Chain (CC), uncomment the lines above.
        return self

    def predict(self, x_new, check_input=True):
        pred = np.arange(len(self.model))
        for i in range(len(self.model)):
            pred[i] = self.model[i].predict(x_new, check_input=check_input)
            x_new = np.concatenate((x_new, pred[i].reshape(1, 1)), axis=1)
        return pred


# Testing scenario.
X = np.array([[8, 9, 3], [7, 15, -1], [5, 9, 3], [1, 5, 7], [2, 8, -3]]).reshape(5, 3)
Y = np.array([[1, 1, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]]).reshape(5, 4)
print(X)
print(Y)
x_new = np.array([5, 2, 3]).reshape(1, -1)
print(x_new)
clf = MultilabelDecisionTreeClassifier()
clf.fit(X, Y)
pred = clf.predict(x_new)
print(pred)
