from numpy import *

from infogainsplitmetric import InfoGainSplitMetric
from sklearn.naive_bayes import GaussianNB
import numpy as np

"""
这个类创建构造hoefffding树的节点。
This class creates the nodes that constucts a Hoeffding tree
"""


class Node():
    def __init__(self):
        """Default values of a Node, constructs a node"""
        self.parent = None
        self.children = None
        self.map = None
        self.statistics = None
        self.first = 0
        self.second = 0
        self._min_frac_weight_for_two_branches_gain = 0.01
        self.INFO_GAIN_SPLIT = 1
        self._selected_split_metric = self.INFO_GAIN_SPLIT
        self._split_confidence = 0.0000001
        self._split_metric = InfoGainSplitMetric(self._min_frac_weight_for_two_branches_gain)
        self.column_name_second = None
        self.column_name_first = None
        self.tie_breaking = 0.05
        self.confidence = 0.95
        self.delta = 1 - self.confidence
        self.n = 0
        self.length_y = 0
        self.number_of_classes = 0
        self.splitCondition = None
        self.activeNode = True
        self.per_column = None

    def setParent(self, node):
        """Sets a node as parent"""
        self.parent = node

    def get_node_statistics(self):
        """Returns the node's statistics"""
        return self.statistics

    def compute_first_second_best(self):
        """
        Look in the map for get the instance per each class and per each attribute, the instances are
           saved in an array. Then is computed the method entropy (information gain in this case) om order
           to obtain the attribute with highest entropy and the second highest.
           self.first is the attribute with highest entropy.
           self.second is the attribute with the second highest entropy.
        """
        metric_temp = 0
        metric_max = 0
        self.first = 0
        for key_attribute in self.map:
            m2 = self.map[key_attribute]
            for key_class in m2:
                column = []
                for key_instance in m2[key_class]:
                    m3 = m2[key_class]
                    for count in range(0, m3[key_instance]):
                        column.append(key_instance)
                metric_temp = self._split_metric.get_metric_range(column)
                if metric_max <= metric_temp:
                    metric_max = metric_temp
                    self.second = self.first
                    self.first = metric_max
                    self.column_name_second = self.column_name_first
                    self.column_name_first = key_class
                    self.n = len(column)

    def compute_hoeffding_bound(self, range, delta, n):
        """Computes the hoeffding bound and returns its value"""
        return math.sqrt(((range * range) * math.log(1.0 / delta)) / (2.0 * n))

    def comparison_entropies_hf_bound(self):
        """First computes the hoeffding bound with the numbers of clasess at the moment.
           then it does the substaction of the values of self.first and self.second
           and lastly it compares if the substraction of the entopies is greater than the hoeffding
           bound. if so then the node must split.
           Note: for the range of the variable, for information gain the range is log c, where c is the number of classes."""
        c = math.log(self.number_of_classes)
        hoeffding_bound = self.compute_hoeffding_bound(c, self.delta, self.n)
        entropy_Xa_Xb = self.first - self.second
        if (entropy_Xa_Xb > hoeffding_bound):
            self.split(self.column_name_first)

    def split(self, column_name):
        """Process of spliting a node, for each y in the map it is created a node classified as child.
           each child  is loaded with its belonging statistics."""
        self.splitCondition = column_name
        self.children = {}
        self.activeNode = False
        previous = 0
        for y_val in self.map:
            for j_val in self.map[y_val][column_name]:
                child = Node()
                self.children[j_val] = child
                child_map = {y_val: {}}
                child_statistics = {y_val: {}}
                for element in self.statistics[y_val][column_name][j_val]:
                    for k in self.map[y_val]:
                        if k != self.column_name_first:
                            for k_val in self.statistics[y_val][k]:  # add one
                                if element in self.statistics[y_val][k][k_val]:
                                    if k not in child_statistics[y_val]:
                                        child_statistics[y_val] = {k: {}}
                                    if k_val not in child_statistics[y_val][k]:
                                        child_statistics[y_val][k] = {k_val: 1}
                                    else:
                                        child_statistics[y_val][k][k_val] += 1
                            if k not in child_map[y_val]:
                                child_map[y_val] = {k: self.map[y_val][k].copy()}
                self.children[j_val].set_statistics(child_map.copy(), child_statistics.copy())
        self.map = None
        self.statistics = None

    # temporary Need to add the real map and statistics
    def set_statistics(self, map, statistics):
        """For adding a map and statistics"""
        self.map = map
        self.statistics = statistics

    def predict(self, x):
        N = len(x)

        if self.activeNode:
            self.n += 1
            clf = GaussianNB()
            x_train = None
            y_train = None
            y_train_dict = {}
            count_total = 0
            for y_value in self.map:
                for j in self.map[y_value]:
                    count = 0
                    for j_value in self.map[y_value][j]:
                        count += self.map[y_value][j][j_value]
                count_total += count
                y_train_dict[y_value] = np.full((1, count), y_value)
                y_train = zeros((1, count_total))
            counter = 0
            for y_value in y_train_dict:
                for y_dict_value in y_train_dict[y_value]:
                    for i in range(len(y_dict_value)):
                        y_train[0][counter] = y_dict_value[i]
                        counter += 1

            y_train = y_train
            x_train = zeros((len(y_train[0]), len(x[0])))

            for y_value in self.map:
                x_train_column = 0
                for j in self.map[y_value]:
                    for j_value in self.map[y_value][j]:
                        x_train_row = 0
                        repetitions = self.map[y_value][j][j_value]
                        for k in range(repetitions):
                            if x_train_row < len(x_train) and x_train_column < len(x_train[0]):
                                x_train[x_train_row][x_train_column] = j_value
                            x_train_row += 1
                x_train_column += 1

            clf.fit(x_train, y_train[0])
            Y = clf.predict(x)
            return Y
        else:
            try:
                to_transfer = x[0][int(self.splitCondition)]
                e = (zeros(len(x[0] - 1)))
                for i in x[0]:
                    j = 0
                    if i != int(self.splitCondition):
                        e[j] = x[0][i]
                        j += 1
                e = [e]

                Y = self.children[str(to_transfer)].predict(e)
                return Y
            except KeyError:
                print('value ' + to_transfer + ' for ' + self.splitCondition + ' not in the tree')

    def update_statistics(self, x, y):
        """This method fits the x and y, if the node is active then the statistics are updating according the values of x an y
           It is also invoqued the methods for obtaing the first and second best
           if is not an active node, it goes to the child leaves and recursively is called again the method."""
        self.length_y = len(y[0])
        self.number_of_classes = len(x[0])
        if self.activeNode:
            if self.statistics is None:
                self.statistics = {}
            if self.map is None:
                self.map = {}
            for i in range(len(y)):
                y_val = str(y[0][i])
                if y_val not in self.map:
                    self.map[y_val] = {}
                if y_val not in self.statistics:
                    self.statistics[y_val] = {}
                for j in range(len(x[0]) - 1):
                    value = x[0][j]

                    if str(j) not in self.statistics[y_val]:
                        l = []
                        l.append(self.n)
                        self.statistics[y_val][str(j)] = {str(value): l}
                    else:
                        if str(value) not in self.statistics[y_val][str(j)]:
                            l = []
                            l.append(self.n)
                            self.statistics[y_val][str(j)] = {str(value): l}
                        else:
                            self.statistics[y_val][str(j)][str(value)].append(self.n)

                    if str(j) not in self.map[y_val]:
                        self.map[y_val][str(j)] = {str(value): 1}
                    else:
                        if str(value) not in self.map[y_val][str(j)]:
                            self.map[y_val][str(j)] = {str(value): 1}
                        else:
                            self.map[y_val][str(j)][str(value)] += 1

            self.compute_first_second_best()
            self.comparison_entropies_hf_bound()
        else:
            to_transfer = x[0][int(self.splitCondition)]
            e = (zeros(len(x[0] - 1)))
            for i in x[0]:
                j = 0
                if i != int(self.splitCondition):
                    e[j] = x[0][i]
                    j += 1
            e = [e]
            try:
                self.children[str(to_transfer)].update_statistics(e, y)
            except KeyError:
                child = Node()
                self.children[str(to_transfer)] = child
                self.children[str(to_transfer)].update_statistics(e, y)
