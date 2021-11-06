class SplitCandidate(object):
    # 用于处理拆分候选对象的类
    """Class for handling a split candidate."""
    def __init__(self, split_test, post_split_dists, merit):
        self.split_test = split_test
        self.post_split_class_distributions = post_split_dists
        self.split_merit = merit

    def num_splits(self):
        return len(self.post_split_class_distributions)