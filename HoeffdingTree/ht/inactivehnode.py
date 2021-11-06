from ht.leafnode import LeafNode

class InactiveHNode(LeafNode):
    """
    非活动的hoefffding树节点（不支持增长）
    A Hoeffding Tree node that is inactive (does not support growth).
    """
    def __init__(self, class_distribution):
        super().__init__(class_distribution)

    def update_node(self, instance):
        self.update_distribution(instance)