from enum import Enum


class NodeType(Enum):
    NT = 0
    NT_NT = 1
    PT = 2
    DN = 3


class NodeInfo:
    def __init__(self, type: NodeType, label: str, ref=None):
        self.type: NodeType = type
        self.label: str = label

        # this node points to the original tree
        self.ref = ref

    def set_ref(self, ref: 'Node'):
        self.ref = ref

    def copy(self, ref_node: 'Node'):
        return NodeInfo(self.type, self.label, ref=ref_node)

    @staticmethod
    def merge_two_infos(node_info1: 'NodeInfo', node_info2: 'NodeInfo'):
        new_label = node_info1.label + "-" + node_info2.label
        new_type = NodeType.NT_NT
        return NodeInfo(new_type, new_label)


class Node:
    def __init__(self, node_info: NodeInfo, parent=None, left=None, right=None):
        self.node_info = node_info

        self.left = left
        self.right = right
        self.parent = parent

        # Since it should be an attribute of the node for printing the tree
        self.label = self.get_label()

    def set_left(self, left: 'Node') -> None:
        self.left = left

    def set_right(self, right: 'Node') -> None:
        self.right = right

    def set_parent(self, parent: 'Node') -> None:
        self.parent = parent

    def is_leaf(self) -> bool:
        return self.left is None or self.right is None

    def get_label(self) -> str:
        return self.node_info.label

    def is_eps(self) -> bool:
        """ Predicate that returns true for nodes of the type X-X """
        if self.node_info.type != NodeType.NT_NT:
            return False

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        return self.label


class NodePair(Node):
    def __init__(self, node_info1: NodeInfo, node_info2: NodeInfo, parent=None, left=None,
                 right=None):
        self.node_info1 = node_info1
        self.node_info2 = node_info2

        super().__init__(NodeInfo.merge_two_infos(node_info1, node_info2), parent, left,
                         right)

    def get_first_node_label(self) -> str:
        return self.node_info1.label

    def get_second_node_label(self) -> str:
        return self.node_info2.label

    def get_label(self) -> str:
        return self.node_info1.label + "-" + self.node_info2.label

    def is_eps(self) -> bool:
        return self.node_info1.ref == self.node_info2.ref

    def __str__(self) -> str:   
        return self.get_label()

    def __repr__(self) -> str: 
        return self.get_label()


class DepNode(Node):
    def __init__(self, node_info: NodeInfo, parent=None, left=None, right=None, dep=None):
        super().__init__(node_info, parent, left, right)
        self.dep = dep

    def set_dep(self, dep: 'Node') -> None:
        self.dep = dep

