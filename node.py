from enum import Enum


class NodeType(Enum):
    NT = 0
    NT_NT = 1
    PT = 2


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
        self.depth = 0

    def set_left(self, left: 'Node') -> None:
        self.left = left

    def set_right(self, right: 'Node') -> None:
        self.right = right

    def set_parent(self, parent: 'Node') -> None:
        self.parent = parent

    def is_leaf(self) -> bool:
        return self.left is None or self.right is None

    def get_label(self) -> str:
        # sink label in node_info and node label
        return self.node_info.label

    def update_label(self) -> None:
        self.label = self.node_info.label

    def update_node_info(self, type, label) -> None:
        self.node_info.type = type
        self.node_info.label = label
        self.update_label()

    def is_right_child(self) -> bool:
        if self.parent is None:
            return False
        return self.parent.right == self

    def get_sibling(self) -> 'Node':
        if self.parent is None:
            return None
        return self.parent.left if self.is_right_child() else self.parent.right

    def is_eps(self) -> bool:
        """ Predicate that returns true for nodes of the type X-X """
        if self.node_info.type != NodeType.NT_NT:
            return False

    @staticmethod
    def is_topo_eq(node1: 'Node', node2: 'Node') -> bool:
        if node1 is None and node2 is not None:
            return False
        if node1 is not None and node2 is None:
            return False
        if node1 is None and node2 is None:
            return True
        return Node.is_topo_eq(node1.left, node2.left) and Node.is_topo_eq(node1.right, node2.right)

    def __str__(self) -> str:
        if self.node_info.type == NodeType.PT:
            return "\t"*self.depth + "(" + self.label + ")"
        else:
            if self.left.node_info.type == NodeType.PT and self.right.node_info.type == NodeType.PT:
                return "\t"*self.depth + "(" + self.label + " " + str(self.left) + " " + str(self.right) + ")"
            else:
                self.left.depth = self.depth + 1
                self.right.depth = self.depth + 1
                return "\t"*self.depth + "(" + self.label + "\n" + str(self.left) + "\n" + str(self.right) + ")"

    def __repr__(self) -> str:
        return self.label

    def __eq__(self, other):
        if self is None and other is not None:
            return False
        if self is not None and other is None:
            return False
        if self is None and other is None:
            return True
        if self.label != other.label:
            return False
        return self.left == other.left and self.right == other.right


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
