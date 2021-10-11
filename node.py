from enum import Enum


class NodeType(Enum):
    NT = 0
    NT_NT = 1
    PT = 2


class Node:
    def __init__(self, type: NodeType, label: str, parent: 'Node', ref=None, left=None, right=None,
                 terminal=""):
        self.type = type
        self.label = label
        self.left = left
        self.right = right
        self.terminal = terminal
        self.parent = parent

        # this node points to the original tree
        self.ref = ref

    def set_left(self, left: 'Node') -> None:
        self.left = left

    def set_right(self, right: 'Node') -> None:
        self.right = right
