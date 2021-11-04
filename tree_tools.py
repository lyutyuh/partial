import random
import string
from enum import Enum

import numpy as np
from nltk import ParentedTree as Tree


class NodeType(Enum):
    NT = 0
    NT_NT = 1
    PT = 2


def find_node_type(node: Tree) -> NodeType:
    if len(node) == 1:
        return NodeType.PT
    elif node.label().find("\\") != -1:
        return NodeType.NT_NT
    else:
        return NodeType.NT


def is_node_epsilon(node: Tree) -> bool:
    if len(node.leaves()) == 1 and node.leaves()[0] == "EPS":
        return True
    return False


def random_tree(node: Tree, depth=0, p=.75, cutoff=7) -> None:
    """ sample a random tree
    @param input_str: list of sampled terminals
    """
    if np.random.binomial(1, p) == 1 and depth < cutoff:
        # add the left child tree
        left_label = "X/" + str(depth)
        left = Tree(left_label, [])
        node.insert(0, left)
        random_tree(left, depth=depth + 1, p=p, cutoff=cutoff)
    else:
        label = "X/" + str(depth)
        left = Tree(label, [random.choice(string.ascii_letters)])
        node.insert(0, left)

    if np.random.binomial(1, p) == 1 and depth < cutoff:
        # add the right child tree
        right_label = "X/" + str(depth)
        right = Tree(right_label, [])
        node.insert(1, right)
        random_tree(right, depth=depth + 1, p=p, cutoff=cutoff)
    else:
        label = "X/" + str(depth)
        right = Tree(label, [random.choice(string.ascii_letters)])
        node.insert(1, right)
