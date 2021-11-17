import random
import string
from enum import Enum

import numpy as np
from nltk import ParentedTree
from nltk import Tree


class NodeType(Enum):
    NT = 0
    NT_NT = 1
    PT = 2


def find_node_type(node: ParentedTree) -> NodeType:
    if len(node) == 1:
        return NodeType.PT
    elif node.label().find("\\") != -1:
        return NodeType.NT_NT
    else:
        return NodeType.NT


def is_node_epsilon(node: ParentedTree) -> bool:
    if len(node.leaves()) == 1 and node.leaves()[0] == "EPS":
        return True
    return False


def is_topo_equal(first: ParentedTree, second: ParentedTree) -> bool:
    if len(first) == 1 and len(second) != 1:
        return False
    if len(first) != 1 and len(second) == str:
        return False
    if len(first) == 1 and len(second) == 1:
        return True
    return is_topo_equal(first[0], second[0]) and is_topo_equal(first[1], second[1])


def random_tree(node: ParentedTree, depth=0, p=.75, cutoff=7) -> None:
    """ sample a random tree
    @param input_str: list of sampled terminals
    """
    if np.random.binomial(1, p) == 1 and depth < cutoff:
        # add the left child tree
        left_label = "X/" + str(depth)
        left = ParentedTree(left_label, [])
        node.insert(0, left)
        random_tree(left, depth=depth + 1, p=p, cutoff=cutoff)
    else:
        label = "X/" + str(depth)
        left = ParentedTree(label, [random.choice(string.ascii_letters)])
        node.insert(0, left)

    if np.random.binomial(1, p) == 1 and depth < cutoff:
        # add the right child tree
        right_label = "X/" + str(depth)
        right = ParentedTree(right_label, [])
        node.insert(1, right)
        random_tree(right, depth=depth + 1, p=p, cutoff=cutoff)
    else:
        label = "X/" + str(depth)
        right = ParentedTree(label, [random.choice(string.ascii_letters)])
        node.insert(1, right)


def rc_preprocess(tree: Tree, remove_top=False) -> ParentedTree:
    from transform import RightCornerTransformer

    if remove_top:
        cut_off_tree = tree[0] # remove TOP
    else:
        cut_off_tree = tree
    cut_off_tree.collapse_unary(collapsePOS=True, collapseRoot=True)
    cut_off_tree.chomsky_normal_form()
    ptree = ParentedTree.convert(cut_off_tree)
    root_label = ptree.label()
    tree_rc = ParentedTree(root_label, [])
    RightCornerTransformer.transform(tree_rc, ptree, ptree)
    return tree_rc


def rc_postprocess(rc_tree: ParentedTree, root_label) -> Tree:
    from transform import RightCornerTransformer

    tree = ParentedTree(root_label, ["", ""])
    tree = RightCornerTransformer.rev_transform(tree, rc_tree)
    tree = Tree.convert(tree)
    tree.un_chomsky_normal_form()
    rooted_tree = Tree("TOP", [tree])
    return rooted_tree
