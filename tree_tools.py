import random
import string

import numpy as np

from node import Node, NodeType, NodeInfo


def random_tree(node: Node, input_str: [str], depth=0, p=.75, cutoff=7) -> None:
    """ sample a random tree
    @param input_str: list of sampled terminals
    """
    np.random.seed(7)
    if np.random.binomial(1, p) == 1 and depth < cutoff:
        # add the left child tree
        left_label = "X/" + str(depth)
        left = Node(NodeInfo(NodeType.NT, left_label), node)
        node.set_left(left)
        random_tree(left, input_str, depth=depth + 1, p=p, cutoff=cutoff)
    else:
        label = random.choice(string.ascii_letters)
        input_str.append(label)
        left = Node(NodeInfo(NodeType.PT, label), node)
        node.set_left(left)

    if np.random.binomial(1, p) == 1 and depth < cutoff:
        # add the right child tree
        right_label = "X/" + str(depth)
        right = Node(NodeInfo(NodeType.NT, right_label), node)
        node.set_right(right)
        random_tree(right, input_str, depth=depth + 1, p=p, cutoff=cutoff)
    else:
        label = random.choice(string.ascii_letters)
        input_str.append(label)
        right = Node(NodeInfo(NodeType.PT, label), node)
        node.set_right(right)


def example_tree_with_labels() -> Node:
    root = Node(NodeInfo(NodeType.NT, "S"), None)
    np = Node(NodeInfo(NodeType.NT, "NP"), root)
    vp = Node(NodeInfo(NodeType.NT, "VP"), root)
    root.set_left(np)
    root.set_right(vp)

    det = Node(NodeInfo(NodeType.PT, "Det(the)"), np)
    n = Node(NodeInfo(NodeType.PT, "N(dog)"), np)
    np.set_left(det)
    np.set_right(n)

    v = Node(NodeInfo(NodeType.PT, "V(ran)"), vp)
    adv = Node(NodeInfo(NodeType.PT, "Adv(fast)"), vp)
    vp.set_left(v)
    vp.set_right(adv)

    return root


def example_tree_without_labels() -> Node:
    root = Node(NodeInfo(NodeType.NT, "Z_c"), None)
    Y_b = Node(NodeInfo(NodeType.NT, "Y_b"), root)
    Y_c = Node(NodeInfo(NodeType.NT, "Y_c"), root)
    root.set_left(Y_c)
    root.set_right(Y_b)

    X_a = Node(NodeInfo(NodeType.PT, "X_a"), Y_b)
    X_b = Node(NodeInfo(NodeType.PT, "X_b"), Y_b)
    Y_b.set_left(X_b)
    Y_b.set_right(X_a)

    X_c = Node(NodeInfo(NodeType.PT, "X_c"), Y_c)
    X_d = Node(NodeInfo(NodeType.PT, "X_d"), Y_c)
    Y_c.set_left(X_d)
    Y_c.set_right(X_c)
    return root


def tetratagger_example() -> Node:
    root = Node(NodeInfo(NodeType.NT, "S"), None)
    one = Node(NodeInfo(NodeType.NT, "1"), root)
    E = Node(NodeInfo(NodeType.PT, "E"), root)
    root.set_left(one)
    root.set_right(E)

    A = Node(NodeInfo(NodeType.PT, "A"), one)
    two = Node(NodeInfo(NodeType.NT, "2"), one)
    one.set_left(A)
    one.set_right(two)

    B = Node(NodeInfo(NodeType.PT, "B"), two)
    three = Node(NodeInfo(NodeType.NT, "3"), two)
    two.set_left(B)
    two.set_right(three)

    C = Node(NodeInfo(NodeType.PT, "C"), three)
    D = Node(NodeInfo(NodeType.PT, "D"), three)
    three.set_left(C)
    three.set_right(D)
    return root
