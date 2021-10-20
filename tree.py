import numpy as np
from ppbtree import print_tree
from node import Node, DepNode, NodeType, NodeInfo


def random_tree(node: Node, label: str, depth=0, p=.75, cutoff=7) -> None:
    """ sample a random tree """

    left, right = None, None
    if np.random.binomial(1, p) == 1 and depth < cutoff:
        # add the left child tree
        left_label = label + "l"
        left = Node(NodeInfo(NodeType.NT, left_label), node)
        node.set_left(left)
        random_tree(left, left_label, depth=depth + 1)
    else:
        left = Node(NodeInfo(NodeType.PT, label + "l"), node)
        node.set_left(left)

    if np.random.binomial(1, p) == 1 and depth < cutoff:
        # add the right child tree
        right_label = label + "r"
        right = Node(NodeInfo(NodeType.NT, right_label), node)
        node.set_right(right)
        random_tree(right, right_label, depth=depth + 1)
    else:
        right = Node(NodeInfo(NodeType.PT, label + "r"), node)
        node.set_right(right)


def random_dep_tree(node: DepNode, counter=0, depth=0, p=.75, cutoff=2, labels=["A", "B", "C", "D", "E", "F", "G"]) -> None:
    """ sample a random dependency tree """

    left, right = None, None
    if np.random.binomial(1, p) == 1 and depth < cutoff:
        # add the left child tree
        left_label = labels[depth+1]
        left = DepNode(NodeInfo(NodeType.DN, left_label), node)
        node.set_left(left)
        counter = random_dep_tree(left, counter=counter, depth=depth+1)
    else:
        left = DepNode(NodeInfo(NodeType.PT, labels[depth+1]+"_"+str(counter)), node)
        node.set_left(left)
        counter += 1

    if np.random.binomial(1, p) == 1 and depth < cutoff:
        # add the right child tree
        right_label = labels[depth+1]
        right = DepNode(NodeInfo(NodeType.DN, right_label), node)
        node.set_right(right)
        counter = random_dep_tree(right, counter=counter, depth=depth + 1)
    else:
        right = DepNode(NodeInfo(NodeType.PT, labels[depth+1]+"_"+str(counter)), node)
        node.set_right(right)
        counter += 1

    # randomly sample dependent
    if np.random.binomial(1, .5) == 1:
        node.set_dep(node.right)
        node.label = node.label + "_" + node.right.label.split("_")[1]
    else:
        node.set_dep(node.left)    
        node.label = node.label + "_" + node.left.label.split("_")[1]

    return counter


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


