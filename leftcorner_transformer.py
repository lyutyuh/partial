from node import Node, NodeInfo, NodePair, NodeType


# TODO: We should have a better abstraction for this other than string processing

def extract_left_corner(node: Node) -> Node:
    while node.left is not None:
        node = node.left
    return node


def extract_right_corner(node: Node) -> Node:
    while node.right is not None:
        node = node.right
    return node


def expand_nt_left_corner(node: Node) -> None:
    leftcorner_node = extract_left_corner(node.node_info.ref)
    leftcorner_node_info = leftcorner_node.node_info.copy(leftcorner_node)

    new_right_node = NodePair(node.node_info, leftcorner_node_info, parent=node)
    new_left_node = Node(leftcorner_node_info, node)

    node.set_left(new_left_node)
    node.set_right(new_right_node)


def expand_nt_right_corner(node: Node) -> None:
    rightcorner_node = extract_right_corner(node.node_info.ref)
    rightcorner_node_info = rightcorner_node.node_info.copy(rightcorner_node)

    new_left_node = NodePair(node.node_info, rightcorner_node_info, parent=node)
    new_right_node = Node(rightcorner_node_info, node)

    node.set_left(new_left_node)
    node.set_right(new_right_node)


def expand_nt_nt_left_corner(node: NodePair) -> None:
    parent_node = node.node_info2.ref.parent
    new_right_node = NodePair(node.node_info1, parent_node.node_info.copy(parent_node), parent=node)

    sibling_node = node.node_info2.ref.parent.right
    sibling_node_info = NodeInfo(sibling_node.node_info.type, sibling_node.node_info.label,
                                 ref=sibling_node)
    new_left_node = Node(sibling_node_info, parent=node)

    node.set_left(new_left_node)
    node.set_right(new_right_node)


def expand_nt_nt_right_corner(node: NodePair) -> None:
    parent_node = node.node_info2.ref.parent
    new_left_node = NodePair(node.node_info1, parent_node.node_info.copy(parent_node), parent=node)

    sibling_node = node.node_info2.ref.parent.left
    sibling_node_info = NodeInfo(sibling_node.node_info.type, sibling_node.node_info.label,
                                 ref=sibling_node)
    new_right_node = Node(sibling_node_info, parent=node)

    node.set_left(new_left_node)
    node.set_right(new_right_node)


def eps(node: NodePair) -> bool:
    """ Predicate that returns true for nodes of the type X-X """
    return node.node_info1.ref == node.node_info2.ref


def left_corner_transform(cur: Node) -> None:
    if cur is None:
        return
    if cur.node_info.type == NodeType.NT:
        expand_nt_left_corner(cur)
    elif cur.node_info.type == NodeType.NT_NT:
        assert isinstance(cur, NodePair)
        if not eps(cur):
            expand_nt_nt_left_corner(cur)
    else:
        return
    left_corner_transform(cur.left)
    left_corner_transform(cur.right)


def right_corner_transform(cur: Node) -> None:
    if cur is None:
        return
    if cur.node_info.type == NodeType.NT:
        expand_nt_right_corner(cur)
    elif cur.node_info.type == NodeType.NT_NT:
        assert isinstance(cur, NodePair)
        if not eps(cur):
            expand_nt_nt_right_corner(cur)
    else:
        return
    right_corner_transform(cur.left)
    right_corner_transform(cur.right)

