from node import Node, NodeInfo, NodePair, NodeType


# TODO: We should have a better abstraction for this other than string processing

def extract_left_corner(node: Node) -> Node:
    while node.left is not None:
        node = node.left
    return node


def expand_nt(node: Node) -> None:
    leftcorner_node = extract_left_corner(node.node_info.ref)
    leftcorner_node_info = leftcorner_node.node_info.copy(leftcorner_node)

    new_right_node = NodePair(node.node_info, leftcorner_node_info, parent=node)
    new_left_node = Node(leftcorner_node_info, node)

    node.set_left(new_left_node)
    node.set_right(new_right_node)


def expand_nt_nt(node: NodePair) -> None:
    parent_node = node.node_info2.ref.parent
    new_right_node = NodePair(node.node_info1, parent_node.node_info.copy(parent_node), parent=node)

    sibling_node = node.node_info2.ref.parent.right
    sibling_node_info = NodeInfo(sibling_node.node_info.type, sibling_node.node_info.label,
                                 ref=sibling_node)
    new_left_node = Node(sibling_node_info, parent=node)

    node.set_left(new_left_node)
    node.set_right(new_right_node)


def eps(node: NodePair) -> bool:
    """ Predicate that returns true for nodes of the type X-X """
    return node.node_info1.ref == node.node_info2.ref


def transform(cur: Node) -> None:
    if cur is None:
        return
    if cur.node_info.type == NodeType.NT:
        expand_nt(cur)
    elif cur.node_info.type == NodeType.NT_NT:
        assert isinstance(cur, NodePair)
        if not eps(cur):
            expand_nt_nt(cur)
    else:
        return
    transform(cur.left)
    transform(cur.right)
