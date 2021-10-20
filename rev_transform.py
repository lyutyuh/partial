from node import NodePair, Node, NodeType, NodeInfo


def expand_leaf_nodes(node: Node) -> Node:
    if node.node_info.type == NodeType.NT_NT:
        assert isinstance(node, NodePair)
        return Node(NodeInfo(NodeType.NT_NT, node.get_first_node_label()), None)
    else:
        assert node.node_info.type == NodeType.PT
        return Node(NodeInfo(NodeType.PT, node.label), None)


def expand_nt_nt(node: NodePair, upper_subtree: Node, right_subtree: Node) -> Node:
    local_root = upper_subtree.left if upper_subtree.left is not None else upper_subtree

    local_root.set_right(right_subtree)
    right_subtree.set_parent(local_root)

    new_left_node = Node(node.node_info2, parent=local_root)
    local_root.set_left(new_left_node)

    return local_root


def rev_transform(node: Node) -> Node:
    if node.is_leaf():
        return expand_leaf_nodes(node)
    if node.node_info.type == NodeType.NT:
        return rev_transform(node.right)
    if node.node_info.type == NodeType.NT_NT:
        assert isinstance(node, NodePair)
        upper_subtree = rev_transform(node.right)
        right_subtree = rev_transform(node.left)
        return expand_nt_nt(node, upper_subtree, right_subtree)
