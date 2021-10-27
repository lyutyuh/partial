from node import Node, NodeType, NodeInfo


def rev_rc_transform(node: Node, ref_node: Node) -> Node:
    if ref_node.right.node_info.type == NodeType.PT and ref_node.left.node_info.type == NodeType.NT:
        # X -> X word
        pt_node = Node(ref_node.right.node_info.copy(ref_node.right), node)
        if node.right is None:
            node.set_right(pt_node)
        else:
            node.set_left(pt_node)
            par_node = Node(NodeInfo(NodeType.NT, "X"), None)
            par_node.set_right(node)
            node = par_node
        return rev_rc_transform(node, ref_node.left)
    elif ref_node.right.node_info.type == NodeType.NT and ref_node.left.is_eps():
        # X -> X-X X
        if node.right is None:
            raise ValueError("When reaching the root the right branch should already exist")
        left_node = rev_rc_transform(Node(NodeInfo(NodeType.NT, "X")), ref_node.right)
        node.left = left_node
        left_node.set_parent(node)
        return node
    elif ref_node.right.node_info.type == NodeType.PT and ref_node.left.is_eps():
        # X -> X-X word
        if node.right is None:
            raise ValueError(
                "When reaching the end of the chain the right branch should already exist")
        node.left = Node(ref_node.right.node_info.copy(ref_node.right), node)
        return node
    elif ref_node.right.node_info.type == NodeType.NT and ref_node.left.node_info.type == NodeType.NT:
        # X -> X X
        left_node = rev_rc_transform(Node(NodeInfo(NodeType.NT, "X")), ref_node.right)
        node.left = left_node
        left_node.set_parent(node)
        par_node = Node(NodeInfo(NodeType.NT, "X"), None)
        par_node.set_right(node)
        return rev_rc_transform(par_node, ref_node.left)
