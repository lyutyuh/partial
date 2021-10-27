from node import Node, NodeType, NodeInfo
from visualize import print_tree


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


def rev_lc_transform(node: Node, ref_node: Node) -> Node:
    if ref_node.left.node_info.type == NodeType.PT and ref_node.right.node_info.type == NodeType.NT:
        # X -> word X
        pt_node = Node(ref_node.left.node_info.copy(ref_node.left), node)
        if node.left is None:
            node.set_left(pt_node)
        else:
            node.set_right(pt_node)
            par_node = Node(NodeInfo(NodeType.NT, "X"), None)
            par_node.set_left(node)
            node = par_node
        return rev_lc_transform(node, ref_node.right)
    elif ref_node.left.node_info.type == NodeType.NT and ref_node.right.is_eps():
        # X -> X X-X
        if node.left is None:
            raise ValueError("When reaching the root the left branch should already exist")
        right_node = rev_lc_transform(Node(NodeInfo(NodeType.NT, "X")), ref_node.left)
        node.right = right_node
        right_node.set_parent(node)
        return node
    elif ref_node.left.node_info.type == NodeType.PT and ref_node.right.is_eps():
        # X -> word X-X
        if node.left is None:
            raise ValueError(
                "When reaching the end of the chain the left branch should already exist")
        node.right = Node(ref_node.left.node_info.copy(ref_node.left), node)
        return node
    elif ref_node.left.node_info.type == NodeType.NT and ref_node.right.node_info.type == NodeType.NT:
        # X -> X X
        right_node = rev_lc_transform(Node(NodeInfo(NodeType.NT, "X")), ref_node.left)
        node.right = right_node
        right_node.set_parent(node)
        par_node = Node(NodeInfo(NodeType.NT, "X"), None)
        par_node.set_left(node)
        return rev_lc_transform(par_node, ref_node.right)
