from node import NodePair, Node, NodeType, NodeInfo
from visualize import print_tree

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
        print_tree(node)
        print()
        return rev_rc_transform(node, ref_node.left)
    elif ref_node.right.node_info.type == NodeType.NT and ref_node.left.is_eps():
        # X -> X-X X
        if node.right is None:
            raise ValueError("When reaching the root the right branch should already exist")
        left_node = rev_rc_transform(Node(NodeInfo(NodeType.NT, "X")), ref_node.right)
        node.left = left_node
        left_node.set_parent(node)
        print_tree(node)
        print()
        return node
    elif ref_node.right.node_info.type == NodeType.PT and ref_node.left.is_eps():
        # X -> X-X word
        if node.right is None:
            raise ValueError(
                "When reaching the end of the chain the right branch should already exist")
        node.left = Node(ref_node.right.node_info.copy(ref_node.right), node)
        print_tree(node)
        print()
        return node
    elif ref_node.right.node_info.type == NodeType.NT and ref_node.left.node_info.type == NodeType.NT:
        # X -> X X
        pass