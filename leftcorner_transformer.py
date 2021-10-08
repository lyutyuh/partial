from node import Node, NodeType


def extract_left_corner(node: Node) -> Node:
    while node.left is not None:
        node = node.left
    return node


def handle_nt(node: Node, ref_node: Node) -> (Node, Node):
    leftcorner_node = extract_left_corner(ref_node)

    new_label = node.label + "-" + leftcorner_node.label
    new_right_node = Node(NodeType.NT_NT, new_label, node)

    new_left_node = Node(NodeType.PT, leftcorner_node.label, node, sibling=new_right_node)
    new_right_node.set_sibling(new_left_node)

    node.set_left(new_left_node)
    node.set_right(new_right_node)
    return leftcorner_node, leftcorner_node


def handle_nt_minus_nt(node: Node, ref_node: Node) -> (Node, Node):
    first_chunk_label = node.label.split("-")[0]
    new_label = first_chunk_label + "-" + ref_node.parent.label
    new_right_node = Node(NodeType.NT_NT, new_label, node)

    new_left_node = Node(ref_node.sibling.type, ref_node.sibling.label, node,
                         sibling=new_right_node)
    new_right_node.set_sibling(new_left_node)

    node.left = new_left_node
    node.right = new_right_node
    return ref_node.sibling, ref_node.parent


def check_epsilon(node: Node):
    return node.label.split("-")[0] == node.label.split("-")[1]


def transform(cur_node: Node, ref_node: Node) -> None:
    if cur_node.type == NodeType.NT:
        ref_node_left, ref_node_right = handle_nt(cur_node, ref_node)
    elif cur_node.type == NodeType.NT_NT and not check_epsilon(cur_node):
        ref_node_left, ref_node_right = handle_nt_minus_nt(cur_node, ref_node)
    else:
        return

    transform(cur_node.left, ref_node_left)
    transform(cur_node.right, ref_node_right)
