from node import Node, NodeType


# TODO: We should have a better abstraction for this other than string processing
# TODO: We should have a better discussion of what "ref" means. I think it's
# a many to one relationship between nodes in the transformed tree and the original

def extract_left_corner(node: Node) -> Node:
    while node.left is not None:
        node = node.left
    return node


def expand_nt(node: Node) -> None:
    leftcorner_node = extract_left_corner(node.ref)

    new_label = node.label + "-" + leftcorner_node.label
    new_right_node = Node(NodeType.NT_NT, new_label, node)

    new_left_node = Node(NodeType.PT, leftcorner_node.label, node)

    node.set_left(new_left_node)
    node.set_right(new_right_node)

    node.left.ref = leftcorner_node
    node.right.ref = leftcorner_node


def expand_nt_nt(node: Node) -> None:
    first_chunk_label = node.label.split("-")[0]
    new_label = first_chunk_label + "-" + node.ref.parent.label
    new_right_node = Node(NodeType.NT_NT, new_label, node)
    sibling_node = node.ref.parent.right

    new_left_node = Node(sibling_node.type, sibling_node.label, node)

    # modifies the left children of the input node
    node.left = new_left_node
    node.left.ref = sibling_node
    
    # modifies the right children of the input node
    node.right = new_right_node
    node.right.ref = node.ref.parent


def eps(node: Node):
    """ Predicate that returns true for nodes of the type X-X """
    return node.label.split("-")[0] == node.label.split("-")[1]


def transform(cur: Node) -> None:
    if cur.type == NodeType.NT:
        expand_nt(cur)
    elif cur.type == NodeType.NT_NT and not eps(cur):
        expand_nt_nt(cur)
    else:
        return
    transform(cur.left)
    transform(cur.right)



