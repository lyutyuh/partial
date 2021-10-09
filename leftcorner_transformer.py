from node import Node, NodeType


def extract_left_corner(node: Node) -> Node:
    while node.left is not None:
        node = node.left
    return node


def original(node: Node) -> None:
    leftcorner_node = extract_left_corner(node.ref)

    new_label = node.label + "-" + leftcorner_node.label
    new_right_node = Node(NodeType.NT_NT, new_label, node)

    new_left_node = Node(NodeType.PT, leftcorner_node.label, node, sibling=new_right_node)
    new_right_node.set_sibling(new_left_node)

    node.set_left(new_left_node)
    node.set_right(new_right_node)

    node.left.ref = leftcorner_node
    node.right.ref = leftcorner_node


def slashed(node: Node) -> None:
    # TODO: We should have a better abstraction for this other than string processing
    first_chunk_label = node.label.split("-")[0]
    new_label = first_chunk_label + "-" + node.ref.parent.label
    new_right_node = Node(NodeType.NT_NT, new_label, node)

    new_left_node = Node(node.ref.sibling.type, node.ref.sibling.label, node,
                         sibling=new_right_node)
    new_right_node.set_sibling(new_left_node)

    # modifies the left children of the input node in place
    node.left = new_left_node
    node.left.ref = node.ref.sibling
    
    # modifies the right children of the input node in place
    node.right = new_right_node
    node.right.ref = node.ref.parent


def eps(node: Node):
    """ Predicate that returns true for nodes of the type X-X """
    return node.label.split("-")[0] == node.label.split("-")[1]


def transform(node: Node) -> None:
    stack = [node]
    while len(stack) > 0:
        cur = stack.pop()

        # in-place transformation
        if cur.type == NodeType.NT:
            original(cur)
        elif cur.type == NodeType.NT_NT and not eps(cur):
            slashed(cur)

        # push children to the stack
        if cur.left is not None:
            stack.append(cur.left)
        if cur.right is not None:
            stack.append(cur.right)

