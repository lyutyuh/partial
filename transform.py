from node import Node, NodeInfo, NodePair, NodeType


class Transformer:
    @classmethod
    def expand_nt(cls, node: Node) -> None:
        raise NotImplementedError("expand non-terminal is not implemented")

    @classmethod
    def expand_nt_nt(cls, node: NodePair) -> None:
        raise NotImplementedError("expand paired non-terimnal is not implemented")

    @classmethod
    def transform(cls, cur: Node) -> None:
        if cur is None:
            return
        if cur.node_info.type == NodeType.NT:
            cls.expand_nt(cur)
        elif cur.node_info.type == NodeType.NT_NT:
            assert isinstance(cur, NodePair)
            if not cur.is_eps():
                cls.expand_nt_nt(cur)
        else:
            return
        cls.transform(cur.left)
        cls.transform(cur.right)


class LeftCornerTransformer(Transformer):
    @classmethod
    def extract_left_corner(cls, node: Node) -> Node:
        while node.left is not None:
            node = node.left
        return node

    @classmethod
    def extract_left_corner_no_eps(cls, node: Node) -> Node:
        while node.left is not None:
            if not node.left.is_eps():
                node = node.left
            else:
                node = node.right
        return node

    @classmethod
    def expand_nt(cls, node: Node):
        leftcorner_node = cls.extract_left_corner(node.node_info.ref)
        leftcorner_node_info = leftcorner_node.node_info.copy(leftcorner_node)

        new_right_node = NodePair(node.node_info, leftcorner_node_info, parent=node)
        new_left_node = Node(leftcorner_node_info, node)

        node.set_left(new_left_node)
        node.set_right(new_right_node)

    @classmethod
    def expand_nt_nt(cls, node: NodePair) -> None:
        parent_node = node.node_info2.ref.parent
        new_right_node = NodePair(node.node_info1, parent_node.node_info.copy(parent_node),
                                  parent=node)

        sibling_node = node.node_info2.ref.parent.right
        sibling_node_info = NodeInfo(sibling_node.node_info.type,
                                     sibling_node.node_info.label,
                                     ref=sibling_node)
        new_left_node = Node(sibling_node_info, parent=node)

        node.set_left(new_left_node)
        node.set_right(new_right_node)


class RightCornerTransformer(Transformer):
    @classmethod
    def extract_right_corner(cls, node: Node) -> Node:
        while node.right is not None:
            node = node.right
        return node

    @classmethod
    def expand_nt(cls, node: Node) -> None:
        rightcorner_node = cls.extract_right_corner(node.node_info.ref)
        rightcorner_node_info = rightcorner_node.node_info.copy(rightcorner_node)

        new_left_node = NodePair(node.node_info, rightcorner_node_info, parent=node)
        new_right_node = Node(rightcorner_node_info, node)

        node.set_left(new_left_node)
        node.set_right(new_right_node)

    @classmethod
    def expand_nt_nt(cls, node: NodePair) -> None:
        parent_node = node.node_info2.ref.parent
        new_left_node = NodePair(node.node_info1, parent_node.node_info.copy(parent_node),
                                 parent=node)

        sibling_node = node.node_info2.ref.parent.left
        sibling_node_info = NodeInfo(sibling_node.node_info.type,
                                     sibling_node.node_info.label,
                                     ref=sibling_node)
        new_right_node = Node(sibling_node_info, parent=node)

        node.set_left(new_left_node)
        node.set_right(new_right_node)


