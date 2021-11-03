from node import Node, NodePair, NodeType
from nltk import ParentedTree as Tree


def find_type(node: Tree):
    if len(node) == 1:
        return NodeType.PT
    elif node.label().find("\\") != -1:
        return NodeType.NT_NT
    else:
        return NodeType.NT


class Transformer:
    @classmethod
    def expand_nt(cls, node: Tree, ref_node: Tree) -> (Tree, Tree, Tree, Tree):
        raise NotImplementedError("expand non-terminal is not implemented")

    @classmethod
    def expand_nt_nt(cls, node: Tree, ref_node1: Tree, ref_node2: Tree) -> (Tree, Tree, Tree, Tree):
        raise NotImplementedError("expand paired non-terimnal is not implemented")

    @classmethod
    def extract_right_corner(cls, node: Tree) -> Tree:
        while len(node) > 1:
            node = node[1]
        return node

    @classmethod
    def extract_left_corner(cls, node: Tree) -> Tree:
        while len(node) > 1:
            node = node[0]
        return node

    @classmethod
    def transform(cls, node: Tree, ref_node1: Tree, ref_node2: Tree) -> None:
        if node is None:
            return
        type = find_type(node)
        if type == NodeType.NT:
            left_ref1, left_ref2, right_ref1, right_ref2 = cls.expand_nt(node, ref_node1)
        elif type == NodeType.NT_NT:
            left_ref1, left_ref2, right_ref1, right_ref2 = cls.expand_nt_nt(node, ref_node1, ref_node2)
        else:
            return
        cls.transform(node[0], left_ref1, left_ref2)
        cls.transform(node[1], right_ref1, right_ref2)


class LeftCornerTransformer(Transformer):

    @classmethod
    def extract_left_corner_no_eps(cls, node: Tree) -> Tree:
        while len(node) > 1:
            if not node.left.is_eps():
                node = node[0]
            else:
                node = node[1]
        return node

    @classmethod
    def expand_nt(cls, node: Tree, ref_node: Tree) -> (Tree, Tree, Tree, Tree):
        leftcorner_node = cls.extract_left_corner(ref_node)
        new_right_node = Tree(node.label() + "\\" + leftcorner_node.label(), [])
        new_left_node = Tree(leftcorner_node.label(), leftcorner_node.leaves())

        node.insert(0, new_left_node)
        node.insert(1, new_right_node)
        return leftcorner_node, leftcorner_node, ref_node, leftcorner_node

    @classmethod
    def expand_nt_nt(cls, node: Tree, ref_node1: Tree, ref_node2: Tree) -> (Tree, Tree, Tree, Tree):
        parent_node = ref_node2.parent()
        if ref_node1 == parent_node:
            new_right_node = Tree(node.label().split("\\")[0] + "\\" + parent_node.label(), ["EPS"])
        else:
            new_right_node = Tree(node.label().split("\\")[0] + "\\" + parent_node.label(), [])

        sibling_node = ref_node2.right_sibling()
        if len(sibling_node) == 1:
            new_left_node = Tree(sibling_node.label(), sibling_node.leaves())
        else:
            new_left_node = Tree(sibling_node.label(), [])

        node.insert(0, new_left_node)
        node.insert(1, new_right_node)
        return sibling_node, sibling_node, ref_node1, parent_node


class RightCornerTransformer(Transformer):

    @classmethod
    def expand_nt(cls, node: Tree, ref_node: Tree) -> (Tree, Tree, Tree, Tree):
        rightcorner_node = cls.extract_right_corner(ref_node)
        new_left_node = Tree(node.label() + "\\" + rightcorner_node.label(), [])
        new_right_node = Tree(rightcorner_node.label(), rightcorner_node.leaves())

        node.insert(0, new_left_node)
        node.insert(1, new_right_node)
        return ref_node, rightcorner_node, rightcorner_node, rightcorner_node

    @classmethod
    def expand_nt_nt(cls, node: Tree, ref_node1: Tree, ref_node2: Tree) -> (Tree, Tree, Tree, Tree):
        parent_node = ref_node2.parent()
        if ref_node1 == parent_node:
            new_left_node = Tree(node.label().split("\\")[0] + "\\" + parent_node.label(), ["EPS"])
        else:
            new_left_node = Tree(node.label().split("\\")[0] + "\\" + parent_node.label(), [])

        sibling_node = ref_node2.left_sibling()
        if len(sibling_node) == 1:
            new_right_node = Tree(sibling_node.label(), sibling_node.leaves())
        else:
            new_right_node = Tree(sibling_node.label(), [])

        node.insert(0, new_left_node)
        node.insert(1, new_right_node)
        return ref_node1, parent_node, sibling_node, sibling_node
