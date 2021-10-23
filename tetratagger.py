from enum import Enum

from node import Node, NodeType
from transform import LeftCornerTransformer

from visualize import print_tree


class TetraType(Enum):
    r = 0
    l = 1
    R = 2
    L = 3


class TopDownTetratagger(object):

    def __init__(self):
        pass

    def convert(self, root: Node):
        """ convert left-corner transformed tree to shifts and reduces """
        actions = []
        stack: [Node] = [root]
        while len(stack) > 0:
            node = stack[-1]

            if node.node_info.type == NodeType.NT:
                stack.pop()
                actions.append(TetraType.R)


                if node.right is not None and not node.right.is_eps():
                    stack.append(node.right)
                if node.left is not None and not node.left.is_eps():
                    stack.append(node.left)

            elif node.node_info.type == NodeType.PT:
                actions.append(TetraType.r)
                stack.pop()

            elif node.node_info.type == NodeType.NT_NT:
                stack.pop()

                # did I build a complete constituent?
                if node.left.node_info.type == NodeType.NT:
                    actions.pop()
                    actions.append(TetraType.l)
                else:
                    actions.append(TetraType.L)
                if not node.right.is_eps():
                    stack.append(node.right)
                if not node.left.is_eps():
                    stack.append(node.left)

        return actions


class BottomUpTetratagger(object):
    """ Kitaev and Klein (2020)"""

    def __init__(self):
        pass

    def convert(self, tree):
        """ convert right-corner transformed tree to shifts and reduces """
        actions = []
        lc = LeftCornerTransformer.extract_left_corner_no_eps(tree)
        actions.append(TetraType.r)
        stack = [lc]

        while len(stack) != 1 or stack[0].label != "S":


            node = stack[-1]

            if node != node.parent.right and node.parent.right is not None:
                lc = LeftCornerTransformer.extract_left_corner_no_eps(node.parent.right)
                actions.append(TetraType.r)
                stack.append(lc)

            elif len(stack) >= 2 and stack[-2].node_info.type == NodeType.NT_NT:
                prev_node = stack[-2]
                if prev_node.node_info2.label == node.label:
                    actions.pop()
                    actions.append(TetraType.l)
                else:
                    actions.append(TetraType.L)
                stack.pop()
                stack.pop()
                stack.append(node.parent)

            elif len(stack) == 1 and node.node_info.type == NodeType.PT or node.node_info.type == NodeType.NT:
                actions.append(TetraType.R)
                stack.pop()
                stack.append(node.parent)
            else:
                print("ERROR")

        return actions


def tetra_visualize(actions):
    for a in actions:
        if a == TetraType.r:
            yield "-->"
        if a == TetraType.l:
            yield "<--"
        if a == TetraType.R:
            yield "==>"
        if a == TetraType.L:
            yield "<=="



