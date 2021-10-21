from enum import Enum

from node import Node, NodeType
from transform import LeftCornerTransformer


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
                print("==>\tREDUCE[ {0} --> {1} {2}]".format(
                    *(node.label, node.left.label, node.right.label)))
                actions.append(TetraType.R)

                if not node.right.is_eps():
                    stack.append(node.right)
                if not node.left.is_eps():
                    stack.append(node.left)

            elif node.node_info.type == NodeType.PT:
                actions.append(TetraType.r)
                print("-->\tSHIFT[ {0} ]".format(node.label))
                stack.pop()

            elif node.node_info.type == NodeType.NT_NT:
                stack.pop()
                print("<==\tREDUCE[ {0} --> {1} {2}]".format(
                    *(node.label, node.left.label, node.right.label)))

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

        print()
        return actions


class BottomUpTetratagger(object):
    """ Kitaev and Klein (2020)"""

    def __init__(self):
        pass

    def convert(self, tree):
        """ convert right-corner transformed tree to shifts and reduces """
        actions = []
        lc = LeftCornerTransformer.extract_left_corner_no_eps(tree)

        print()
        print("-->\tSHIFT[ {0} ]".format(lc.label))
        actions.append(TetraType.r)
        stack = [lc]
        print(stack)
        print()

        while len(stack) != 1 or stack[0].label != "S":

            print(stack)
            node = stack[-1]

            if node != node.parent.right and node.parent.right is not None:
                if node.parent.right.node_info.type == NodeType.PT:
                    print("-->\tSHIFT[ {0} ]".format(node.parent.right.label))
                    actions.append(TetraType.r)
                    stack.append(node.parent.right)

            elif len(stack) >= 2:
                prev_node = stack[-2]
                if prev_node.node_info.type == NodeType.NT_NT:
                    print("<==\tREDUCE[ {0} {1} --> {2} ]".format(
                        *(prev_node.label, node.label, node.parent.label)))

                    if prev_node.node_info2.label == node.label:
                        actions.pop()
                        actions.append(TetraType.l)
                    else:
                        actions.append(TetraType.L)
                    stack.pop()
                    stack.pop()
                    stack.append(node.parent)

            elif len(stack) == 1:
                if node.node_info.type == NodeType.PT or node.node_info.type == NodeType.NT:
                    print(
                        "==>\tREDUCE[ {0} --> {1} ]".format(*(node.label, node.parent.label)))
                    actions.append(TetraType.R)
                    stack.pop()
                    stack.append(node.parent)

            print(stack)
            print()

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


def tetra_alternate(actions):
    last = actions[0]
    for a in actions[1:]:
        if last == TetraType.r or last == TetraType.l:
            assert a == TetraType.R or a == TetraType.L
        else:
            assert a == TetraType.r or a == TetraType.l
        last = a
