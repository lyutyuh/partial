from enum import Enum

from node import Node, NodeType, NodeInfo, NodePair
from transform import LeftCornerTransformer
import logging

from visualize import print_tree


class TetraType(Enum):
    r = 0
    l = 1
    R = 2
    L = 3


class SRAction(Enum):
    S = "SHIFT"
    R = "REDUCE"


class TetraTagger:

    @classmethod
    def tree_to_tags(cls, root: Node) -> [TetraType]:
        raise NotImplementedError("tree to tags is not implemented")

    @classmethod
    def decide_tag(cls, sr_action: SRAction, node: Node) -> TetraType:
        raise NotImplementedError("decide tags is not implemented")

    @classmethod
    def tetra_visualize(cls, actions: [TetraType]):
        for a in actions:
            if a == TetraType.r:
                yield "-->"
            if a == TetraType.l:
                yield "<--"
            if a == TetraType.R:
                yield "==>"
            if a == TetraType.L:
                yield "<=="


class BottomUpTetratagger(TetraTagger):
    """ Kitaev and Klein (2020)"""

    @classmethod
    def tree_to_tags(cls, root: Node) -> []:
        tags = []
        lc = LeftCornerTransformer.extract_left_corner_no_eps(root)
        logging.debug("SHIFT {}".format(lc))
        tags.append(TetraType.r)
        stack = [lc]

        while len(stack) != 1 or stack[0].label != root.label:
            node = stack[-1]
            if not node.is_right_child() and node.parent.right is not None:
                lc = LeftCornerTransformer.extract_left_corner_no_eps(node.parent.right)
                stack.append(lc)
                logging.debug("--> \t SHIFT {}".format(lc))
                tags.append(TetraType.r)

            elif len(stack) >= 2 and node.get_sibling() == stack[-2]:
                prev_node = stack[-2]
                logging.debug("==> \t REDUCE[ {0} {1} --> {2} ]".format(
                    *(prev_node.label, node.label, node.parent.label)))

                tags.append(TetraType.R)  # normal reduce
                stack.pop()
                stack.pop()
                stack.append(node.parent)

            elif node.node_info.type != NodeType.NT_NT:
                logging.debug(
                    "<== \t REDUCE[ {0} --> {1} ]".format(*(node.label, node.parent.label)))
                tags.append(TetraType.L)  # unary reduce
                stack.pop()
                stack.append(node.parent)
            else:
                logging.error("ERROR: Undefined stack state")
                return
        logging.debug("=" * 20)
        return tags

    @classmethod
    def _unary_reduce(cls, node, last_node):
        info = NodeInfo(NodeType.NT, "X")
        l_child = NodePair(info, info, node)
        node.set_left(l_child)
        node.set_right(last_node)
        return node

    @classmethod
    def _reduce(cls, node, last_node, last_2_node):
        node.set_right(last_node)
        node.set_left(last_2_node)
        return node

    @classmethod
    def tags_to_tree(cls, tags: [TetraType], input_seq: [str]) -> Node:
        created_node_stack = []
        node = None
        for tag in tags:
            if tag == TetraType.r:  # shift
                created_node_stack.append(Node(NodeInfo(NodeType.PT, input_seq[0]), None))
                input_seq.pop(0)
            else:
                node = Node(NodeInfo(NodeType.NT, "X"), None)
                if tag == TetraType.R:  # normal reduce
                    last_node = created_node_stack.pop()
                    last_2_node = created_node_stack.pop()
                    created_node_stack.append(cls._reduce(node, last_node, last_2_node))
                elif tag == TetraType.L:  # unary reduce
                    created_node_stack.append(
                        cls._unary_reduce(node, created_node_stack.pop()))
        if len(input_seq) != 0:
            raise ValueError("All the input sequence is not used")
        return node


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

        return actions
