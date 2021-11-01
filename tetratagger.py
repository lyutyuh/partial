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
    def tags_to_tree(cls, tags: [TetraType], input_seq: [str]) -> Node:
        raise NotImplementedError("tags to tree is not implemented")

    @classmethod
    def tetra_visualize(cls, tags: [TetraType]):
        for tag in tags:
            if tag == TetraType.r:
                yield "-->"
            if tag == TetraType.l:
                yield "<--"
            if tag == TetraType.R:
                yield "==>"
            if tag == TetraType.L:
                yield "<=="

    @classmethod
    def expand_tags(cls, tags: [TetraType]) -> [TetraType]:
        new_tags = []
        for tag in tags:
            if tag == TetraType.l:
                new_tags.append(TetraType.r)
                new_tags.append(TetraType.R)
            else:
                new_tags.append(tag)
        return new_tags

    @classmethod
    def is_alternating(cls, tags: [TetraType]) -> bool:
        prev_state = True #true means reduce
        for tag in tags:
            if tag == TetraType.r or tag == TetraType.l:
                state = False
            else:
                state = True
            if state == prev_state:
                return False
            prev_state = state
        return True


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
            if node.node_info.type == NodeType.NT:  # special case: merge the reduce and last shift
                last_tag = tags.pop()
                last_two_tag = tags.pop()
                if last_tag != TetraType.R or last_two_tag != TetraType.r:
                    raise ValueError(
                        "When reaching NT the right PT should already be shifted")
                tags.append(TetraType.l)  # merged shift

            if not node.is_right_child() and node.parent.right is not None:
                lc = LeftCornerTransformer.extract_left_corner_no_eps(node.parent.right)
                stack.append(lc)
                logging.debug("--> \t SHIFT {}".format(lc))
                tags.append(TetraType.r)  # normal shift

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
        expanded_tags = cls.expand_tags(tags)
        for tag in expanded_tags:
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


class TopDownTetratagger(TetraTagger):

    @classmethod
    def tree_to_tags(cls, root: Node) -> [TetraType]:
        """ convert left-corner transformed tree to shifts and reduces """
        stack: [Node] = [root]
        logging.debug("SHIFT {}".format(root.label))
        tags = []
        while len(stack) > 0:
            node = stack[-1]

            if node.node_info.type == NodeType.NT or node.node_info.type == NodeType.NT_NT:
                stack.pop()
                logging.debug("REDUCE[ {0} --> {1} {2}]".format(
                    *(node.label, node.left.label, node.right.label)))
                if not node.right.is_eps():
                    stack.append(node.right)
                    tags.append(TetraType.R)  # normal reduce
                else:
                    tags.append(TetraType.L)  # unary reduce
                stack.append(node.left)

            elif node.node_info.type == NodeType.PT:
                tags.append(TetraType.r)  # shift
                logging.debug("-->\tSHIFT[ {0} ]".format(node.label))
                stack.pop()

        return tags

    @classmethod
    def tags_to_tree(cls, tags: [TetraType], input_seq: [str]) -> Node:
        root = Node(NodeInfo(NodeType.NT, "X"), None)
        created_node_stack = [root]
        for tag in tags:
            if tag == TetraType.r:  # shift
                node = created_node_stack.pop()
                node.update_node_info(NodeType.PT, input_seq[0])
                input_seq.pop(0)
            elif tag == TetraType.R or tag == TetraType.L:
                parent = created_node_stack.pop()
                if tag == TetraType.R:  # normal reduce
                    r_node = Node(NodeInfo(NodeType.NT, "X"), parent)
                    created_node_stack.append(r_node)
                else:
                    node_info = NodeInfo(NodeType.NT, "X")
                    r_node = NodePair(node_info, node_info, parent)

                l_node = Node(NodeInfo(NodeType.NT, "X"), parent)
                created_node_stack.append(l_node)
                parent.set_left(l_node)
                parent.set_right(r_node)

            else:
                raise ValueError("Invalid tag type")
        if len(input_seq) != 0:
            raise ValueError("All the input sequence is not used")
        return root
