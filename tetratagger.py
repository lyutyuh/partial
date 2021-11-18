import logging

from nltk import ParentedTree as Tree

from transform import LeftCornerTransformer
from tree_tools import find_node_type, is_node_epsilon, NodeType


class TetraTagger:
    def __init__(self, trees: [Tree] = None):
        if not trees:
            self.vocab = {'l', 'r', 'L', 'R'}
        else:
            tag_vocab = set()
            for tree in trees:
                for tag in self.tree_to_tags(tree):
                    tag_vocab.add(tag)
            self.tag_vocab = sorted(tag_vocab)

    @classmethod
    def tree_to_tags(cls, root: Tree) -> [str]:
        raise NotImplementedError("tree to tags is not implemented")

    @classmethod
    def tags_to_tree(cls, tags: [str], input_seq: [str]) -> Tree:
        raise NotImplementedError("tags to tree is not implemented")

    @classmethod
    def expand_tags(cls, tags: [str]) -> [str]:
        raise NotImplementedError("expand tags is not implemented")

    @classmethod
    def tetra_visualize(cls, tags: [str]):
        for tag in tags:
            if tag.startswith('r'):
                yield "-->"
            if tag.startswith('l'):
                yield "<--"
            if tag.startswith('R'):
                yield "==>"
            if tag.startswith('L'):
                yield "<=="

    @classmethod
    def create_shift_tag(cls, label, left_or_right):
        if label.find("+") != -1:
            return left_or_right + "/" + "/".join(label.split("+")[:-1])
        else:
            return left_or_right

    @classmethod
    def _create_bi_reduce_tag(cls, label):
        label = label.split("\\")[1]
        if label.find("|") != -1:  # drop extra node labels created after binarization
            return "R"
        else:
            return "R" + "/" + label.replace("+", "/")

    @classmethod
    def _create_unary_reduce_tag(cls, label):
        label = label.split("\\")[0]
        if label.find("|") != -1:  # drop extra node labels created after binarization
            return "L"
        else:
            return "L" + "/" + label.replace("+", "/")

    @classmethod
    def create_merge_shift_tag(cls, label, left_or_right):
        if label.find("/") != -1:
            return left_or_right + "/" + "/".join(label.split("/")[1:])
        else:
            return left_or_right

    @classmethod
    def _create_pre_terminal_labels(cls, tag: str) -> str:
        idx = tag.find("/")
        if idx != -1:
            return tag[idx + 1:].replace("/", "+")
        else:
            return "X"

    @classmethod
    def _create_unary_reduce_labels(cls, tag: str) -> str:
        idx = tag.find("/")
        if idx == -1:
            return "X|"
        return tag[idx + 1:].replace("/", "+")

    @classmethod
    def _create_reduce_labels(cls, tag: str) -> str:
        idx = tag.find("/")
        if idx == -1:
            label = "X\\|"  # to mark the second part as an extra node created via binarizaiton
        else:
            label = "X\\" + tag[idx + 1:].replace("/", "+")
        return label

    @classmethod
    def is_alternating(cls, tags: [str]) -> bool:
        prev_state = True  # true means reduce
        for tag in tags:
            if tag.startswith('r') or tag.startswith('l'):
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
    def expand_tags(cls, tags: [str]) -> [str]:
        new_tags = []
        for tag in tags:
            if tag.startswith('r'):
                new_tags.append("l" + tag[1:])
                new_tags.append("R")
            else:
                new_tags.append(tag)
        return new_tags

    @classmethod
    def tree_to_tags(cls, root: Tree) -> []:
        tags = []
        lc = LeftCornerTransformer.extract_left_corner_no_eps(root)
        tags.append(cls.create_shift_tag(lc.label(), "l"))

        logging.debug("SHIFT {}".format(lc.label()))
        stack = [lc]

        while len(stack) > 0:
            node = stack[-1]
            if find_node_type(
                    node) == NodeType.NT:  # special case: merge the reduce and last shift
                last_tag = tags.pop()
                last_two_tag = tags.pop()
                if not last_tag.startswith('R') or not last_two_tag.startswith('l'):
                    raise ValueError(
                        "When reaching NT the right PT should already be shifted")
                # merged shift
                tags.append(cls.create_merge_shift_tag(last_two_tag, "r"))

            if node.left_sibling() is None and node.right_sibling() is not None:
                lc = LeftCornerTransformer.extract_left_corner_no_eps(node.right_sibling())
                stack.append(lc)
                logging.debug("<-- \t SHIFT {}".format(lc.label()))
                # normal shift
                tags.append(cls.create_shift_tag(lc.label(), "l"))

            elif len(stack) >= 2 and (
                    node.right_sibling() == stack[-2] or node.left_sibling() == stack[-2]):
                prev_node = stack[-2]
                logging.debug("==> \t REDUCE[ {0} {1} --> {2} ]".format(
                    *(prev_node.label(), node.label(), node.parent().label())))

                tags.append(
                    cls._create_bi_reduce_tag(prev_node.label()))  # normal reduce
                stack.pop()
                stack.pop()
                stack.append(node.parent())

            elif find_node_type(node) != NodeType.NT_NT:
                if stack[0].parent() is None and len(stack) == 1:
                    stack.pop()
                    continue
                logging.debug(
                    "<== \t REDUCE[ {0} --> {1} ]".format(
                        *(node.label(), node.parent().label())))
                tags.append(cls._create_unary_reduce_tag(
                    node.parent().label()))  # unary reduce
                stack.pop()
                stack.append(node.parent())
            else:
                logging.error("ERROR: Undefined stack state")
                return
        logging.debug("=" * 20)
        return tags

    @classmethod
    def _unary_reduce(cls, node, last_node, tag):
        label = cls._create_unary_reduce_labels(tag)
        node.insert(0, Tree(label + "\\" + label, ["EPS"]))
        node.insert(1, last_node)
        return node

    @classmethod
    def _reduce(cls, node, last_node, last_2_node, tag):
        label = cls._create_reduce_labels(tag)
        last_2_node.set_label(label)
        node.insert(0, last_2_node)
        node.insert(1, last_node)
        return node

    @classmethod
    def tags_to_tree(cls, tags: [str], input_seq: [str]) -> Tree:
        created_node_stack = []
        node = None
        expanded_tags = cls.expand_tags(tags)
        if len(expanded_tags) == 1:  # base case
            assert expanded_tags[0].startswith('l')
            return Tree(input_seq[0][1], [input_seq[0][0]])
        for tag in expanded_tags:
            if tag.startswith('l'):  # shift
                created_node_stack.append(Tree(input_seq[0][1], [input_seq[0][0]]))
                input_seq.pop(0)
            else:
                node = Tree("X", [])
                if tag.startswith('R'):  # normal reduce
                    last_node = created_node_stack.pop()
                    last_2_node = created_node_stack.pop()
                    created_node_stack.append(cls._reduce(node, last_node, last_2_node, tag))
                elif tag.startswith('L'):  # unary reduce
                    created_node_stack.append(
                        cls._unary_reduce(node, created_node_stack.pop(), tag))
        if len(input_seq) != 0:
            raise ValueError("All the input sequence is not used")
        return node


class TopDownTetratagger(TetraTagger):
    @classmethod
    def expand_tags(cls, tags: [str]) -> [str]:
        new_tags = []
        for tag in tags:
            if tag.startswith('l'):
                new_tags.append("R")
                new_tags.append("r" + tag[1:])
            else:
                new_tags.append(tag)
        return new_tags

    @classmethod
    def tree_to_tags(cls, root: Tree) -> [str]:
        """ convert left-corner transformed tree to shifts and reduces """
        stack: [Tree] = [root]
        logging.debug("SHIFT {}".format(root.label()))
        tags = []
        while len(stack) > 0:
            node = stack[-1]
            if find_node_type(node) == NodeType.NT or find_node_type(node) == NodeType.NT_NT:
                stack.pop()
                logging.debug("REDUCE[ {0} --> {1} {2}]".format(
                    *(node.label(), node[0].label(), node[1].label())))
                if find_node_type(node) == NodeType.NT:
                    if find_node_type(node[0]) != NodeType.PT:
                        raise ValueError("Left child of NT should be a PT")
                    stack.append(node[1])
                    tags.append(
                        cls.create_merge_shift_tag(node[0].label(), "l"))  # merged shift
                else:
                    if not is_node_epsilon(node[1]):
                        stack.append(node[1])
                        tags.append(cls._create_bi_reduce_tag(node[1].label()))
                        # normal reduce
                    else:
                        tags.append(cls._create_unary_reduce_tag(node[1].label()))
                        # unary reduce
                    stack.append(node[0])

            elif find_node_type(node) == NodeType.PT:
                tags.append(cls.create_shift_tag(node.label(), "r"))  # normal shift
                logging.debug("-->\tSHIFT[ {0} ]".format(node.label()))
                stack.pop()

        return tags

    @classmethod
    def tags_to_tree(cls, tags: [str], input_seq: [str]) -> Tree:
        expanded_tags = cls.expand_tags(tags)
        root = Tree("X", [])
        created_node_stack = [root]
        if len(expanded_tags) == 1:  # base case
            assert expanded_tags[0].startswith('r')
            return Tree(input_seq[0][1], [input_seq[0][0]])
        for tag in expanded_tags:
            if tag.startswith('r'):  # shift
                node = created_node_stack.pop()
                node.set_label(input_seq[0][1])
                node.insert(0, input_seq[0][0])
                input_seq.pop(0)
            elif tag.startswith('R') or tag.startswith('L'):
                parent = created_node_stack.pop()
                if tag.startswith('R'):  # normal reduce
                    label = cls._create_reduce_labels(tag)
                    r_node = Tree(label, [])
                    created_node_stack.append(r_node)
                else:
                    label = cls._create_unary_reduce_labels(tag)
                    r_node = Tree(label + "\\" + label, ["EPS"])

                simple_label = cls._create_reduce_labels(tag)
                l_node = Tree(simple_label, [])
                created_node_stack.append(l_node)
                parent.insert(0, l_node)
                parent.insert(1, r_node)
            else:
                raise ValueError("Invalid tag type")
        if len(input_seq) != 0:
            raise ValueError("All the input sequence is not used")
        return root
