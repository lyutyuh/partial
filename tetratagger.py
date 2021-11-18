import logging

from nltk import ParentedTree as PTree
from nltk import Tree

from transform import LeftCornerTransformer, RightCornerTransformer
from tree_tools import find_node_type, is_node_epsilon, NodeType


class TetraTagger:
    def __init__(self, add_remove_top=False):
        self.vocab = {'l': 0, 'r': 1, 'L': 2, 'R': 3}
        self.vocab_list = ['l', 'r', 'L', 'R']
        self.first_unused_idx = len(self.vocab_list)
        self.add_remove_top = add_remove_top

    def add_tags_to_vocab(self, tags: [str]):
        for tag in tags:
            if tag not in self.vocab:
                self.vocab[tag] = self.first_unused_idx
                self.vocab_list.append(tag)
                self.first_unused_idx += 1

    def tree_to_tags(self, root: PTree) -> [str]:
        raise NotImplementedError("tree to tags is not implemented")

    def tags_to_tree(self, tags: [str], input_seq: [str]) -> PTree:
        raise NotImplementedError("tags to tree is not implemented")

    def expand_tags(self, tags: [str]) -> [str]:
        raise NotImplementedError("expand tags is not implemented")

    def tree_to_tags_pipeline(self, tree: Tree) -> [str]:
        ptree = self.preprocess(tree)
        return self.tree_to_tags(ptree)

    def tree_to_ids_pipeline(self, tree: Tree) -> [int]:
        tags = self.tree_to_tags_pipeline(tree)
        return [self.vocab[tag] for tag in tags]

    def tags_to_tree_pipeline(self, tags: [str], input_seq: []) -> Tree:
        ptree = self.tags_to_tree(tags, input_seq)
        return self.postprocess(ptree)

    def ids_to_tree_pipeline(self, ids: [int], input_seq: []) -> Tree:
        tags = [self.vocab_list[idx] for idx in ids]
        return self.tags_to_tree_pipeline(tags, input_seq)

    def preprocess(self, tree: Tree) -> PTree:
        if self.add_remove_top:
            cut_off_tree = tree[0]
        else:
            cut_off_tree = tree
        cut_off_tree.collapse_unary(collapsePOS=True, collapseRoot=True)
        cut_off_tree.chomsky_normal_form()
        ptree = PTree.convert(cut_off_tree)
        return ptree

    def postprocess(self, tree: PTree) -> Tree:
        tree = Tree.convert(tree)
        tree.un_chomsky_normal_form()
        if self.add_remove_top:
            return Tree("TOP", [tree])
        else:
            return tree

    @staticmethod
    def tetra_visualize(tags: [str]):
        for tag in tags:
            if tag.startswith('r'):
                yield "-->"
            if tag.startswith('l'):
                yield "<--"
            if tag.startswith('R'):
                yield "==>"
            if tag.startswith('L'):
                yield "<=="

    @staticmethod
    def create_shift_tag(label: str, left_or_right: str) -> str:
        if label.find("+") != -1:
            return left_or_right + "/" + "/".join(label.split("+")[:-1])
        else:
            return left_or_right

    @staticmethod
    def _create_bi_reduce_tag(label: str) -> str:
        label = label.split("\\")[1]
        if label.find("|") != -1:  # drop extra node labels created after binarization
            return "R"
        else:
            return "R" + "/" + label.replace("+", "/")

    @staticmethod
    def _create_unary_reduce_tag(label: str) -> str:
        label = label.split("\\")[0]
        if label.find("|") != -1:  # drop extra node labels created after binarization
            return "L"
        else:
            return "L" + "/" + label.replace("+", "/")

    @staticmethod
    def create_merge_shift_tag(label: str, left_or_right: str) -> str:
        if label.find("/") != -1:
            return left_or_right + "/" + "/".join(label.split("/")[1:])
        else:
            return left_or_right

    @staticmethod
    def _create_pre_terminal_label(tag: str) -> str:
        idx = tag.find("/")
        if idx != -1:
            return tag[idx + 1:].replace("/", "+")
        else:
            return "X"

    @staticmethod
    def _create_unary_reduce_label(tag: str) -> str:
        idx = tag.find("/")
        if idx == -1:
            return "X|"
        return tag[idx + 1:].replace("/", "+")

    @staticmethod
    def _create_reduce_label(tag: str) -> str:
        idx = tag.find("/")
        if idx == -1:
            label = "X\\|"  # to mark the second part as an extra node created via binarizaiton
        else:
            label = "X\\" + tag[idx + 1:].replace("/", "+")
        return label

    @staticmethod
    def is_alternating(tags: [str]) -> bool:
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

    def expand_tags(self, tags: [str]) -> [str]:
        new_tags = []
        for tag in tags:
            if tag.startswith('r'):
                new_tags.append("l" + tag[1:])
                new_tags.append("R")
            else:
                new_tags.append(tag)
        return new_tags

    def preprocess(self, tree: Tree) -> PTree:
        ptree: PTree = super().preprocess(tree)
        root_label = ptree.label()
        tree_rc = PTree(root_label, [])
        RightCornerTransformer.transform(tree_rc, ptree, ptree)
        return tree_rc

    def tree_to_tags(self, root: PTree) -> []:
        tags = []
        lc = LeftCornerTransformer.extract_left_corner_no_eps(root)
        tags.append(self.create_shift_tag(lc.label(), "l"))

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
                tags.append(self.create_merge_shift_tag(last_two_tag, "r"))

            if node.left_sibling() is None and node.right_sibling() is not None:
                lc = LeftCornerTransformer.extract_left_corner_no_eps(node.right_sibling())
                stack.append(lc)
                logging.debug("<-- \t SHIFT {}".format(lc.label()))
                # normal shift
                tags.append(self.create_shift_tag(lc.label(), "l"))

            elif len(stack) >= 2 and (
                    node.right_sibling() == stack[-2] or node.left_sibling() == stack[-2]):
                prev_node = stack[-2]
                logging.debug("==> \t REDUCE[ {0} {1} --> {2} ]".format(
                    *(prev_node.label(), node.label(), node.parent().label())))

                tags.append(
                    self._create_bi_reduce_tag(prev_node.label()))  # normal reduce
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
                tags.append(self._create_unary_reduce_tag(
                    node.parent().label()))  # unary reduce
                stack.pop()
                stack.append(node.parent())
            else:
                logging.error("ERROR: Undefined stack state")
                return
        logging.debug("=" * 20)

        self.add_tags_to_vocab(tags)
        return tags

    def _unary_reduce(self, node, last_node, tag):
        label = self._create_unary_reduce_label(tag)
        node.insert(0, PTree(label + "\\" + label, ["EPS"]))
        node.insert(1, last_node)
        return node

    def _reduce(self, node, last_node, last_2_node, tag):
        label = self._create_reduce_label(tag)
        last_2_node.set_label(label)
        node.insert(0, last_2_node)
        node.insert(1, last_node)
        return node

    def postprocess(self, transformed_tree: PTree) -> Tree:
        tree = PTree("X", ["", ""])
        tree = RightCornerTransformer.rev_transform(tree, transformed_tree)
        return super().postprocess(tree)

    def tags_to_tree(self, tags: [str], input_seq: [str]) -> PTree:
        created_node_stack = []
        node = None
        expanded_tags = self.expand_tags(tags)
        if len(expanded_tags) == 1:  # base case
            assert expanded_tags[0].startswith('l')
            return PTree(input_seq[0][1], [input_seq[0][0]])
        for tag in expanded_tags:
            if tag.startswith('l'):  # shift
                created_node_stack.append(PTree(input_seq[0][1], [input_seq[0][0]]))
                input_seq.pop(0)
            else:
                node = PTree("X", [])
                if tag.startswith('R'):  # normal reduce
                    last_node = created_node_stack.pop()
                    last_2_node = created_node_stack.pop()
                    created_node_stack.append(self._reduce(node, last_node, last_2_node, tag))
                elif tag.startswith('L'):  # unary reduce
                    created_node_stack.append(
                        self._unary_reduce(node, created_node_stack.pop(), tag))
        if len(input_seq) != 0:
            raise ValueError("All the input sequence is not used")
        return node


class TopDownTetratagger(TetraTagger):

    def expand_tags(self, tags: [str]) -> [str]:
        new_tags = []
        for tag in tags:
            if tag.startswith('l'):
                new_tags.append("R")
                new_tags.append("r" + tag[1:])
            else:
                new_tags.append(tag)
        return new_tags

    def preprocess(self, tree: Tree) -> PTree:
        ptree = super(TopDownTetratagger, self).preprocess(tree)
        root_label = ptree.label()
        tree_lc = PTree(root_label, [])
        LeftCornerTransformer.transform(tree_lc, ptree, ptree)
        return tree_lc

    def tree_to_tags(self, root: PTree) -> [str]:
        """ convert left-corner transformed tree to shifts and reduces """
        stack: [PTree] = [root]
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
                        self.create_merge_shift_tag(node[0].label(), "l"))  # merged shift
                else:
                    if not is_node_epsilon(node[1]):
                        stack.append(node[1])
                        tags.append(self._create_bi_reduce_tag(node[1].label()))
                        # normal reduce
                    else:
                        tags.append(self._create_unary_reduce_tag(node[1].label()))
                        # unary reduce
                    stack.append(node[0])

            elif find_node_type(node) == NodeType.PT:
                tags.append(self.create_shift_tag(node.label(), "r"))  # normal shift
                logging.debug("-->\tSHIFT[ {0} ]".format(node.label()))
                stack.pop()

        self.add_tags_to_vocab(tags)
        return tags

    def postprocess(self, transformed_tree: PTree) -> Tree:
        ptree = PTree("X", ["", ""])
        ptree = LeftCornerTransformer.rev_transform(ptree, transformed_tree)
        return super().postprocess(ptree)

    def tags_to_tree(self, tags: [str], input_seq: [str]) -> PTree:
        expanded_tags = self.expand_tags(tags)
        root = PTree("X", [])
        created_node_stack = [root]
        if len(expanded_tags) == 1:  # base case
            assert expanded_tags[0].startswith('r')
            return PTree(input_seq[0][1], [input_seq[0][0]])
        for tag in expanded_tags:
            if tag.startswith('r'):  # shift
                node = created_node_stack.pop()
                node.set_label(input_seq[0][1])
                node.insert(0, input_seq[0][0])
                input_seq.pop(0)
            elif tag.startswith('R') or tag.startswith('L'):
                parent = created_node_stack.pop()
                if tag.startswith('R'):  # normal reduce
                    label = self._create_reduce_label(tag)
                    r_node = PTree(label, [])
                    created_node_stack.append(r_node)
                else:
                    label = self._create_unary_reduce_label(tag)
                    r_node = PTree(label + "\\" + label, ["EPS"])

                l_node_label = self._create_reduce_label(tag)
                l_node = PTree(l_node_label, [])
                created_node_stack.append(l_node)
                parent.insert(0, l_node)
                parent.insert(1, r_node)
            else:
                raise ValueError("Invalid tag type")
        if len(input_seq) != 0:
            raise ValueError("All the input sequence is not used")
        return root
