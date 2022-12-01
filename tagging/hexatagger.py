from abc import ABC

from nltk import ParentedTree as PTree
from nltk import Tree

from const import DUMMY_LABEL
from tagging.tetratagger import TopDownTetratagger
from tagging.transform import LeftCornerTransformer
from tagging.tree_tools import binarize_lex_tree, debinarize_lex_tree


class HexaTagger(TopDownTetratagger, ABC):
    def preprocess(self, original_tree: Tree) -> PTree:
        tree = original_tree.copy(deep=True)
        tree.collapse_unary(collapsePOS=True, collapseRoot=True)
        new_root = Tree(tree.label(), [])
        new_root = binarize_lex_tree(tree, new_root, "X")

        ptree = PTree.convert(new_root)
        root_label = ptree.label()
        tree_lc = PTree(root_label, [])
        LeftCornerTransformer.transform(tree_lc, ptree, ptree)
        return tree_lc

    @staticmethod
    def create_shift_tag(label: str, left_or_right: str) -> str:
        arc_label = label.split("^^^")[-1]
        return left_or_right + "/" + arc_label

    @staticmethod
    def create_merge_shift_tag(label: str, left_or_right: str) -> str:
        arc_label = label.split("^^^")[-1]
        return left_or_right + "/" + arc_label

    @staticmethod
    def _create_bi_reduce_tag(label: str, left_or_right: str) -> str:
        label = label.split("\\")[1]
        head_idx = int(label.split("^^^")[-1])
        # label = label.split("^^^")[0]
        if label.find("|") != -1:  # drop extra node labels created after binarization
            return f'{left_or_right}' + "/" + f"{DUMMY_LABEL}^^^{head_idx}"
        else:
            return f'{left_or_right}' + "/" + label.replace("+", "/")

    @staticmethod
    def _create_unary_reduce_tag(label: str, left_or_right: str) -> str:
        label = label.split("\\")[0]
        head_idx = int(label.split("^^^")[-1])
        # label = label.split("^^^")[0]
        if label.find("|") != -1:  # drop extra node labels created after binarization
            return f'{left_or_right}' + f"/{DUMMY_LABEL}^^^{head_idx}"
        else:
            return f'{left_or_right}' + "/" + label

    def tags_to_tree_pipeline(self, tags: [str], input_seq: []) -> Tree:
        ptree = self.tags_to_tree(tags, input_seq)
        return self.postprocess(ptree)

    @staticmethod
    def _create_pre_terminal_label(tag: str, default="X") -> str:
        arc_label = tag.split("/")[1]
        return f"X^^^{arc_label}"

    @staticmethod
    def _create_unary_reduce_label(tag: str) -> str:
        idx = tag.find("/")
        if idx == -1:
            return DUMMY_LABEL
        return tag[idx + 1:].replace("/", "+")

    @staticmethod
    def _create_reduce_label(tag: str) -> str:
        idx = tag.find("/")
        if idx == -1:
            label = "X\\|"  # to mark the second part as an extra node created via binarizaiton
        else:
            label = "X\\" + tag[idx + 1:].replace("/", "+")
        return label

    def postprocess(self, transformed_tree: PTree) -> Tree:
        tree = PTree("X", ["", ""])
        tree = LeftCornerTransformer.rev_transform(tree, transformed_tree)
        tree = Tree.convert(tree)
        if len(tree.leaves()) == 1:
            # edge case with one node
            return tree
        debinarized_tree = Tree(tree.label(), [])
        debinarize_lex_tree(tree, debinarized_tree)
        return debinarized_tree
