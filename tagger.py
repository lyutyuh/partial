from abc import ABC

from nltk import ParentedTree as PTree
from nltk import Tree
from tqdm import tqdm as tq


class Tagger(ABC):
    def __init__(self, trees=None, add_remove_top=False):
        self.vocab = {}
        self.vocab_list = []
        self.first_unused_idx = len(self.vocab_list)
        self.add_remove_top = add_remove_top

        if trees is not None:
            for tree in tq(trees):
                self.tree_to_tags_pipeline(tree)

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

