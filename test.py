import logging
import unittest

import numpy as np
from nltk import ParentedTree
from nltk import Tree

from tetratagger import BottomUpTetratagger, TopDownTetratagger
from transform import LeftCornerTransformer, RightCornerTransformer
from tree_tools import random_tree, is_topo_equal, rc_preprocess, rc_postprocess, \
    lc_postprocess, lc_preprocess

from original_tetratagger import TetraTagSequence
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from tqdm import tqdm as tq

# logging.getLogger().setLevel(logging.DEBUG)

np.random.seed(0)


class TestTransforms(unittest.TestCase):
    def test_transform(self):
        tree = ParentedTree.fromstring("(S (NP (det the) (N dog)) (VP (V ran) (Adv fast)))")
        tree.pretty_print()
        new_tree_lc = ParentedTree("S", [])
        LeftCornerTransformer.transform(new_tree_lc, tree, tree)
        new_tree_lc.pretty_print()

        new_tree_rc = ParentedTree("S", [])
        RightCornerTransformer.transform(new_tree_rc, tree, tree)
        new_tree_rc.pretty_print()

    def test_rev_rc_transform(self, trials=100):
        for _ in range(trials):
            t = ParentedTree("ROOT", [])
            random_tree(t, depth=0, cutoff=5)
            new_tree_rc = ParentedTree("S", [])
            RightCornerTransformer.transform(new_tree_rc, t, t)
            tree_back = ParentedTree("X", ["", ""])
            tree_back = RightCornerTransformer.rev_transform(tree_back, new_tree_rc)
            self.assertEqual(tree_back, t)

    def test_rev_lc_transform(self, trials=100):
        for _ in range(trials):
            t = ParentedTree("ROOT", [])
            random_tree(t, depth=0, cutoff=5)
            new_tree_lc = ParentedTree("S", [])
            LeftCornerTransformer.transform(new_tree_lc, t, t)
            tree_back = ParentedTree("X", ["", ""])
            tree_back = LeftCornerTransformer.rev_transform(tree_back, new_tree_lc)
            self.assertEqual(tree_back, t)


class TestTagging(unittest.TestCase):
    def test_buttom_up(self):
        tree = ParentedTree.fromstring("(S (NP (det the) (N dog)) (VP (V ran) (Adv fast)))")
        tree.pretty_print()
        tree_rc = ParentedTree("S", [])
        RightCornerTransformer.transform(tree_rc, tree, tree)
        tree_rc.pretty_print()
        tagger = BottomUpTetratagger()
        tags = tagger.tree_to_tags(tree_rc)

        for tag in tagger.tetra_visualize(tags):
            print(tag)
        print("--" * 20)

    def test_buttom_up_alternate(self, trials=100):
        for _ in range(trials):
            t = ParentedTree("ROOT", [])
            random_tree(t, depth=0, cutoff=5)
            t_rc = ParentedTree("S", [])
            RightCornerTransformer.transform(t_rc, t, t)
            tagger = BottomUpTetratagger()
            tags = tagger.tree_to_tags(t_rc)
            self.assertTrue(tagger.is_alternating(tags))
            self.assertTrue((2 * len(t.leaves()) - 1) == len(tags))

    def round_trip_test_buttom_up(self, trials=100):
        for _ in range(trials):
            tree = ParentedTree("ROOT", [])
            random_tree(tree, depth=0, cutoff=5)
            tree_rc = ParentedTree("S", [])
            RightCornerTransformer.transform(tree_rc, tree, tree)
            tagger = BottomUpTetratagger()
            tags = tagger.tree_to_tags(tree_rc)
            root_from_tags = tagger.tags_to_tree(tags, tree.leaves())
            tree_back = ParentedTree("X", ["", ""])
            tree_back = RightCornerTransformer.rev_transform(tree_back, root_from_tags,
                                                             pick_up_labels=False)
            self.assertTrue(is_topo_equal(tree, tree_back))

    def test_top_down(self):
        tree = ParentedTree.fromstring("(S (NP (det the) (N dog)) (VP (V ran) (Adv fast)))")
        tree.pretty_print()
        tree_lc = ParentedTree("S", [])
        LeftCornerTransformer.transform(tree_lc, tree, tree)
        tree_lc.pretty_print()
        tagger = TopDownTetratagger()
        tags = tagger.tree_to_tags(tree_lc)

        for tag in tagger.tetra_visualize(tags):
            print(tag)
        print("--" * 20)

    def test_top_down_alternate(self, trials=100):
        for _ in range(trials):
            t = ParentedTree("ROOT", [])
            random_tree(t, depth=0, cutoff=5)
            t_lc = ParentedTree("S", [])
            LeftCornerTransformer.transform(t_lc, t, t)
            tagger = TopDownTetratagger()
            tags = tagger.tree_to_tags(t_lc)
            self.assertTrue(tagger.is_alternating(tags))
            self.assertTrue((2 * len(t.leaves()) - 1) == len(tags))

    def round_trip_test_top_down(self, trials=100):
        for _ in range(trials):
            tree = ParentedTree("ROOT", [])
            random_tree(tree, depth=0, cutoff=5)
            tree_lc = ParentedTree("S", [])
            LeftCornerTransformer.transform(tree_lc, tree, tree)
            tagger = TopDownTetratagger()
            tags = tagger.tree_to_tags(tree_lc)
            root_from_tags = tagger.tags_to_tree(tags, tree.leaves())
            tree_back = ParentedTree("X", ["", ""])
            tree_back = LeftCornerTransformer.rev_transform(tree_back, root_from_tags,
                                                            pick_up_labels=False)
            self.assertTrue(is_topo_equal(tree, tree_back))


class TestPipeline(unittest.TestCase):
    def test_example_colab(self):
        example_tree = Tree.fromstring(
            "(S (NP (PRP She)) (VP (VBZ enjoys) (S (VP (VBG playing) (NP (NN tennis))))) (. .))")
        example_tree_rc = rc_preprocess(example_tree)
        example_tree_rc.pretty_print()
        tagger = BottomUpTetratagger()
        tags = tagger.tree_to_tags(example_tree_rc)
        print(tags)
        for tag in tagger.tetra_visualize(tags):
            print(tag)

    def compare_to_original_tetratagger(self):
        READER = BracketParseCorpusReader('data', ['train', 'dev', 'test'])
        trees = READER.parsed_sents('test')
        tagger = BottomUpTetratagger()
        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            original_tags = TetraTagSequence.from_tree(original_tree)
            rc_tree = rc_preprocess(tree, remove_top=True)
            tags = tagger.tree_to_tags(rc_tree)
            rc_tree_back = tagger.tags_to_tree(tags, tree.pos())
            tree_back = rc_postprocess(rc_tree_back, tree[0].label(), add_top=True)
            self.assertEqual(original_tree, tree_back)
            self.assertEqual(original_tags, tags)

    def test_example_colab_lc(self):
        example_tree = Tree.fromstring(
            "(S (NP (PRP She)) (VP (VBZ enjoys) (S (VP (VBG playing) (NP (NN tennis))))) (. .))")
        original_tree = example_tree.copy(deep=True)
        print("original tree")
        example_tree.pretty_print()
        example_tree_lc = lc_preprocess(example_tree)
        print("tree leftcornered")
        example_tree_lc.pretty_print()
        tagger = TopDownTetratagger()
        tags = tagger.tree_to_tags(example_tree_lc)
        print(tags)
        for tag in tagger.tetra_visualize(tags):
            print(tag)
        lc_tree_back = tagger.tags_to_tree(tags, example_tree.pos())
        lc_tree_back.pretty_print()
        tree_back = lc_postprocess(lc_tree_back, example_tree.label())
        tree_back.pretty_print()
        self.assertEqual(original_tree, tree_back)

    def top_down_tetratagger(self):
        READER = BracketParseCorpusReader('data', ['train', 'dev', ' test'])
        trees = READER.parsed_sents('test')
        tagger = TopDownTetratagger()
        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            lc_tree = lc_preprocess(tree, remove_top=True)
            tags = tagger.tree_to_tags(lc_tree)
            lc_tree_back = tagger.tags_to_tree(tags, tree.pos())
            tree_back = lc_postprocess(lc_tree_back, tree[0].label(), add_top=True)
            self.assertEqual(original_tree, tree_back)


if __name__ == '__main__':
    unittest.main()
