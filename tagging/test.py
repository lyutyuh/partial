# import logging
import unittest

import numpy as np
from nltk import ParentedTree
from nltk import Tree

from tetratagger import BottomUpTetratagger, TopDownTetratagger
from tagging.srtagger import SRTaggerBottomUp, SRTaggerTopDown
from transform import LeftCornerTransformer, RightCornerTransformer
from tree_tools import random_tree, is_topo_equal

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
        tagger = BottomUpTetratagger()
        tags = tagger.tree_to_tags_pipeline(example_tree)
        print(tags)
        for tag in tagger.tetra_visualize(tags):
            print(tag)

    def test_compare_to_original_tetratagger(self):
        READER = BracketParseCorpusReader('../data', ['train', 'dev', 'test'])
        trees = READER.parsed_sents('test')
        tagger = BottomUpTetratagger(add_remove_top=True)
        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            original_tags = TetraTagSequence.from_tree(original_tree)
            tags = tagger.tree_to_tags_pipeline(tree)
            tree_back = tagger.tags_to_tree_pipeline(tags, tree.pos())
            # self.assertEqual(original_tree, tree_back)
            self.assertEqual(list(original_tags), list(tags))

    def test_example_colab_lc(self):
        example_tree = Tree.fromstring(
            "(S (NP (PRP She)) (VP (VBZ enjoys) (S (VP (VBG playing) (NP (NN tennis))))) (. .))")
        original_tree = example_tree.copy(deep=True)
        tagger = TopDownTetratagger()
        tags = tagger.tree_to_tags_pipeline(example_tree)
        tree_back = tagger.tags_to_tree_pipeline(tags, example_tree.pos())
        tree_back.pretty_print()
        self.assertEqual(original_tree, tree_back)

    def test_top_down_tetratagger(self):
        READER = BracketParseCorpusReader('../data', ['train', 'dev', ' test'])
        trees = READER.parsed_sents('test')
        tagger = TopDownTetratagger(add_remove_top=True)
        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            tags = tagger.tree_to_tags_pipeline(tree)
            tree_back = tagger.tags_to_tree_pipeline(tags, tree.pos())
            self.assertEqual(original_tree, tree_back)

    def test_tag_ids_top_down(self):
        READER = BracketParseCorpusReader('../data', ['train', 'dev', ' test'])
        trees = READER.parsed_sents('test')
        tagger = TopDownTetratagger(trees, add_remove_top=True)
        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            ids = tagger.tree_to_ids_pipeline(tree)
            tree_back = tagger.ids_to_tree_pipeline(ids, tree.pos())
            self.assertEqual(original_tree, tree_back)

    def test_tag_ids_bottom_up(self):
        READER = BracketParseCorpusReader('../data', ['train', 'dev', ' test'])
        trees = READER.parsed_sents('test')
        tagger = BottomUpTetratagger(trees, add_remove_top=True)
        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            ids = tagger.tree_to_ids_pipeline(tree)
            tree_back = tagger.ids_to_tree_pipeline(ids, tree.pos())
            self.assertEqual(original_tree, tree_back)

    def compare_top_down_bottom_up_tags(self):
        READER = BracketParseCorpusReader('../data', ['train', 'dev', ' test'])
        trees = READER.parsed_sents('test')
        tagger_bu = BottomUpTetratagger(add_remove_top=True)
        tagger_td = TopDownTetratagger(add_remove_top=True)
        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            tags_bu = tagger_bu.tree_to_tags_pipeline(tree)
            tags_td = tagger_td.tree_to_tags_pipeline(original_tree)
            print(tags_bu)
            print(tags_td)
            print("=" * 20)


class TestSRTagger(unittest.TestCase):
    def test_tag_sequence_example(self):
        READER = BracketParseCorpusReader('../data', ['train', 'dev', ' test'])
        trees = READER.parsed_sents('train')
        tagger = SRTaggerBottomUp(add_remove_top=True)
        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            tags = tagger.tree_to_tags_pipeline(tree)
            for tag in tags:
                if tag.find('/X/') != -1:
                    print(tag)
                    original_tree.pretty_print()
            tree_back = tagger.tags_to_tree_pipeline(tags, tree.pos())
            self.assertEqual(original_tree, tree_back)

    def test_example_both_version(self):
        import nltk
        example_tree = nltk.Tree.fromstring(
            "(TOP (S (NP (PRP She)) (VP (VBZ enjoys) (S (VP (VBG playing) (NP (NN tennis))))) (. .)))")
        example_tree.pretty_print()
        td_tagger = SRTaggerTopDown(add_remove_top=True)
        bu_tagger = SRTaggerBottomUp(add_remove_top=True)
        t1 = example_tree.copy(deep=True)
        t2 = example_tree.copy(deep=True)
        td_tags = td_tagger.tree_to_tags_pipeline(t1)
        bu_tags = bu_tagger.tree_to_tags_pipeline(t2)
        # self.assertEqual(set(td_tags), set(bu_tags))
        print(list(td_tags))
        print(list(bu_tags))

    def test_td_bu(self):
        READER = BracketParseCorpusReader('../data', ['train', 'dev', ' test'])
        trees = READER.parsed_sents('dev')
        td_tagger = SRTaggerTopDown(add_remove_top=True)
        bu_tagger = SRTaggerBottomUp(add_remove_top=True)

        for tree in tq(trees):
            td_tags = td_tagger.tree_to_tags_pipeline(tree)
            bu_tags = bu_tagger.tree_to_tags_pipeline(tree)
            self.assertEqual(set(td_tags), set(bu_tags))

    def test_tag_ids(self):
        READER = BracketParseCorpusReader('../data', ['train', 'dev', ' test'])
        trees = READER.parsed_sents('dev')
        tagger = SRTaggerBottomUp(READER.parsed_sents('train'), add_remove_top=True)
        for tree in tq(trees):
            ids = tagger.tree_to_ids_pipeline(tree)
            tree_back = tagger.ids_to_tree_pipeline(ids, tree.pos())
            self.assertEqual(tree, tree_back)

    def test_max_length(self):
        READER = BracketParseCorpusReader('../data', ['train', 'dev', ' test'])
        trees = READER.parsed_sents('train')
        tagger = SRTaggerBottomUp(trees, add_remove_top=True)
        print(len(tagger.tag_vocab))


if __name__ == '__main__':
    unittest.main()
