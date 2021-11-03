import unittest

import numpy as np
from nltk import Tree, ParentedTree

from node import Node, NodeInfo, NodeType
from rev_transform import rev_rc_transform, rev_lc_transform
from tetratagger import BottomUpTetratagger, TopDownTetratagger
from transform import LeftCornerTransformer, RightCornerTransformer
from tree_tools import example_tree_with_labels, tetratagger_example, random_tree
from visualize import print_tree

np.random.seed(0)


class TestLeftCornerTransform(unittest.TestCase):
    def test_labeled_complete_binary(self):
        print("Please check the left corner transform")
        root = example_tree_with_labels()
        print_tree(root)

        new_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
        LeftCornerTransformer.transform(new_root)
        print_tree(new_root)
        print("=" * 20)
        self.assertTrue(True)


class TestRightCornerTransform(unittest.TestCase):
    def test_labeled_complete_binary(self):
        print("Please check the right corner transform")
        root = example_tree_with_labels()
        print_tree(root)

        new_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
        RightCornerTransformer.transform(new_root)
        print_tree(new_root)
        print("=" * 20)
        self.assertTrue(True)


class TetrataggerTest(unittest.TestCase):

    def test_bottom_up_johnson(self):
        root = example_tree_with_labels()
        print_tree(root)
        print("--" * 20)
        rc_root = Node(NodeInfo(NodeType.NT, root.label, ref=root), None)
        RightCornerTransformer.transform(rc_root)
        print_tree(rc_root)
        print("--" * 20)

        tagger = BottomUpTetratagger()
        tags = tagger.tree_to_tags(rc_root)

        for tag in tagger.tetra_visualize(tags):
            print(tag)
        print("--" * 20)

        root_from_tags = tagger.tags_to_tree(tags,
                                             ["Det(the)", "N(dog)", "V(ran)", "Adv(fast)"])
        print_tree(root_from_tags)
        print("--" * 20)
        original_tree_root_back = Node(NodeInfo(NodeType.NT, "X"))
        original_tree_root_back = rev_rc_transform(original_tree_root_back, root_from_tags)
        print_tree(original_tree_root_back)
        print("=" * 20)
        self.assertTrue(Node.is_topo_eq(root, original_tree_root_back))
        self.assertFalse(Node.is_topo_eq(root, rc_root))

    def test_bottom_up_kitaev(self):
        print("Checking the bottom-up tetratagger on Kitaev and Klein (2020)'s Figure 1")
        root = tetratagger_example()
        print_tree(root)
        print("--" * 20)
        rc_root = Node(NodeInfo(NodeType.NT, root.label, ref=root), None)
        RightCornerTransformer.transform(rc_root)
        print_tree(rc_root)
        print("--" * 20)

        tagger = BottomUpTetratagger()
        tags = tagger.tree_to_tags(rc_root)

        for tag in tagger.tetra_visualize(tags):
            print(tag)
        print("--" * 20)

        root_from_tags = tagger.tags_to_tree(tags, ["A", "B", "C", "D", "E"])
        print_tree(root_from_tags)
        print("--" * 20)
        original_tree_root_back = Node(NodeInfo(NodeType.NT, "X"))
        rev_rc_transform(original_tree_root_back, root_from_tags)
        print_tree(original_tree_root_back)
        print("=" * 20)
        self.assertTrue(Node.is_topo_eq(root, original_tree_root_back))
        self.assertFalse(Node.is_topo_eq(root, rc_root))

    def test_top_down_kitaev(self):
        print("Checking the top-down tetratagger on Kitaev and Klein (2020)'s Figure 1")
        root = tetratagger_example()
        print_tree(root)
        print("--" * 20)
        lc_root = Node(NodeInfo(NodeType.NT, root.label, ref=root), None)
        LeftCornerTransformer.transform(lc_root)
        print_tree(lc_root)
        print("--" * 20)

        tagger = TopDownTetratagger()
        tags = tagger.tree_to_tags(lc_root)

        for tag in tagger.tetra_visualize(tags):
            print(tag)
        print("--" * 20)

        root_from_tags = tagger.tags_to_tree(tags, ["A", "B", "C", "D", "E"])
        print_tree(root_from_tags)
        print("--" * 20)
        original_tree_root_back = Node(NodeInfo(NodeType.NT, "X"))
        original_tree_root_back = rev_lc_transform(original_tree_root_back, root_from_tags)
        print_tree(original_tree_root_back)
        print("=" * 20)
        self.assertTrue(Node.is_topo_eq(root, original_tree_root_back))
        self.assertFalse(Node.is_topo_eq(root, lc_root))

    def test_bottom_up_round_trip(self, trials=1000):
        print("Checking the bottom-up tetratagger on {0} random trees".format(trials))

        for trial in range(trials):
            root = Node(NodeInfo(NodeType.NT, "ROOT"), None)
            input_str = []
            random_tree(root, input_str, depth=0, cutoff=2)
            # print_tree(root)

            rc_root = Node(NodeInfo(NodeType.NT, "ROOT", ref=root), None)
            RightCornerTransformer.transform(rc_root)
            # print_tree(rc_root)

            tagger = BottomUpTetratagger()
            tags = tagger.tree_to_tags(rc_root)
            root_from_tags = tagger.tags_to_tree(tags, input_str)
            # print_tree(root_from_tags)

            original_tree_root_back = Node(NodeInfo(NodeType.NT, "X"))
            original_tree_root_back = rev_rc_transform(original_tree_root_back,
                                                       root_from_tags)
            # print_tree(original_tree_root_back)

            self.assertTrue(Node.is_topo_eq(root, original_tree_root_back))

    def test_top_down_round_trip(self, trials=1000):
        print("Checking the top-down tetratagger on {0} random trees".format(trials))

        for trial in range(trials):
            root = Node(NodeInfo(NodeType.NT, "ROOT"), None)
            input_str = []
            random_tree(root, input_str, depth=0, cutoff=2)
            # print_tree(root)

            lc_root = Node(NodeInfo(NodeType.NT, "ROOT", ref=root), None)
            LeftCornerTransformer.transform(lc_root)
            # print_tree(rc_root)

            tagger = TopDownTetratagger()
            tags = tagger.tree_to_tags(lc_root)
            root_from_tags = tagger.tags_to_tree(tags, input_str)
            # print_tree(root_from_tags)

            original_tree_root_back = Node(NodeInfo(NodeType.NT, "X"))
            original_tree_root_back = rev_lc_transform(original_tree_root_back,
                                                       root_from_tags)
            # print_tree(original_tree_root_back)

            self.assertTrue(Node.is_topo_eq(root, original_tree_root_back))

    def test_alternation_and_length(self, trials=100):
        for trial in range(trials):
            root = Node(NodeInfo(NodeType.NT, "ROOT"), None)
            input_str = []
            random_tree(root, input_str, depth=0, cutoff=2)
            n = len(input_str)

            rc_root = Node(NodeInfo(NodeType.NT, "ROOT", ref=root), None)
            lc_root = Node(NodeInfo(NodeType.NT, "ROOT", ref=root), None)
            RightCornerTransformer.transform(rc_root)
            LeftCornerTransformer.transform(lc_root)

            tagger_bu = BottomUpTetratagger()
            tagger_td = TopDownTetratagger()
            tags_bu = tagger_bu.tree_to_tags(rc_root)
            tags_td = tagger_td.tree_to_tags(lc_root)

            self.assertTrue(tagger_bu.is_alternating(tags_bu))
            self.assertTrue(tagger_td.is_alternating(tags_td))
            self.assertTrue(2*n == len(tags_bu))
            self.assertTrue((2*n-1) == len(tags_td))

class TestNLTKTree(unittest.TestCase):
    def test_string_format(self):
        root = Node(NodeInfo(NodeType.NT, "ROOT"), None)
        input_str = []
        random_tree(root, input_str, depth=0, cutoff=5)
        print_tree(root)
        print(str(root))
        t = Tree.fromstring(str(root))
        t.pretty_print()

    def test_nltk_lc(self):
        tree = ParentedTree.fromstring("(S (NP (det the) (N dog)) (VP (V ran) (Adv fast)))")
        tree.pretty_print()
        new_tree_lc = ParentedTree("S", [])
        LeftCornerTransformer.transform(new_tree_lc, tree, tree)
        new_tree_lc.pretty_print()

        new_tree_rc = ParentedTree("S", [])
        RightCornerTransformer.transform(new_tree_rc, tree, tree)
        new_tree_rc.pretty_print()



if __name__ == '__main__':
    unittest.main()

