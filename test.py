import unittest

from node import Node, NodeInfo, NodeType
from rev_transform import rev_transform
from tetratagger import TopDownTetratagger, BottomUpTetratagger, tetra_visualize
from transform import LeftCornerTransformer, RightCornerTransformer
from tree_tools import example_tree_with_labels, is_equal, tetratagger_example, random_tree
from visualize import print_tree


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

    def test_labeled_complete_binary_reverse(self):
        root = example_tree_with_labels()
        transformed_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)

        LeftCornerTransformer.transform(transformed_root)
        rev_transform(transformed_root)
        rev_transformed_root = rev_transform(transformed_root)
        while rev_transformed_root.parent is not None:
            rev_transformed_root = rev_transformed_root.parent
        self.assertTrue(is_equal(root, rev_transformed_root))


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
    def test_bottom_up(self):
        print("Please check the bottom up parse")
        root = tetratagger_example()
        rc_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
        RightCornerTransformer.transform(rc_root)
        print_tree(rc_root)

        butt = BottomUpTetratagger()
        actions = butt.convert(rc_root)
        for a in tetra_visualize(actions):
            print(a)
        print("=" * 20)

    def test_top_down(self):
        # TODO: set random seed to debug
        #import numpy as np
        #np.random.seed(4)
        print("Please check the top down parse")
        root = Node(NodeInfo(NodeType.NT, "S"), None)
        random_tree(root, depth=3, cutoff=5)
        print("Original Tree")
        print_tree(root)
        new_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
        LeftCornerTransformer.transform(new_root)
        print("LC-Transformed Tree")
        print_tree(new_root)

        tdtt = TopDownTetratagger()
        actions = tdtt.convert(new_root)

        for a in tetra_visualize(actions):
            print(a)
        print("=" * 20)


if __name__ == '__main__':
    unittest.main()

# TODO: add a pruning method for nodes of the form X-X and its inverse, which I think is possible. Also remove unnecessary unaries
# TODO: make sure that the leaf nodes stay after the transform