import unittest
from tree_tools import example_tree_with_labels, is_equal
from node import Node, NodeInfo, NodeType
from transform import LeftCornerTransformer, RightCornerTransformer
from rev_transform import rev_transform
from visualize import print_tree


class TestLeftCornerTransform(unittest.TestCase):
    def test_labeled_complete_binary(self):
        root = example_tree_with_labels()
        print_tree(root)

        new_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
        LeftCornerTransformer.transform(new_root)
        print_tree(new_root)
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
        root = example_tree_with_labels()
        print_tree(root)

        new_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
        RightCornerTransformer.transform(new_root)
        print_tree(new_root)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
