import unittest
from tree_tools import example_tree_with_labels, is_equal, tetratagger_example
from node import Node, NodeInfo, NodeType
from transform import LeftCornerTransformer, RightCornerTransformer
from rev_transform import rev_transform
from visualize import print_tree
from tetratagger import TopDownTetratagger, BottomUpTetratagger, tetra_visualize, tetra_alternate



class TestLeftCornerTransform(unittest.TestCase):
    def test_labeled_complete_binary(self):
        print("Please check the left corner transform")
        root = example_tree_with_labels()
        print_tree(root)

        new_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
        LeftCornerTransformer.transform(new_root)
        print_tree(new_root)
        print("="*20)
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
        print("="*20)
        self.assertTrue(True)

class TetrataggerTest(unittest.TestCase):
    def test_bottom_up(self):
        root = tetratagger_example()
        rc_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
        RightCornerTransformer.transform(rc_root)
        print_tree(rc_root)

        butt = BottomUpTetratagger()
        actions = butt.convert(rc_root)
        for a in tetra_visualize(actions):
            print(a)
        print("="*20)

    def test_top_down(self):
        pass

if __name__ == '__main__':
    unittest.main()


# TODO: add a pruning method for nodes of the form X-X and its inverse, which I think is possible. Also remove unnecessary unaries
# TODO: fix tree visualization
# TODO: make sure that the leaf nodes stay after the transform
# TODO: unify the left- and right-corner transform code


# random_tree(root, "S")
# new_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
# LeftCornerTransformer.transform(new_root)
#
# print();print()
# tdtt = TopDownTetratagger()
# actions = tdtt.convert(new_root)
#
# for a in tetra_visualize(actions):
#     print(a)
#
# exit(0)
#
# butt = BottomUpTetratagger()
# actions = butt.convert(rc_root)
# for a in tetra_visualize(actions):
#     print(a)
# tetra_alternate(actions)
# exit(0)




