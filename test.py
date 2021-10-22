import unittest

from node import Node, NodeInfo, NodeType
from rev_transform import rev_transform
from tetratagger import TopDownTetratagger, BottomUpTetratagger, TetraType, tetra_visualize
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

    def alternate(self, actions):
        last = actions[0]
        for a in actions[1:]:
            if last == TetraType.r or last == TetraType.l:
                result = a == TetraType.R or a == TetraType.L
                self.assertEqual(result, True)
            else:
                result = a == TetraType.r or a == TetraType.l
                self.assertEqual(result, True)
            last = a

    def test_bottom_up_kitaev(self):
        print("Checking the bottom-up tetratagger on Kitaev and Klein (2020)'s Figure 1")
        root = tetratagger_example()
        rc_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)

        butt = BottomUpTetratagger()
        actions = butt.convert(rc_root)

        # the alternation test
        self.alternate(actions)

    def test_bottom_up_random(self, trials=1000):
        print("Checking the bottom-up tetratagger on {0} random trees".format(trials))

        import numpy as np
        np.random.seed(0)
        for trial in range(trials):
            #print("trial:\t{0}".format(trial))
            
            root = Node(NodeInfo(NodeType.NT, "S"), None)
            random_tree(root, depth=3, cutoff=5)
            rc_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
            RightCornerTransformer.transform(rc_root)
    
            butt = BottomUpTetratagger()
            actions = butt.convert(rc_root)

            # the alternation test
            self.alternate(actions)
    

    def test_top_down_random(self, trials=1000):
        print("Checking the top-down tetratagger on {0} random trees".format(trials))

        for trial in range(trials):
            root = Node(NodeInfo(NodeType.NT, "S"), None)
            random_tree(root, depth=3, cutoff=5)

            lc_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
            LeftCornerTransformer.transform(lc_root)

            tdtt = TopDownTetratagger()
            actions = tdtt.convert(lc_root)

            # the alternation test
            self.alternate(actions)

    def test_equiv_random(self, trials=1000):

        for trial in range(trials):
            #print("trial:\t{0}".format(trial))
            
            root = Node(NodeInfo(NodeType.NT, "S"), None)
            random_tree(root, depth=3, cutoff=5)

            rc_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
            RightCornerTransformer.transform(rc_root)
    
            lc_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
            LeftCornerTransformer.transform(lc_root)

            butt = BottomUpTetratagger()
            actions_butt = butt.convert(rc_root)

            tdtt = TopDownTetratagger()
            actions_tdtt = tdtt.convert(lc_root)

            d = {}
            for a1, a2 in zip(actions_tdtt, actions_butt):
                if (a1.name, a2.name) not in d:
                    d[(a1.name, a2.name)] = 0
                d[(a1.name, a2.name)] += 1


            print(d)


    def test_bottom_up_round_trip(self):
        pass

    def test_top_down_roundd_trip(self):
        pass


if __name__ == '__main__':
    unittest.main()

# TODO: add a pruning method for nodes of the form X-X and its inverse, which I think is possible. Also remove unnecessary unaries
# TODO: make sure that the leaf nodes stay after the transform
