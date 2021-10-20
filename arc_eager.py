from ppbtree import print_tree

from node import Node, DepNode, NodeType, NodeInfo
from transform import LeftCornerTransformer, RightCornerTransformer
from leftcorner_rev_transformer import rev_transform

from tetratagger import TopDownTetratagger, BottomUpTetratagger, tetra_visualize, tetra_alternate

from tree import random_tree, random_dep_tree

class ArcEager(object):


	def __init__(self):
		pass


	def arc_eager(self, tree):
		pass

	def shift_reduce(self,tree):
		stack = []

		pass


root = DepNode(NodeInfo(NodeType.NT, "A"), None)

random_dep_tree(root)
print_tree(root)