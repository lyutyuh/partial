import numpy as np
np.random.seed(0)
from ppbtree import print_tree

from node import Node, DepNode, NodeType, NodeInfo
from transform import LeftCornerTransformer, RightCornerTransformer
#from rev_transformer import rev_transform

from tetratagger import TopDownTetratagger, BottomUpTetratagger, tetra_visualize, tetra_alternate

from tree_tools import random_tree, random_dep_tree

class ArcEager(object):


	def __init__(self):
		pass


	def convert(self, tree):
		""" convert right-corner transformed tree to shifts and reduces """
		#actions = []
		print_tree(tree)
		print()
		print()
		lc = LeftCornerTransformer.extract_left_corner_no_eps(tree)

		print()
		print("-->\tSHIFT[ {0} ]".format(lc.label))
		#actions.append(TetraType.r)
		stack = [lc]
		print(stack)
		print()

		while len(stack) != 1 or stack[0].label != "S":

			print(stack)
			node = stack[-1]
			input()
			print(node, node.parent.right)
			if node != node.parent.right and node.parent.right is not None:
				print(node)
				print("HERE")
				if node.parent.right.node_info.type == NodeType.PT:
					print("-->\tSHIFT[ {0} ]".format(node.parent.right.label))
					#actions.append(TetraType.r)
					stack.append(node.parent.right)

			elif len(stack) >= 2:
				prev_node = stack[-2]
				if prev_node.node_info.type == NodeType.NT_NT:
					print("<==\tREDUCE[ {0} {1} --> {2} ]".format(*(prev_node.label, node.label, node.parent.label)))

					#if prev_node.node_info2.label == node.label:
					#	actions.pop()
					#	actions.append(TetraType.l)
					#else:
					#	#	actions.append(TetraType.L)
					stack.pop()
					stack.pop()
					stack.append(node.parent)

			elif len(stack) == 1:
				if node.node_info.type == NodeType.PT or node.node_info.type == NodeType.NT:
					print("==>\tREDUCE[ {0} --> {1} ]".format(*(node.label, node.parent.label)))
					actions.append(TetraType.R)
					stack.pop()
					stack.append(node.parent)

			print(stack)
			print()

		return actions

root = DepNode(NodeInfo(NodeType.NT, "A"), None)

random_dep_tree(root)
print_tree(root)

print()
rc_root = Node(NodeInfo(NodeType.NT, root.label, ref=root), None)
RightCornerTransformer.transform(rc_root)
print_tree(rc_root)
exit(0)

arc_eager = ArcEager()
arc_eager.convert(rc_root)