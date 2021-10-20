import numpy as np
#np.random.seed(3)
from visualize import print_tree

from node import Node, DepNode, NodeType, NodeInfo
from transform import LeftCornerTransformer, RightCornerTransformer
#from rev_transformer import rev_transform

from tetratagger import TopDownTetratagger, BottomUpTetratagger, tetra_visualize, tetra_alternate

from tree_tools import random_tree, random_dep_tree

class ArcEager(object):


	def __init__(self):
		pass


	def convert(self, tree, sep="/"):
		""" convert right-corner transformed tree to shifts and reduces """
		#actions = []
		stop = tree.label
		print_tree(tree)
		print()
		print()
		lc = LeftCornerTransformer.extract_left_corner_no_eps(tree)

		print()
		#print("-->\tSHIFT[ {0} ]".format(lc.label))
		print("SHIFT({0})".format(lc.label.split(sep)[-1]))

		#actions.append(TetraType.r)
		stack = [lc]
		#print(stack)
		#print()

		while len(stack) != 1 or stack[0].label != stop:
			#input()
			print(stack)
			node = stack[-1]

			if node != node.parent.right and node.parent.right is not None and node.parent.right.node_info.type == NodeType.PT:
				print("SHIFT({0})".format(node.parent.right.label.split(sep)[-1]))

				stack.append(node.parent.right)

			elif len(stack) >= 2 and stack[-2].node_info.type == NodeType.NT_NT:
				prev_node = stack[-2]
				if prev_node.label.split("-")[0] == node.parent.label:
					print("REDUCE")
				else:
					two = int(node.parent.label.split("-")[0].split(sep)[-1])
					three = int(node.parent.label.split("-")[1].split(sep)[-1])
					print("{0} --> {1}".format(*(two, three)))

				stack.pop()
				stack.pop()
				stack.append(node.parent)

			elif len(stack) == 1 and (node.node_info.type == NodeType.PT or node.node_info.type == NodeType.NT):
				one = int(node.label.split(sep)[-1])
				two = int(node.parent.label.split("-")[0].split(sep)[-1])
				three = int(node.parent.label.split("-")[1].split(sep)[-1])
				#(one, two, three)
				if one == two:
					print("{0} --> {1}".format(*(two, three)))
				else:
					print("{0} <-- {1}".format(*(one, two)))

				stack.pop()
				stack.append(node.parent)
			elif node.parent is not None and node.parent.left == node:
				lc = LeftCornerTransformer.extract_left_corner_no_eps(node.parent.right)
				print("SHIFT({0})".format(lc.label.split(sep)[-1]))

				stack.append(lc)
			print(stack)
	
		#return actions

root = DepNode(NodeInfo(NodeType.NT, "A"), None)

random_dep_tree(root)
print_tree(root)

print()
rc_root = Node(NodeInfo(NodeType.NT, root.label, ref=root), None)
RightCornerTransformer.transform(rc_root)
print_tree(rc_root)

arc_eager = ArcEager()
arc_eager.convert(rc_root)