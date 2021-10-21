import numpy as np

seed = np.random.randint(1000)
print(seed)
np.random.seed(seed)
#np.random.seed(502)
np.random.seed(630)
#np.random.seed(80)
# right arc example
#np.random.seed(150)
np.random.seed(4)
# left arc example
#np.random.seed(101)
#np.random.seed(9)
#np.random.seed(671)

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
		arcs = set()
		#actions = []
		stop = tree.label
		print_tree(tree)
		#print()
		print()
		lc = LeftCornerTransformer.extract_left_corner(tree)
		stack = [lc]
		if lc.node_info.type == NodeType.PT:
			print("SHIFT({0})".format(int(lc.label.split(sep)[-1])))

		while len(stack) != 1 or stack[0].label != stop:
			#input()
			#print(stack)
			node = stack[-1]
			#if node.is_eps():
			#	print("NULL[ {0} ]".format(node))
			if node == node.parent.left and node.parent.right is not None and (node.parent.right.node_info.type == NodeType.PT):
				print("SHIFT({0})".format(node.parent.right.label.split(sep)[-1]))
				stack.append(node.parent.right)
			elif len(stack) >= 2 and stack[-2].parent == node.parent:
				prev_node = stack[-2]
				

				if prev_node.parent.node_info.type == NodeType.NT_NT:
					# relevant indices
					l1 = int(prev_node.parent.node_info1.label.split(sep)[-1])
					l2 = int(prev_node.parent.node_info2.label.split(sep)[-1])
					r1 = int(node.label.split("-")[0].split(sep)[-1])
				
					sib = int(prev_node.parent.parent.right.label.split(sep)[-1])

					#print(l1, l2, r1, sib)
					if l1 == l2:
						output = "{0} <-- {1}".format(*(r1, l1))
						print(output)
						arcs.add(output)
					else:
						output = "{0} --> {1}".format(*(r1, sib))
						print(output)
						arcs.add(output)

				elif prev_node.parent.node_info.type == NodeType.NT and prev_node.node_info.type == NodeType.NT_NT:
					# relevant indices
					l1 = int(prev_node.node_info1.label.split(sep)[-1])
					l2 = int(prev_node.node_info2.label.split(sep)[-1])
					r1 = int(node.label.split("-")[0].split(sep)[-1])
					
				else:
					l = int(prev_node.label.split(sep)[-1])
					r = int(node.label.split(sep)[-1])
					h = int(node.parent.label.split(sep)[-1])

					if h == l:
						output = "{0} --> {1}".format(*(l, r))
						print(output)
						arcs.add(output)
					else:
						output = "{0} <-- {1}".format(*(l, r))
						print(output)
						arcs.add(output)

				stack.pop(); stack.pop()
				stack.append(node.parent)

			elif len(stack) < 2 or len(stack) >= 2 and stack[-2].parent != node.parent:
				lc = LeftCornerTransformer.extract_left_corner(node.parent.right)
				stack.append(lc)
				if lc.node_info.type == NodeType.PT:
					print("SHIFT({0})".format(int(lc.label.split(sep)[-1])))
			else:
				print("BREAK")

		return arcs


trials = 1
for _ in range(trials):
	root = DepNode(NodeInfo(NodeType.NT, "A"), None)
	arcs1 = set()
	random_dep_tree(root, arcs1)
	
	print_tree(root)
	#print()

	rc_root = Node(NodeInfo(NodeType.NT, root.label, ref=root), None)
	RightCornerTransformer.transform(rc_root)
	#RightCornerTransformer.partial_transform(root)
	print_tree(rc_root)
	#exit(0)



	#print()
	#rc_root = Node(NodeInfo(NodeType.NT, root.label, ref=root), None)
	#RightCornerTransformer.transform(rc_root)


	lc_root = Node(NodeInfo(NodeType.NT, root.label, ref=root), None)
	LeftCornerTransformer.transform(lc_root)
	print_tree(lc_root)
	exit(0)
	arc_eager = ArcEager()
	arcs2 = arc_eager.convert(root)
	assert arcs1==arcs2