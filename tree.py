import numpy as np
from ppbtree import print_tree
from node import Node, NodeType, NodeInfo


def random_tree(node, idx, depth=0, p=.75, cutoff=7):
	""" sample a random tree """

	left, right = None, None
	if np.random.binomial(1,p) == 1 and depth < cutoff:
		# add the left child tree
		lidx = idx+"l"
		left = Node(NodeInfo(NodeType.NT, lidx), node)
		node.set_left(left)
		random_tree(left, lidx, depth=depth+1)
	else:
		left = Node(NodeInfo(NodeType.PT, idx+"l"), node)		
		node.set_left(left)


	if np.random.binomial(1,p) == 1 and depth < cutoff:
		# add the right child tree
		ridx = idx+"r"
		right = Node(NodeInfo(NodeType.NT, ridx), node)
		node.set_right(right)
		random_tree(right, ridx, depth=depth+1)
	else:
		right = Node(NodeInfo(NodeType.PT, idx+"r"), node)		
		node.set_right(right)


root = Node(NodeInfo(NodeType.NT, "S"), None)
random_tree(root, "S")
print_tree(root)