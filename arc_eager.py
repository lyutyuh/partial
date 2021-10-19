from ppbtree import print_tree

from node import Node, NodeType, NodeInfo
from leftcorner_transformer import left_corner_transform, right_corner_transform


class ArcEager(object):


	def __init__(self):
		pass


	def arc_eager(self, tree):
		pass

	def shift_reduce(self,tree):
		stack = []

		pass


root = Node(NodeInfo(NodeType.NT, "Z_c"), None)
Y_b = Node(NodeInfo(NodeType.NT, "Y_b"), root)
Y_c = Node(NodeInfo(NodeType.NT, "Y_c"), root)
root.set_left(Y_c)
root.set_right(Y_b)

X_a = Node(NodeInfo(NodeType.PT, "X_a"), Y_b)
X_b = Node(NodeInfo(NodeType.PT, "X_b"), Y_b)
Y_b.set_left(X_b)
Y_b.set_right(X_a)

#a = Node(NodeInfo(NodeType.PT, "a"), X_a)
#X_a.set_right(a)
#b = Node(NodeInfo(NodeType.PT, "b"), X_b)
#X_b.set_right(b)

X_c = Node(NodeInfo(NodeType.PT, "X_c"), Y_c)
X_d = Node(NodeInfo(NodeType.PT, "X_d"), Y_c)
Y_c.set_left(X_d)
Y_c.set_right(X_c)

#c = Node(NodeInfo(NodeType.PT, "c"), X_c)
#X_c.set_right(c)
#d = Node(NodeInfo(NodeType.PT, "d"), X_d)
#X_d.set_right(d)

print_tree(root, nameattr='label', left_child='left', right_child='right')

new_root = Node(NodeInfo(NodeType.NT, "Z_c", ref=root), None)

left_corner_transform(new_root)

print_tree(new_root, nameattr='label', left_child='left', right_child='right')
