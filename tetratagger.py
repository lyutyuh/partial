from ppbtree import print_tree
from node import Node, NodeType, NodeInfo
from leftcorner_transformer import eps

class TopDownTetratagger(object):

	def __init__(self):
		pass

	def convert(self, root):
		""" convert left-corner transformed tree to shifts and reduces """
		print_tree(root, nameattr='label', left_child='left', right_child='right')
		stack = [root]
		while len(stack) > 0:
			print(stack)
			node = stack.pop()

			if node.node_info.type == NodeType.NT or node.node_info.type == NodeType.NT_NT:				
				print("REDUCE[ {0} --> {1} {2}]".format(*(node.label, node.left.label, node.right.label)))
				if not eps(node.right):
					stack.append(node.right)
				if not eps(node.left):
					stack.append(node.left)
				print(stack); print()
			elif node.node_info.type == NodeType.PT:
				print("SHIFT[ {0} ]".format(node.label))
				print(stack); print()

			#print(node.label)
			#if node.left is not None and node.left.node_info.type == NodeType.PT:
			#		print("SHIFT[ {0} ]".format(node.left.label))
			
			#if node.right is not None:
		    #		stack.append(node.right)



class BottomupTetratagger(object):
	""" Kitaev and Klein (2020)"""

	def __init__(self):
		pass


