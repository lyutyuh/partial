from ppbtree import print_tree
from node import Node, NodeType, NodeInfo
from leftcorner_transformer import eps, extract_left_corner, extract_left_corner_no_eps
from enum import Enum


class TetraType(Enum):
    r = 0
    l = 1
    R = 2
    L = 3

def tetra_visualize(actions):
	for a in actions:
		if a == TetraType.r:
			yield "-->"
		if a == TetraType.l:
 			yield "<--"
		if a == TetraType.R:
 			yield "==>"
		if a == TetraType.L:
 			yield "<=="


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
				if not eps(node.right):
					stack.append(node.right)
				if not eps(node.left):
					stack.append(node.left)

				print("<==\tREDUCE[ {0} --> {1} {2}]".format(*(node.label, node.left.label, node.right.label)))


				print(stack); print()
			elif node.node_info.type == NodeType.PT:
				print("<--\tSHIFT[ {0} ]".format(node.label))
				print(stack); print()

			#print(node.label)
			#if node.left is not None and node.left.node_info.type == NodeType.PT:
			#		print("SHIFT[ {0} ]".format(node.left.label))
			
			#if node.right is not None:
		    #		stack.append(node.right)



class BottomUpTetratagger(object):
	""" Kitaev and Klein (2020)"""

	def __init__(self):
		pass

	def convert(self, tree):
		""" convert right-corner transformed tree to shifts and reduces """
		actions = []
		print_tree(tree, nameattr='label', left_child='left', right_child='right')
		print();print()
		lc = extract_left_corner_no_eps(tree)

		print()
		print("-->\tSHIFT[ {0} ]".format(lc.label))
		actions.append(TetraType.r)
		stack = [lc]
		print(stack)
		print()


		while len(stack) != 1 or stack[0].label != "S":

			print(stack)
			node = stack[-1]
					
			if node != node.parent.right and node.parent.right is not None:
				if node.parent.right.node_info.type == NodeType.PT:
					print("-->\tSHIFT[ {0} ]".format(node.parent.right.label))
					actions.append(TetraType.r)
					stack.append(node.parent.right)
			elif len(stack) >= 2:
				prev_node = stack[-2]
				if prev_node.node_info.type == NodeType.NT_NT:
					print("<==\tREDUCE[ {0} {1} --> {2} ]".format(*(prev_node.label, node.label, node.parent.label)))
					
					if prev_node.node_info2.label == node.label:
						actions.pop()
						actions.append(TetraType.l)

					else:
						actions.append(TetraType.L)
					stack.pop(); stack.pop()
					stack.append(node.parent)
			elif len(stack) == 1:
				if node.node_info.type == NodeType.PT or node.node_info.type == NodeType.NT:
					print("==>\tREDUCE[ {0} --> {1} ]".format(*(node.label, node.parent.label)))
					actions.append(TetraType.R)
					stack.pop()
					stack.append(node.parent)
			#input()
			"""
			if node.node_info.type == NodeType.PT:	
				if node.parent.node_info.type == NodeType.NT and node.node_info.type == NodeType.PT:
					print("-->\tSHIFT[ {0} ]".format(node.label))
					actions.append(TetraType.r)
				elif node.parent.node_info.type == NodeType.NT_NT:# and eps(node.parent.left):
					print("==>\tREDUCE[ {0} --> {1} ]".format(*(node.label, node.parent.label)))
					actions.append(TetraType.R)
					stack.pop()
					stack.append(node.parent)
					input()
			elif node.node_info.type == NodeType.NT_NT:
				if node.parent.left == node and node.parent.node_info.type == NodeType.NT_NT:
					print("-->\tSHIFT[ {0} ]".format(node.parent.right))
					actions.append(TetraType.r)
					stack.append(node.parent.right)
				elif node.parent.left == node and node.parent.node_info.type == NodeType.NT:
					print("<--\tSHIFT[ {0} ]".format(node.parent.right))
					actions.append(TetraType.r)
				elif node.parent.right == node:
					print("==>\tREDUCE[ {0} --> {1} ]".format(*(node.right.label, node.label)))
					actions.append(TetraType.R)

			elif node.node_info.type == NodeType.NT:
				if node.parent is not None:
					print("==>\tREDUCE[ {0} --> {1} ]".format(*(node.label, node.parent.label)))
					actions.append(TetraType.R)
			"""
			print(stack);print()

		return actions
