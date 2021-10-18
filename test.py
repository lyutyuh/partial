from ppbtree import print_tree

from node import Node, NodeType, NodeInfo
from leftcorner_transformer import transform
from leftcorner_rev_transformer import rev_transform

root = Node(NodeInfo(NodeType.NT, "S"), None)
np = Node(NodeInfo(NodeType.NT, "NP"), root)
vp = Node(NodeInfo(NodeType.NT, "VP"), root)
root.set_left(np)
root.set_right(vp)

det = Node(NodeInfo(NodeType.PT, "Det(the)"), np)
n = Node(NodeInfo(NodeType.PT, "N(dog)"), np)
np.set_left(det)
np.set_right(n)

v = Node(NodeInfo(NodeType.PT, "V(ran)"), vp)
adv = Node(NodeInfo(NodeType.PT, "Adv(fast)"), vp)
vp.set_left(v)
vp.set_right(adv)

new_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)

print_tree(root, nameattr='label', left_child='left', right_child='right')

transform(new_root)

print_tree(new_root, nameattr='label', left_child='left', right_child='right')

rev_new_root = rev_transform(new_root)
while rev_new_root.parent is not None:
    rev_new_root = rev_new_root.parent

print_tree(rev_new_root, nameattr='label', left_child='left', right_child='right')
