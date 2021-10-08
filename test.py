from ppbtree import print_tree

from node import Node, NodeType
from leftcorner_transformer import transform

root = Node(NodeType.NT, "S", None)
np = Node(NodeType.NT, "NP", root)
vp = Node(NodeType.NT, "VP", root, sibling=np)
np.set_sibling(vp)
root.set_left(np)
root.set_right(vp)

det = Node(NodeType.PT, "Det(the)", np)
n = Node(NodeType.PT, "N(dog)", np, sibling=det)
det.set_sibling(n)
np.set_left(det)
np.set_right(n)

v = Node(NodeType.PT, "V(ran)", vp)
adv = Node(NodeType.PT, "Adv(fast)", vp, sibling=v)
v.set_sibling(adv)
vp.set_left(v)
vp.set_right(adv)

new_root = Node(NodeType.NT, "S", None)
transform(new_root, root)

print_tree(new_root, nameattr='label', left_child='left', right_child='right')
