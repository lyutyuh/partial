from ppbtree import print_tree

from node import Node, NodeType, NodeInfo
from leftcorner_transformer import left_corner_transform, right_corner_transform
from leftcorner_rev_transformer import rev_transform

from tetratagger import TopDownTetratagger, BottomUpTetratagger, tetra_visualize, tetra_alternate

from tree import random_tree

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


#print_tree(root, nameattr='label', left_child='left', right_child='right')

new_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
left_corner_transform(new_root)

rc_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
right_corner_transform(rc_root)
#print_tree(rc_root, nameattr='label', left_child='left', right_child='right')

rev_new_root = rev_transform(new_root)
while rev_new_root.parent is not None:
    rev_new_root = rev_new_root.parent

#print_tree(rev_new_root, nameattr='label', left_child='left', right_child='right')

# TODO: add a pruning method for nodes of the form X-X and its inverse, which I think is possible. Also remove unnecessary unaries
# TODO: fix tree visualization
# TODO: make sure that the leaf nodes stay after the transform
# TODO: unify the left- and right-corner transform code


random_tree(root, "S")
new_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
left_corner_transform(new_root)

print();print()
tdtt = TopDownTetratagger()
actions = tdtt.convert(new_root)

for a in tetra_visualize(actions):
    print(a)

exit(0)

butt = BottomUpTetratagger()
actions = butt.convert(rc_root)
for a in tetra_visualize(actions):
    print(a)
tetra_alternate(actions)
exit(0)


root = Node(NodeInfo(NodeType.NT, "4"), None)
one = Node(NodeInfo(NodeType.NT, "1"), root)
E = Node(NodeInfo(NodeType.PT, "E"), root)
root.set_left(one)
root.set_right(E)

A = Node(NodeInfo(NodeType.PT, "A"), one)
two = Node(NodeInfo(NodeType.NT, "2"), one)
one.set_left(A)
one.set_right(two)

B = Node(NodeInfo(NodeType.PT, "B"), two)
three = Node(NodeInfo(NodeType.NT, "3"), two)
two.set_left(B)
two.set_right(three)

C = Node(NodeInfo(NodeType.PT, "C"), three)
D = Node(NodeInfo(NodeType.PT, "D"), three)
three.set_left(C)
three.set_right(D)

rc_root = Node(NodeInfo(NodeType.NT, "S", ref=root), None)
right_corner_transform(rc_root)


butt = BottomUpTetratagger()
actions = butt.convert(rc_root)
for a in tetra_visualize(actions):
    print(a)