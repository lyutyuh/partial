import logging

from nltk import ParentedTree as PTree
from nltk import Tree
from tqdm import tqdm as tq

from tagger import Tagger
from transform import LeftCornerTransformer


class UFTagger(Tagger):
    def add_trees_to_vocab(self, trees: []) -> None:
        for tree in tq(trees):
            for tag in self.tree_to_tags_pipeline(tree):
                for subtag in tag.split(" "):
                    self.tag_vocab.add(subtag)
        self.tag_vocab = sorted(self.tag_vocab)

    def clumped_tag_to_clumped_ids(self, tags: [str]) -> [str]:
        clumped_ids = []
        for tag in tags:
            indices = []
            for subtag in tag.split(" "):
                indices.append(str(self.tag_vocab.index(subtag)))
            clumped_ids.append(" ".join(indices))
        return clumped_ids

    def clumped_ids_to_clumped_tags(self, ids: [str]) -> [str]:
        clumped_tags = []
        for id in ids:
            tags = []
            for sub_id in id.split(" "):
                tags.append(self.tag_vocab[int(sub_id)])
            clumped_tags.append(" ".join(tags))
        return clumped_tags

    def tree_to_ids_pipeline(self, tree: Tree) -> [int]:
        tags = self.tree_to_tags_pipeline(tree)
        return self.clumped_tag_to_clumped_ids(tags)

    def ids_to_tree_pipeline(self, ids: [int], input_seq: []) -> Tree:
        tags = self.clumped_ids_to_clumped_tags(ids)
        return self.tags_to_tree_pipeline(tags, input_seq)

    @staticmethod
    def create_shift_tag(label: str) -> str:
        if label.find("+") != -1:
            return "s" + "/" + "/".join(label.split("+")[:-1])
        else:
            return "s"

    @staticmethod
    def create_reduce_tag(label: str) -> str:
        if label.find("|") != -1:  # drop extra node labels created after binarization
            return "r"
        else:
            return "r" + "/" + label.replace("+", "/")

    @staticmethod
    def _create_reduce_label(tag: str) -> str:
        idx = tag.find("/")
        if idx == -1:
            label = "|"  # to mark the second part as an extra node created via binarizaiton
        else:
            label = tag[idx + 1:].replace("/", "+")
        return label

    @staticmethod
    def clump_tags(tags: [str]) -> [str]:
        clumped_tags = []
        for tag in tags:
            if tag.startswith('s'):
                clumped_tags.append(tag)
            else:
                clumped_tags[-1] = clumped_tags[-1] + " " + tag
        return clumped_tags

    @staticmethod
    def flatten_tags(tags: [str]) -> [str]:
        raw_tags = []
        for tag in tags:
            raw_tags += tag.split(" ")
        return raw_tags

    def tree_to_tags(self, root: PTree) -> [str]:
        tags = []
        lc = LeftCornerTransformer.extract_left_corner_no_eps(root)
        tags.append(self.create_shift_tag(lc.label()))

        logging.debug("SHIFT {}".format(lc.label()))
        stack = [lc]

        while len(stack) > 0:
            node = stack[-1]

            if node.left_sibling() is None and node.right_sibling() is not None:
                lc = LeftCornerTransformer.extract_left_corner_no_eps(node.right_sibling())
                stack.append(lc)
                logging.debug("SHIFT {}".format(lc.label()))
                tags.append(self.create_shift_tag(lc.label()))

            elif len(stack) >= 2 and (
                    node.right_sibling() == stack[-2] or node.left_sibling() == stack[-2]):
                prev_node = stack[-2]
                logging.debug("REDUCE[ {0} {1} --> {2} ]".format(
                    *(prev_node.label(), node.label(), node.parent().label())))

                tags.append(self.create_reduce_tag(node.parent().label()))
                stack.pop()
                stack.pop()
                stack.append(node.parent())

            elif stack[0].parent() is None and len(stack) == 1:
                stack.pop()
                continue

        return self.clump_tags(tags)

    def tags_to_tree(self, tags: [str], input_seq: [str]) -> PTree:
        tags = self.flatten_tags(tags)
        created_node_stack = []
        node = None

        if len(tags) == 1:  # base case
            assert tags[0].startswith('s')
            return PTree(input_seq[0][1], [input_seq[0][0]])
        for tag in tags:
            if tag.startswith('s'):
                created_node_stack.append(PTree(input_seq[0][1], [input_seq[0][0]]))
                input_seq.pop(0)
            else:
                last_node = created_node_stack.pop()
                last_2_node = created_node_stack.pop()
                node = PTree(self._create_reduce_label(tag), [last_2_node, last_node])
                created_node_stack.append(node)

        if len(input_seq) != 0:
            raise ValueError("All the input sequence is not used")
        return node
