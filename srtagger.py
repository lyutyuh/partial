import logging

from nltk import ParentedTree as PTree
from nltk import Tree
from tqdm import tqdm as tq

from decode import BeamSearch
from tagger import Tagger
from transform import LeftCornerTransformer

import numpy as np


class SRTagger(Tagger):
    def __init__(self):
        super().__init__()

        is_even_mask = np.concatenate(
            [
                np.zeros(len(self.tag_vocab)),
                np.ones(len(self.tag_vocab)),
            ]
        )
        self._odd_tags_only = np.asarray(-1e9 * is_even_mask, dtype=float)
        self._even_tags_only = np.asarray(
            -1e9 * (1 - is_even_mask), dtype=float
        )

        stack_depth_change_by_id = [None] * len(self.tag_vocab)
        for i, tag in enumerate(self.tag_vocab):
            if tag.startswith("s"):
                stack_depth_change_by_id[i] = +1
            elif tag.startswith("r"):
                stack_depth_change_by_id[i] = -1
        assert None not in stack_depth_change_by_id
        self._stack_depth_change_by_id = np.array(
            stack_depth_change_by_id, dtype=np.int32
        )

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
        return tags

    def tags_to_tree(self, tags: [str], input_seq: [str]) -> PTree:
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

    def ids_from_logits(self, logits: [], mask) -> [int]:
        beam_search = BeamSearch(
            initial_stack_depth=0,
            stack_depth_change_by_id=self._stack_depth_change_by_id,
            max_depth=12,
            keep_per_depth=1,
        )

        last_t = None
        for t in range(logits.shape[0]):
            if mask is not None and not mask[t]:
                continue
            if last_t is not None:
                beam_search.advance(
                    logits[last_t, :] + self._odd_tags_only
                )
            beam_search.advance(logits[t, :] + self._even_tags_only)
            last_t = t

        score, best_tag_ids = beam_search.get_path()
        return best_tag_ids

    def tree_from_logits(self, logits: [], leave_nodes: [], mask=None) -> Tree:
        ids = self.ids_from_logits(logits, mask)
        return self.ids_to_tree_pipeline(ids, leave_nodes)
