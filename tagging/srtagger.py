import logging
from abc import ABC

from nltk import ParentedTree as PTree
from tqdm import tqdm as tq

from learning.decode import BeamSearch
from tagging.tagger import Tagger
from tagging.transform import LeftCornerTransformer

import numpy as np

from tagging.tree_tools import find_node_type, NodeType


class SRTagger(Tagger, ABC):
    def add_trees_to_vocab(self, trees: []) -> None:
        self.label_vocab = set()
        for tree in tq(trees):
            for tag in self.tree_to_tags_pipeline(tree):
                self.tag_vocab.add(tag)
                idx = tag.find("/")
                if idx != -1:
                    self.label_vocab.add(tag[idx + 1:])
                else:
                    self.label_vocab.add("")
        self.tag_vocab = sorted(self.tag_vocab)
        self.label_vocab = sorted(self.label_vocab)

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


class SRTaggerBottomUp(SRTagger):
    def __init__(self, trees=None, add_remove_top=False):
        super().__init__(trees, add_remove_top)

        stack_depth_change_by_id = [None] * len(self.tag_vocab)
        for i, tag in enumerate(self.tag_vocab):
            if tag.startswith("s"):
                stack_depth_change_by_id[i] = +1
            elif tag.startswith("r"):
                stack_depth_change_by_id[i] = -1
        assert None not in stack_depth_change_by_id
        self._stack_depth_change_by_id = np.array(
            stack_depth_change_by_id, dtype=int)

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

    def logits_to_ids(self, logits: [], mask, crf_transitions=None) -> [int]:
        beam_search = BeamSearch(
            initial_stack_depth=0,
            stack_depth_change_by_id=self._stack_depth_change_by_id,
            max_depth=12,
            keep_per_depth=1,
        )

        last_t = None
        seq_len = sum(mask)
        idx = 1
        for t in range(logits.shape[0]):
            if mask is not None and not mask[t]:
                continue
            if last_t is not None:
                beam_search.advance(
                    logits[last_t, :-len(self.tag_vocab)]
                )
            if idx == seq_len:
                beam_search.advance(logits[t, -len(self.tag_vocab):], is_last=True)
            else:
                beam_search.advance(logits[t, -len(self.tag_vocab):])
            last_t = t

        score, best_tag_ids = beam_search.get_path()
        return best_tag_ids


class SRTaggerTopDown(SRTagger):
    def __init__(self, trees=None, add_remove_top=False):
        super().__init__(trees, add_remove_top)

        reduce_tag_vocab_size = len(
            [tag for tag in self.tag_vocab if tag[0].startswith('r')])
        shift_tag_vocab_size = len(
            [tag for tag in self.tag_vocab if tag[0].startswith('s')])
        is_shift_mask = np.concatenate(
            [
                np.zeros(reduce_tag_vocab_size),
                np.ones(shift_tag_vocab_size),
            ]
        )
        self._reduce_tags_only = np.asarray(-1e9 * is_shift_mask, dtype=float)

        stack_depth_change_by_id = [None] * len(self.tag_vocab)
        stack_depth_change_by_id_l2 = [None] * len(self.tag_vocab)
        for i, tag in enumerate(self.tag_vocab):
            if tag.startswith("s"):
                stack_depth_change_by_id_l2[i] = 0
                stack_depth_change_by_id[i] = -1
            elif tag.startswith("r"):
                stack_depth_change_by_id_l2[i] = -1
                stack_depth_change_by_id[i] = +1
        assert None not in stack_depth_change_by_id
        assert None not in stack_depth_change_by_id_l2
        self._stack_depth_change_by_id = np.array(
            stack_depth_change_by_id, dtype=int)
        self._stack_depth_change_by_id_l2 = np.array(
            stack_depth_change_by_id_l2, dtype=int)

    def tree_to_tags(self, root: PTree) -> [str]:
        stack: [PTree] = [root]
        tags = []

        while len(stack) > 0:
            node = stack[-1]

            if find_node_type(node) == NodeType.NT:
                stack.pop()
                logging.debug("REDUCE[ {0} --> {1} {2}]".format(
                    *(node.label(), node[0].label(), node[1].label())))
                tags.append(self.create_reduce_tag(node.label()))
                stack.append(node[1])
                stack.append(node[0])

            else:
                logging.debug("-->\tSHIFT[ {0} ]".format(node.label()))
                tags.append(self.create_shift_tag(node.label()))
                stack.pop()

        return tags

    def tags_to_tree(self, tags: [str], input_seq: [str]) -> PTree:
        if len(tags) == 1:  # base case
            assert tags[0].startswith('s')
            return PTree(input_seq[0][1], [input_seq[0][0]])

        assert tags[0].startswith('r')
        node = PTree(self._create_reduce_label(tags[0]), [])
        created_node_stack: [PTree] = [node]

        for tag in tags[1:]:
            parent: PTree = created_node_stack[-1]
            if tag.startswith('s'):
                new_node = PTree(input_seq[0][1], [input_seq[0][0]])
                input_seq.pop(0)
            else:
                label = self._create_reduce_label(tag)
                new_node = PTree(label, [])

            if len(parent) == 0:
                parent.insert(0, new_node)
            elif len(parent) == 1:
                parent.insert(1, new_node)
                created_node_stack.pop()

            if tag.startswith('r'):
                created_node_stack.append(new_node)

        if len(input_seq) != 0:
            raise ValueError("All the input sequence is not used")
        return node

    def logits_to_ids(self, logits: [], mask, crf_transitions=None) -> [int]:
        beam_search = BeamSearch(
            initial_stack_depth=1,
            stack_depth_change_by_id=self._stack_depth_change_by_id,
            stack_depth_change_by_id_l2=self._stack_depth_change_by_id_l2,
            max_depth=12,
            min_depth=0,
            keep_per_depth=1,
        )

        last_t = None
        seq_len = sum(mask)
        idx = 1
        is_last = False
        for t in range(logits.shape[0]):
            if mask is not None and not mask[t]:
                continue
            if last_t is not None:
                beam_search.advance(
                    logits[last_t, :-len(self.tag_vocab)]
                )
            if idx == seq_len:
                is_last = True
            if last_t is None:
                beam_search.advance(logits[t, -len(self.tag_vocab):] + self._reduce_tags_only,
                                    is_last=is_last)
            else:
                beam_search.advance(logits[t, -len(self.tag_vocab):], is_last=is_last)
            last_t = t

        score, best_tag_ids = beam_search.get_path()
        return best_tag_ids
