import logging

from nltk import ParentedTree as PTree

from tagger import Tagger
from transform import LeftCornerTransformer


class UFTagger(Tagger):
    @staticmethod
    def create_shift_tag(label: str) -> str:
        if label.find("+") != -1:
            return "s" + "/" + "/".join(label.split("+")[:-1])
        else:
            return "s"

    @staticmethod
    def create_reduce_tag(label:str) -> str:
        if label.find("|") != -1:  # drop extra node labels created after binarization
            return "r"
        else:
            return "r" + "/" + label.replace("+", "/")

    @staticmethod
    def clump_tags(tags:[str]) -> [str]:
        clumped_tags = []
        for tag in tags:
            if tag[0] == 's':
                clumped_tags.append(tag)
            else:
                clumped_tags[-1] = clumped_tags[-1] + " " + tag
        return clumped_tags

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


