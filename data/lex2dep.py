import copy
from tqdm import tqdm as tq

from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.corpus.reader import DependencyCorpusReader


def get_dependency_from_lexicalized_tree(lex_tree, triple_dict, offset=0):
    # this recursion assumes projectivity
    # Input:
    #     root of lex-tree
    # Output:
    #     the global index of the dependency root

    if type(lex_tree) is not str and len(lex_tree) == 1:
        # unary rule
        # returning the global index of the head
        return offset

    head_branch_index = int(lex_tree.label().split("^^^")[1])
    head_global_index = None
    branch_to_global_dict = {}

    for branch_id_child, child in enumerate(lex_tree):
        global_id_child = get_dependency_from_lexicalized_tree(
            child, triple_dict, offset=offset
        )
        offset = offset + len(child.leaves())
        branch_to_global_dict[branch_id_child] = global_id_child
        if branch_id_child == head_branch_index:
            head_global_index = global_id_child

    for branch_id_child, child in enumerate(lex_tree):
        if branch_id_child != head_branch_index:
            triple_dict[branch_to_global_dict[branch_id_child]] = head_global_index

    return head_global_index


def get_dep_triples(lex_tree):
    triple_dict = {}
    dep_triples = []
    sent_root = get_dependency_from_lexicalized_tree(
        lex_tree, triple_dict
    )
    # the root of the whole sentence should refer to ROOT
    assert sent_root not in triple_dict
    # the root of the sentence
    triple_dict[sent_root] = -1
    for head, tail in sorted(triple_dict.items()):

        dep_triples.append((
            head, tail,
            # lex_tree.pos()[head][1].split("^^^")[1]
        ))
    return dep_triples


if __name__ == "__main__":
    for language in ["English"]:

        lex_reader = BracketParseCorpusReader(
            "./", 
            [f"./{language}.lex.{split}" for split in ['dev', 'train', 'test']]
        )
        dep_reader = DependencyCorpusReader(
            "./", 
            [f"./{language}.dep.{split}.conll" for split in ['dev', 'train', 'test']]
        )

        for split in ["dev", "train", "test"]:
            print(f"Converting {language}.lex.{split} to dependency triples")
            lex_trees = lex_reader.parsed_sents(f"./{language}.lex.{split}")
            dep_trees = dep_reader.parsed_sents(f"./{language}.dep.{split}.conll")

            for (lex_tree, dep_tree) in zip(lex_trees, dep_trees):
                # In test, 1 sentence cannot be mapped back
                dep_triples = get_dep_triples(lex_tree)
                gt_dep_triples = []
                for i in range(1, len(dep_tree.nodes)):
                    gt_dep_triples.append(
                        (i-1, dep_tree.nodes[i]["head"]-1)
                )
                if len(set(gt_dep_triples) - set(dep_triples)):
                    print(
                        set(gt_dep_triples) - set(dep_triples)
                    )
