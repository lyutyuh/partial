import copy

from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.corpus.reader import DependencyCorpusReader
from nltk.tree import Tree
import os

from tqdm import tqdm as tq
# using ^^^ as delimiter
# since ^ never appears in PTB
nonproj_counter = 0

def augment_constituent_tree(const_tree, dep_tree):
    # augment constituent tree leaves into dicts

    assert len(const_tree.leaves()) == len(dep_tree.nodes) - 1

    leaf_nodes = list(const_tree.treepositions('leaves'))
    for i, pos in enumerate(leaf_nodes):
        x = dep_tree.nodes[1+i]
        y = const_tree[pos].replace("\\","")
        assert (x['word'] == y), (const_tree, dep_tree)

        # expanding leaves with dependency info
        const_tree[pos] = {
            "word": dep_tree.nodes[1+i]["word"],
            "head": dep_tree.nodes[1+i]["head"],
            "rel": dep_tree.nodes[1+i]["rel"]
        }

    return const_tree


def get_lexicalized_tree(root, offset=0):
    # Return:
    #     The index of the head in this tree
    # and:
    #     The dependant that this tree points to
    if type(root) is dict:
        # leaf node already, return its head
        return 0, root['head']

    # word offset in the current tree
    words_seen = 0
    root_projection = (offset, offset + len(root.leaves()))
    # init the return values to be None
    head, root_points_to = None, None

    # traverse the consituent tree
    for idx, child in enumerate(root):
        head_of_child, child_points_to = get_lexicalized_tree(child, offset+words_seen)
        if type(child) == type(root):
            words_seen += len(child.leaves())
        else:
            # leaf node visited
            words_seen += 1

        if child_points_to < root_projection[0] or child_points_to >= root_projection[1]:
            # pointing to outside of the current tree
            if head is not None:
                # this rarely happens in PTB
                # minimal example:
                # What is greatness ?
                # dependency: is -> What, greatness -> What
                # parse: (TOP (SBARQ (WHNP (WP What)) (SQ (VBZ is) (NP (JJ greatness))) (. ?)))
                print("error! Non-projectivity detected.", root_projection, idx)
                nonproj_counter += 1
                continue # choose the first child as head
            head = idx
            root_points_to = child_points_to

    if root_points_to is None:
        # self contained sub-sentence
        print("multiple roots detected", root)
        root_points_to = 0

    original_label = root.label()
    root.set_label(f"{original_label}^^^{head}")

    return head, root_points_to


def dep2lex(dep_tree, language="English"):

    def dfs(node_idx):
        dependencies = []
        for rel in dep_tree.nodes[node_idx]["deps"]:
            for dependent in dep_tree.nodes[node_idx]["deps"][rel]:
                dependencies.append((dependent, rel))

        dependencies.append((node_idx, "SELF"))
        if len(dependencies) == 1:
            # no dependent at all, leaf node
            return Tree(
                f"X^^^{dep_tree.nodes[node_idx]['rel']}",
                [
                    Tree(
                        f"{dep_tree.nodes[node_idx]['tag']}",
                        [
                            f"{dep_tree.nodes[node_idx]['word']}"
                        ]
                    )
                ]
            )
        # Now, len(dependencies) >= 2, sort dependents
        dependencies = sorted(dependencies)

        lex_tree_root = Tree(f"X^^^{0}", [])
        empty_slot = lex_tree_root # place to fill in the next subtree
        for idx, dependency in enumerate(dependencies):
            if dependency[0] < node_idx:
                # waiting for a head in the right child
                lex_tree_root.set_label(f"X^^^{1}")
                if len(lex_tree_root) == 0:
                    # the first non-head child
                    lex_tree_root.insert(
                        0, dfs(dependency[0])
                    )
                else:
                    # not the first non-head child
                    # insert a sub tree: \
                    #                  X^^^1
                    #                  /   \
                    #                word  [empty_slot]
                    empty_slot.insert(
                        1,
                        Tree(f"X^^^{1}",[
                            dfs(dependency[0])
                        ])
                    )
                    empty_slot = empty_slot[1]
            elif dependency[0] == node_idx:
                if len(empty_slot) == 1:
                    # This is the head
                    empty_slot.insert(
                        1,
                        Tree(
                            f"X^^^{dep_tree.nodes[dependency[0]]['rel']}",
                            [
                                Tree(
                                    f"{dep_tree.nodes[dependency[0]]['tag']}",
                                    [
                                        f"{dep_tree.nodes[dependency[0]]['word']}"
                                    ]
                                )
                            ]
                        )
                    )
                else:
                    lex_tree_root = Tree(
                        f"X^^^{dep_tree.nodes[dependency[0]]['rel']}",
                        [
                            Tree(
                                f"{dep_tree.nodes[dependency[0]]['tag']}",
                                [
                                    f"{dep_tree.nodes[dependency[0]]['word']}"
                                ]
                            )
                        ]
                    )
                pass
            else:
                # moving on to the right of the head
                lex_tree_root = Tree(
                    f"X^^^{0}",
                    [
                        lex_tree_root,
                        dfs(dependency[0])
                    ]
                )
        return lex_tree_root

    return dfs(
        dep_tree.nodes[0]["deps"]["root"][0] if language == "English" else 
        0
    )


if __name__ == "__main__":
    for language in ["English", "Polish", "French","Basque","German","Hebrew","Hungarian","Korean","swedish"]:
        if language == "English":
            path = os.path.dirname(os.path.abspath(__file__))+"/dependency_ptb/{language}.dep.{split}.conll"
            paths = [path.format(language=language, split=split) for split in ["train", "dev", "test"]]
        elif language in ["Hebrew", "swedish"]:
            path = os.path.dirname(os.path.abspath(__file__))+"/spmrl/SPMRL_SHARED_2014_NO_ARABIC/{language_uppercase}_SPMRL/gold/conll/{split}/{split}.{language}.gold.conll"
            paths = [path.format(language=language, language_uppercase=language.upper(), split=split) for split in ["train5k", "dev", "test"]]
        else:
            path = os.path.dirname(os.path.abspath(__file__))+"/spmrl/SPMRL_SHARED_2014_NO_ARABIC/{language_uppercase}_SPMRL/gold/conll/{split}/{split}.{language}.gold.conll"
            paths = [path.format(language=language, language_uppercase=language.upper(), split=split) for split in ["train", "dev", "test"]]

        reader = DependencyCorpusReader(
            os.path.dirname(os.path.abspath(__file__)), 
            paths
        )

        for path, split in zip(paths, ["train", "dev", "test"]):
            nonproj_counter = 0
            print(f"Converting {path} to lexicalized tree")
            dep_trees = reader.parsed_sents(path)

            lexicalized_trees = []
            for dep_tree in tq(dep_trees):
                lex_tree = dep2lex(dep_tree, language=language)
                lexicalized_trees.append(lex_tree)
                lex_tree_leaves = tuple(lex_tree.leaves())
                dep_tree_leaves = tuple([str(node["word"]) for _,node in sorted(dep_tree.nodes.items())])
                if language == "English":
                    dep_tree_leaves = dep_tree_leaves[1:]
                if lex_tree_leaves != dep_tree_leaves:
                    nonproj_counter += 1

            print(f"{nonproj_counter} non-projective trees detected of {len(lexicalized_trees)} trees")
            assert len(lexicalized_trees) == len(dep_trees), (len(lexicalized_trees), len(dep_trees))

            print(f"Writing lexicalized trees to {os.path.dirname(os.path.abspath(__file__))}/{language}.lex.{split}")
            with open(os.path.dirname(os.path.abspath(__file__))+f"/dep/{language}.lex.{split}", "w") as fout:
                for lex_tree in lexicalized_trees:
                    fout.write(lex_tree._pformat_flat("", "()", False) + "\n")
