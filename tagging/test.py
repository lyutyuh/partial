# import logging
import unittest

import numpy as np
from nltk import ParentedTree
from nltk import Tree
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from tqdm import tqdm as tq

from learning.evaluate import evalb
from original_tetratagger import TetraTagSystem
from tagging.hexatagger import HexaTagger
from tagging.srtagger import SRTaggerBottomUp, SRTaggerTopDown
from tetratagger import BottomUpTetratagger, TopDownTetratagger
from transform import LeftCornerTransformer, RightCornerTransformer
from tree_tools import random_tree, is_topo_equal, create_dummy_tree, relabel_tree

# logging.getLogger().setLevel(logging.DEBUG)

np.random.seed(0)


class TestTransforms(unittest.TestCase):
    def test_transform(self):
        tree = ParentedTree.fromstring("(S (NP (det the) (N dog)) (VP (V ran) (Adv fast)))")
        tree.pretty_print()
        new_tree_lc = ParentedTree("S", [])
        LeftCornerTransformer.transform(new_tree_lc, tree, tree)
        new_tree_lc.pretty_print()

        new_tree_rc = ParentedTree("S", [])
        RightCornerTransformer.transform(new_tree_rc, tree, tree)
        new_tree_rc.pretty_print()

    def test_rev_rc_transform(self, trials=100):
        for _ in range(trials):
            t = ParentedTree("ROOT", [])
            random_tree(t, depth=0, cutoff=5)
            new_tree_rc = ParentedTree("S", [])
            RightCornerTransformer.transform(new_tree_rc, t, t)
            tree_back = ParentedTree("X", ["", ""])
            tree_back = RightCornerTransformer.rev_transform(tree_back, new_tree_rc)
            self.assertEqual(tree_back, t)

    def test_rev_lc_transform(self, trials=100):
        for _ in range(trials):
            t = ParentedTree("ROOT", [])
            random_tree(t, depth=0, cutoff=5)
            new_tree_lc = ParentedTree("S", [])
            LeftCornerTransformer.transform(new_tree_lc, t, t)
            tree_back = ParentedTree("X", ["", ""])
            tree_back = LeftCornerTransformer.rev_transform(tree_back, new_tree_lc)
            self.assertEqual(tree_back, t)


class TestTagging(unittest.TestCase):
    def test_buttom_up(self):
        tree = ParentedTree.fromstring("(S (NP (det the) (N dog)) (VP (V ran) (Adv fast)))")
        tree.pretty_print()
        tree_rc = ParentedTree("S", [])
        RightCornerTransformer.transform(tree_rc, tree, tree)
        tree_rc.pretty_print()
        tagger = BottomUpTetratagger()
        tags, _ = tagger.tree_to_tags(tree_rc)

        for tag in tagger.tetra_visualize(tags):
            print(tag)
        print("--" * 20)

    def test_buttom_up_alternate(self, trials=100):
        for _ in range(trials):
            t = ParentedTree("ROOT", [])
            random_tree(t, depth=0, cutoff=5)
            t_rc = ParentedTree("S", [])
            RightCornerTransformer.transform(t_rc, t, t)
            tagger = BottomUpTetratagger()
            tags, _ = tagger.tree_to_tags(t_rc)
            self.assertTrue(tagger.is_alternating(tags))
            self.assertTrue((2 * len(t.leaves()) - 1) == len(tags))

    def test_round_trip_test_buttom_up(self, trials=100):
        for _ in range(trials):
            tree = ParentedTree("ROOT", [])
            random_tree(tree, depth=0, cutoff=5)
            tree_rc = ParentedTree("S", [])
            RightCornerTransformer.transform(tree_rc, tree, tree)
            tree.pretty_print()
            tree_rc.pretty_print()
            tagger = BottomUpTetratagger()
            tags, _ = tagger.tree_to_tags(tree_rc)
            root_from_tags = tagger.tags_to_tree(tags, tree.leaves())
            tree_back = ParentedTree("X", ["", ""])
            tree_back = RightCornerTransformer.rev_transform(tree_back, root_from_tags,
                                                             pick_up_labels=False)
            self.assertTrue(is_topo_equal(tree, tree_back))

    def test_top_down(self):
        tree = ParentedTree.fromstring("(S (NP (det the) (N dog)) (VP (V ran) (Adv fast)))")
        tree.pretty_print()
        tree_lc = ParentedTree("S", [])
        LeftCornerTransformer.transform(tree_lc, tree, tree)
        tree_lc.pretty_print()
        tagger = TopDownTetratagger()
        tags, _ = tagger.tree_to_tags(tree_lc)

        for tag in tagger.tetra_visualize(tags):
            print(tag)
        print("--" * 20)

    def test_top_down_alternate(self, trials=100):
        for _ in range(trials):
            t = ParentedTree("ROOT", [])
            random_tree(t, depth=0, cutoff=5)
            t_lc = ParentedTree("S", [])
            LeftCornerTransformer.transform(t_lc, t, t)
            tagger = TopDownTetratagger()
            tags = tagger.tree_to_tags(t_lc)
            self.assertTrue(tagger.is_alternating(tags))
            self.assertTrue((2 * len(t.leaves()) - 1) == len(tags))

    def round_trip_test_top_down(self, trials=100):
        for _ in range(trials):
            tree = ParentedTree("ROOT", [])
            random_tree(tree, depth=0, cutoff=5)
            tree_lc = ParentedTree("S", [])
            LeftCornerTransformer.transform(tree_lc, tree, tree)
            tagger = TopDownTetratagger()
            tags = tagger.tree_to_tags(tree_lc)
            root_from_tags = tagger.tags_to_tree(tags, tree.leaves())
            tree_back = ParentedTree("X", ["", ""])
            tree_back = LeftCornerTransformer.rev_transform(tree_back, root_from_tags,
                                                            pick_up_labels=False)
            self.assertTrue(is_topo_equal(tree, tree_back))


class TestPipeline(unittest.TestCase):
    def test_example_colab(self):
        example_tree = Tree.fromstring(
            "(S (NP (PRP She)) (VP (VBZ enjoys) (S (VP (VBG playing) (NP (NN tennis))))) (. .))")
        tagger = BottomUpTetratagger()
        tags = tagger.tree_to_tags_pipeline(example_tree)[0]
        print(tags)
        for tag in tagger.tetra_visualize(tags):
            print(tag)

    def test_dummy_tree(self):
        example_tree = Tree.fromstring(
            "(S (NP (PRP She)) (VP (VBZ enjoys) (S (VP (VBG playing) (NP (NN tennis))))) (. .))")
        print(example_tree.pos())
        dummy = create_dummy_tree(example_tree.pos())
        dummy.pretty_print()
        example_tree.pretty_print()
        print(evalb("../EVALB/", [example_tree], [dummy]))

    def test_tree_linearizations(self):
        READER = BracketParseCorpusReader('../data/spmrl/',
                                          ['English.train', 'English.dev', 'English.test'])
        trees = READER.parsed_sents('English.test')
        for tree in trees:
            print(tree)
            print(" ".join(tree.leaves()))

    def test_compare_to_original_tetratagger(self):
        # import pickle
        # with open("../data/tetra.pkl", 'rb') as f:
        #     tag_vocab = pickle.load(f)

        READER = BracketParseCorpusReader('../data/spmrl/',
                                          ['English.train', 'English.dev', 'English.test'])
        trees = READER.parsed_sents('English.test')
        tagger = BottomUpTetratagger(add_remove_top=True)
        tetratagger = TetraTagSystem(trees=trees)
        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            # original_tags = TetraTagSequence.from_tree(original_tree)
            tags = tagger.tree_to_tags_pipeline(tree)[0]
            tetratags = tetratagger.tags_from_tree(tree)
            # ids = tagger.tree_to_ids_pipeline(tree)
            # ids = tagger.ids_from_tree(tree)
            # tree_back = tagger.tags_to_tree_pipeline(tags, tree.pos())
            # self.assertEqual(original_tree, tree_back)
            self.assertEqual(list(tetratags), list(tags))

    def test_example_colab_lc(self):
        example_tree = Tree.fromstring(
            "(S (NP (PRP She)) (VP (VBZ enjoys) (S (VP (VBG playing) (NP (NN tennis))))) (. .))")
        original_tree = example_tree.copy(deep=True)
        tagger = TopDownTetratagger()
        tags = tagger.tree_to_tags_pipeline(example_tree)[0]
        tree_back = tagger.tags_to_tree_pipeline(tags, example_tree.pos())
        tree_back.pretty_print()
        self.assertEqual(original_tree, tree_back)

    def test_top_down_tetratagger(self):
        READER = BracketParseCorpusReader('../data/spmrl/',
                                          ['English.train', 'English.dev', 'English.test'])
        trees = READER.parsed_sents('English.test')
        tagger = TopDownTetratagger(add_remove_top=True)
        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            tags = tagger.tree_to_tags_pipeline(tree)[0]
            tree_back = tagger.tags_to_tree_pipeline(tags, tree.pos())
            self.assertEqual(original_tree, tree_back)

    def test_tag_ids_top_down(self):
        READER = BracketParseCorpusReader('../data/spmrl/',
                                          ['English.train', 'English.dev', 'English.test'])
        trees = READER.parsed_sents('English.test')
        tagger = TopDownTetratagger(trees, add_remove_top=True)
        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            ids = tagger.tree_to_ids_pipeline(tree)
            tree_back = tagger.ids_to_tree_pipeline(ids, tree.pos())
            self.assertEqual(original_tree, tree_back)

    def test_tag_ids_bottom_up(self):
        READER = BracketParseCorpusReader('../data/spmrl/',
                                          ['English.train', 'English.dev', 'English.test'])
        trees = READER.parsed_sents('English.test')
        tagger = BottomUpTetratagger(trees, add_remove_top=True)
        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            ids = tagger.tree_to_ids_pipeline(tree)
            tree_back = tagger.ids_to_tree_pipeline(ids, tree.pos())
            self.assertEqual(original_tree, tree_back)

    def test_decoder_edges(self):
        READER = BracketParseCorpusReader('../data/spmrl/',
                                          ['English.train', 'English.dev', 'English.test'])
        trees = READER.parsed_sents('English.test')
        tagger_bu = SRTaggerBottomUp(add_remove_top=True)
        tagger_td = SRTaggerTopDown(add_remove_top=True)
        unique_tags = dict()
        counter = 0
        for tree in tq(trees):
            counter += 1
            tags = tagger_td.tree_to_tags_pipeline(tree)[0]
            for i, tag in enumerate(tags):
                tag_s = tag.split("/")[0]
                if i < len(tags) - 1:
                    tag_next = tags[i + 1].split("/")[0]
                else:
                    tag_next = None
                if tag_s not in unique_tags and tag_next is not None:
                    unique_tags[tag_s] = {tag_next}
                elif tag_next is not None:
                    unique_tags[tag_s].add(tag_next)
        print(unique_tags)


class TestSRTagger(unittest.TestCase):
    def test_tag_sequence_example(self):
        READER = BracketParseCorpusReader('../data/spmrl/',
                                          ['English.train', 'English.dev', 'English.test'])
        trees = READER.parsed_sents('English.train')
        tagger = SRTaggerBottomUp(add_remove_top=True)
        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            tags = tagger.tree_to_tags_pipeline(tree)[0]
            for tag in tags:
                if tag.find('/X/') != -1:
                    print(tag)
                    original_tree.pretty_print()
            tree_back = tagger.tags_to_tree_pipeline(tags, tree.pos())
            self.assertEqual(original_tree, tree_back)

    def test_example_both_version(self):
        import nltk
        example_tree = nltk.Tree.fromstring(
            "(TOP (S (NP (PRP She)) (VP (VBZ enjoys) (S (VP (VBG playing) (NP (NN tennis))))) (. .)))")
        example_tree.pretty_print()
        td_tagger = SRTaggerTopDown(add_remove_top=True)
        bu_tagger = SRTaggerBottomUp(add_remove_top=True)
        t1 = example_tree.copy(deep=True)
        t2 = example_tree.copy(deep=True)
        td_tags = td_tagger.tree_to_tags_pipeline(t1)[0]
        bu_tags = bu_tagger.tree_to_tags_pipeline(t2)[0]
        self.assertEqual(set(td_tags), set(bu_tags))
        print(list(td_tags))
        print(list(bu_tags))

    def test_bu_binarize(self):
        READER = BracketParseCorpusReader('../data/spmrl/',
                                          ['English.train', 'English.dev', 'English.test'])
        trees = READER.parsed_sents('English.dev')
        tagger = SRTaggerBottomUp(add_remove_top=True)
        for tree in tq(trees):
            tags = tagger.tree_to_tags_pipeline(tree)[0]
            for idx, tag in enumerate(tags):
                if tag.startswith("rr"):
                    if (idx + 1) < len(tags):
                        self.assertTrue(tags[idx + 1].startswith('r'))
                elif tag.startswith("r"):
                    if (idx + 1) < len(tags):
                        self.assertTrue(tags[idx + 1].startswith('s'))

    def test_td_binarize(self):
        READER = BracketParseCorpusReader('../data/spmrl/',
                                          ['English.train', 'English.dev', 'English.test'])
        trees = READER.parsed_sents('English.dev')
        tagger = SRTaggerTopDown(add_remove_top=True)
        for tree in tq(trees):
            tags = tagger.tree_to_tags_pipeline(tree)[0]
            for idx, tag in enumerate(tags):
                if tag.startswith("s"):
                    if (idx + 1) < len(tags):
                        self.assertTrue(
                            tags[idx + 1].startswith('rr') or tags[idx + 1].startswith('s'))
                elif tag.startswith("r"):
                    if (idx + 1) < len(tags):
                        self.assertTrue(not tags[idx + 1].startswith('rr'))

    def test_td_bu(self):
        READER = BracketParseCorpusReader('../data/spmrl/',
                                          ['English.train', 'English.dev', 'English.test'])
        trees = READER.parsed_sents('English.dev')
        td_tagger = SRTaggerTopDown(add_remove_top=True)
        bu_tagger = SRTaggerBottomUp(add_remove_top=True)

        for tree in tq(trees):
            td_tags = td_tagger.tree_to_tags_pipeline(tree)[0]
            bu_tags = bu_tagger.tree_to_tags_pipeline(tree)[0]
            self.assertEqual(set(td_tags), set(bu_tags))

    def test_tag_ids(self):
        READER = BracketParseCorpusReader('../data/spmrl/',
                                          ['English.train', 'English.dev', 'English.test'])
        trees = READER.parsed_sents('English.train')
        tagger = SRTaggerBottomUp(trees, add_remove_top=False)
        for tree in tq(trees):
            ids = tagger.tree_to_ids_pipeline(tree)
            tree_back = tagger.ids_to_tree_pipeline(ids, tree.pos())
            self.assertEqual(tree, tree_back)

    def test_max_length(self):
        READER = BracketParseCorpusReader('../data/spmrl/',
                                          ['English.train', 'English.dev', 'English.test'])
        trees = READER.parsed_sents('English.train')
        tagger = SRTaggerBottomUp(trees, add_remove_top=True)
        print(len(tagger.tag_vocab))


class TestSPMRL(unittest.TestCase):
    def test_reading_trees(self):
        langs = ['Korean']
        for l in langs:
            READER = BracketParseCorpusReader('../data/spmrl/',
                                              [l + '.train', l + '.dev', l + '.test'])
            trees = READER.parsed_sents(l + '.test')
            trees[0].pretty_print()

    def test_tagging_bu_sr(self):
        langs = ['Basque', 'French', 'German', 'Hebrew', 'Hungarian', 'Korean', 'Polish',
                 'Swedish']
        for l in langs:
            READER = BracketParseCorpusReader('../data/spmrl/',
                                              [l + '.train', l + '.dev', l + '.test'])
            trees = READER.parsed_sents(l + '.test')
            trees[0].pretty_print()
            tagger = SRTaggerBottomUp(trees, add_remove_top=True)
            print(tagger.tree_to_tags_pipeline(trees[0]))
            print(tagger.tree_to_ids_pipeline(trees[0]))

    def test_tagging_td_sr(self):
        langs = ['Basque', 'French', 'German', 'Hebrew', 'Hungarian', 'Korean', 'Polish',
                 'Swedish']
        for l in langs:
            READER = BracketParseCorpusReader('../data/spmrl/',
                                              [l + '.train', l + '.dev', l + '.test'])
            trees = READER.parsed_sents(l + '.test')
            trees[0].pretty_print()
            tagger = SRTaggerBottomUp(trees, add_remove_top=True)
            print(tagger.tree_to_tags_pipeline(trees[0]))
            print(tagger.tree_to_ids_pipeline(trees[0]))

    def test_tagging_tetra(self):
        langs = ['Basque', 'French', 'German', 'Hebrew', 'Hungarian', 'Korean', 'Polish',
                 'Swedish']
        for l in langs:
            READER = BracketParseCorpusReader('../data/spmrl/',
                                              [l + '.train', l + '.dev', l + '.test'])
            trees = READER.parsed_sents(l + '.test')
            trees[0].pretty_print()
            tagger = BottomUpTetratagger(trees, add_remove_top=True)
            print(tagger.tree_to_tags_pipeline(trees[0]))
            print(tagger.tree_to_ids_pipeline(trees[0]))

    def test_taggers(self):
        tagger = SRTaggerBottomUp(add_remove_top=False)
        langs = ['Basque', 'French', 'German', 'Hebrew', 'Hungarian', 'Korean', 'Polish',
                 'Swedish']
        for l in tq(langs):
            READER = BracketParseCorpusReader('../data/spmrl/',
                                              [l + '.train', l + '.dev', l + '.test'])
            trees = READER.parsed_sents(l + '.test')
            for tree in tq(trees):
                tags, _ = tagger.tree_to_tags_pipeline(tree)
                tree_back = tagger.tags_to_tree_pipeline(tags, tree.pos())
                self.assertEqual(tree, tree_back)

    def test_korean(self):
        import pickle
        with open("../data/vocab/Korean-bu-sr.pkl", 'rb') as f:
            tag_vocab = pickle.load(f)
        tagger = SRTaggerBottomUp(tag_vocab=tag_vocab, add_remove_top=False)
        l = "Korean"
        READER = BracketParseCorpusReader('../data/spmrl/',
                                          [l + '.train', l + '.dev', l + '.test'])
        trees = READER.parsed_sents(l + '.test')
        for tree in tq(trees):
            tags = tagger.tree_to_tags_pipeline(tree)[0]
            tree_back = tagger.tags_to_tree_pipeline(tags, tree.pos())
            self.assertEqual(tree, tree_back)


class TestStackSize(unittest.TestCase):
    def test_english_tetra(self):
        l = "English"
        READER = BracketParseCorpusReader('../data/spmrl',
                                          [l + '.train', l + '.dev', l + '.test'])
        trees = READER.parsed_sents([l + '.train', l + '.dev', l + '.test'])
        tagger = BottomUpTetratagger(add_remove_top=True)
        stack_size_list = []
        for tree in tq(trees):
            tags, max_depth = tagger.tree_to_tags_pipeline(tree)
            stack_size_list.append(max_depth)

        print(stack_size_list)


class TestDependencyTree(unittest.TestCase):
    def test_predictions(self):
        import pickle
        with open("../data/logit_preds_sample.pkl", "rb") as f:
            predictions = pickle.load(f)
        with open('../data/vocab/English-hexa.pkl', 'rb') as f:
            tag_vocab = pickle.load(f)

        READER = BracketParseCorpusReader('../data/ptb',
                                          ['English.train', 'English.dev', 'English.test'])
        tag_system = HexaTagger(tag_vocab=tag_vocab,
                                trees=READER.parsed_sents('English.train'))

        from run import prepare_test_data
        tagging_schema = "hexa"
        bert_model_path = "../bertlarge"
        lang = "English"
        batch_size = 16
        eval_dataset, eval_dataloader = prepare_test_data(READER,
                                                          tag_system, tagging_schema,
                                                          bert_model_path,
                                                          batch_size,
                                                          lang)

        n = len(eval_dataset)
        eval_labels = np.zeros((n, 256), dtype=int)
        idx = 0

        for batch in tq(eval_dataloader):
            if idx * 16 >= n:
                break
            labels = batch['labels']
            eval_labels[idx * 16:(idx + 1) * 16, :] = labels.cpu().numpy()
            idx += 1

        for i in tq(range(predictions.shape[0])):
            if len(eval_dataset.trees[i].leaves()) > 10:
                continue
            logits = predictions[i]
            is_word = eval_labels[i] != 0
            temp_tree = eval_dataset.trees[i].copy(deep=True)
            tetratagger = TopDownTetratagger()
            print(tetratagger.tree_to_tags_pipeline(temp_tree))
            original_tree = eval_dataset.trees[i]

            input_seq = [(word, "") for (word, pos) in original_tree.pos()]
            tags = tag_system.tree_to_tags_pipeline(original_tree)[0]
            print()
            print(f"gold tags {tags}")
            print(logits.shape)
            tree = tag_system.logits_to_tree(logits, input_seq, mask=is_word,
                                         max_depth=10, keep_per_depth=5, is_greedy=False)
            tree.pretty_print()

    def test_lex_tree(self):
        l = "English"
        split = "train"
        reader = BracketParseCorpusReader('../data/ptb/new', [f'{l}.lex.{split}'])
        trees = reader.parsed_sents([f'{l}.lex.{split}'])

        tagger = HexaTagger()
        depths = []

        for tree in tq(trees):
            original_tree = tree.copy(deep=True)
            # original_tree.pretty_print()
            output = tagger.tree_to_tags_pipeline(tree)
            tags = output[0]
            # print(tags)
            depths.append(output[1])
            # input_seq = [(leaf.split("^^^")[0], "") for leaf in tree.leaves()]
            tree_back = tagger.tags_to_tree_pipeline(tags, tree.pos())
            self.assertEqual(original_tree, tree_back)
            # tagger.tags_to_tree_pipeline()
            # tree.collapse_unary(collapsePOS=True, collapseRoot=True)
            # tree.pretty_print()
            # relabel_tree(tree)
            # relabeled_tree = tree.copy(deep=True)
            # if len(relabeled_tree) == 1:
            #     continue
            # tree.pretty_print()
            # new_root = Tree(tree.label(), [])
            # new_root = binarize_lex_tree(tree, new_root, "X")
            # new_root.pretty_print()
            #
            # ptree = ParentedTree.convert(new_root)
            # root_label = ptree.label()
            # tree_lc = ParentedTree(root_label, [])
            # LeftCornerTransformer.transform(tree_lc, ptree, ptree)
            #
            # print(tagger.tree_to_tags(tree_lc))
            # break
            # root_back = Tree(new_root.label(), [])
            # debinarize_lex_tree(new_root, root_back)
            # root_back.pretty_print()
            # self.assertEqual(relabeled_tree, root_back)
        print(max(depths))

    def test_write_to_file(self):
        l = "English"
        split = "train"
        reader = BracketParseCorpusReader('../data/ptb/new', [f'{l}.lex.{split}'])
        trees = reader.parsed_sents([f'{l}.lex.{split}'])
        simplified_trees = []
        for tree in tq(trees):
            tree.collapse_unary(collapsePOS=True, collapseRoot=True)
            relabel_tree(tree)
            simplified_trees.append(tree)
        with open(f'../data/ptb/{l}.lex-simple.{split}', 'w') as f:
            for tree in simplified_trees:
                f.write(tree._pformat_flat("", "()", False) + "\n")


if __name__ == '__main__':
    unittest.main()
