import logging
import math
import os.path
import re
import subprocess
import tempfile
import time
from typing import List, Tuple
import random
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm as tq

from tagging.tree_tools import create_dummy_tree
from sklearn.metrics import precision_recall_fscore_support


class ParseMetrics(object):
    def __init__(self, recall, precision, fscore, complete_match, tagging_accuracy=100):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore
        self.complete_match = complete_match
        self.tagging_accuracy = tagging_accuracy

    def __str__(self):
        if self.tagging_accuracy < 100:
            return "(Recall={:.4f}, Precision={:.4f}, ParseMetrics={:.4f}, CompleteMatch={:.4f}, TaggingAccuracy={:.4f})".format(
                self.recall, self.precision, self.fscore, self.complete_match, self.tagging_accuracy)
        else:
            return "(Recall={:.4f}, Precision={:.4f}, ParseMetrics={:.4f}, CompleteMatch={:.4f})".format(
                self.recall, self.precision, self.fscore, self.complete_match)


def report_eval_loss(model, eval_dataloader, device, n_iter, writer) -> np.ndarray:
    loss = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            outputs = model(**batch)
            loss.append(torch.mean(outputs[0]).cpu())

    mean_loss = np.mean(loss)
    logging.info("Eval Loss: {}".format(mean_loss))
    if writer is not None:
        writer.add_scalar('eval_loss', mean_loss, n_iter)
    return mean_loss


def predict_partial_order(
    model, eval_dataloader, dataset_size, batch_size, device
) -> Tuple[np.array, np.array]:
    model.eval()
    predictions = []
    eval_labels = []
    max_len = 0

    for batch in tq(eval_dataloader, disable=True):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=True, dtype=torch.bfloat16
        ):
            outputs = model(**batch)

        arc_logits, rel_logits = outputs[1]
        arc_logits, rel_logits = arc_logits.float().cpu().numpy(), rel_logits.float().cpu().numpy()
        max_len = max(max_len, arc_logits.shape[1])

        predictions.append((arc_logits, rel_logits))

        head_labels = batch['head_labels'].int().cpu().numpy()
        rel_labels = batch['rel_labels'].int().cpu().numpy()
        eval_labels.append((head_labels, rel_labels))

    # shape: (num_samples, max_len, max_len), (num_samples, max_len, num_labels)
    predictions = ( # arc_logits, rel_logits
        np.concatenate([np.pad(logits[0], ((0, 0), (0, max_len - logits[0].shape[1]), (0, max_len - logits[0].shape[1])), 'constant', constant_values=-1e6) for logits in predictions], axis=0),
        np.concatenate([np.pad(logits[1], ((0, 0), (0, max_len - logits[1].shape[1]), (0, 0)), 'constant', constant_values=-1e6) for logits in predictions], axis=0)
    )
    eval_labels = ( # -1 for padding
        np.concatenate([np.pad(labels[0], ((0, 0), (0, max_len - labels[0].shape[1])), 'constant', constant_values=-1) for labels in eval_labels], axis=0),
        np.concatenate([np.pad(labels[1], ((0, 0), (0, max_len - labels[1].shape[1])), 'constant', constant_values=-1) for labels in eval_labels], axis=0)
    )

    return predictions, eval_labels


def partial_order_dependency_eval(
    predictions, eval_labels, eval_dataset, output_path,
    model_name, max_depth, keep_per_depth, is_greedy
) -> ParseMetrics:
    predicted_dev_triples, predicted_dev_triples_unlabeled = [], []
    gold_dev_triples, gold_dev_triples_unlabeled = [], []
    c_err = 0

    rev_rel_dict = {v: k for k, v in eval_dataset.rel_dict.items()}
    for i in tq(range(predictions[0].shape[0]), disable=True):
        arc_logits, rel_logits = predictions[0][i], predictions[1][i]
        is_word = (eval_labels[0][i] != -1)

        original_tree = deepcopy(eval_dataset.trees[i])

        # list of (head, tail, label)
        gt_triples = dep_graph_to_dev_triples(original_tree)
        pred_triples = logits_to_dev_triples(arc_logits, rel_logits, original_tree, rev_rel_dict)

        assert len(gt_triples) == len(pred_triples), f"wrong length {len(gt_triples)} vs. {len(pred_triples)}!"

        for x, y in zip(sorted(gt_triples), sorted(pred_triples)):
            if is_punctuation(x[3]):
                # ignoring punctuations for evaluation
                continue
            assert x[0] == y[0], f"wrong tree {gt_triples} vs. {pred_triples}!"
            gold_dev_triples.append(f"{x[0]}-{x[1]}-{x[2].split(':')[0]}")
            gold_dev_triples_unlabeled.append(f"{x[0]}-{x[1]}")

            predicted_dev_triples.append(f"{y[0]}-{y[1]}-{y[2].split(':')[0]}")
            predicted_dev_triples_unlabeled.append(f"{y[0]}-{y[1]}")

    logging.warning("Number of binarization error: {}\n".format(c_err))
    las_recall, las_precision, las_fscore, _ = precision_recall_fscore_support(
        gold_dev_triples, predicted_dev_triples, average='micro'
    )
    uas_recall, uas_precision, uas_fscore, _ = precision_recall_fscore_support(
        gold_dev_triples_unlabeled, predicted_dev_triples_unlabeled, average='micro'
    )

    return (ParseMetrics(las_recall, las_precision, las_fscore, complete_match=1),
            ParseMetrics(uas_recall, uas_precision, uas_fscore, complete_match=1))


def predict(
    model, eval_dataloader, dataset_size, num_tags, batch_size, device
) -> Tuple[np.array, np.array]:
    model.eval()
    predictions = []
    eval_labels = []
    max_len = 0

    for batch in tq(eval_dataloader, disable=True):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=True, dtype=torch.bfloat16
        ):
            outputs = model(**batch)

        logits = outputs[1].float().cpu().numpy()
        max_len = max(max_len, logits.shape[1])
        predictions.append(logits)
        labels = batch['labels'].int().cpu().numpy()
        eval_labels.append(labels)

    # shape: (num_samples, max_len, num_tags)
    predictions = np.concatenate([np.pad(logits, ((0, 0), (0, max_len - logits.shape[1]), (0, 0)), 'constant', constant_values=0) for logits in predictions], axis=0)
    eval_labels = np.concatenate([np.pad(labels, ((0, 0), (0, max_len - labels.shape[1])), 'constant', constant_values=0) for labels in eval_labels], axis=0)

    return predictions, eval_labels


def calc_tag_accuracy(
    predictions, eval_labels, num_leaf_labels, writer, use_tensorboard
) -> Tuple[float, float]:
    even_predictions = predictions[..., -num_leaf_labels:]
    odd_predictions = predictions[..., :-num_leaf_labels]
    even_labels = eval_labels % (num_leaf_labels + 1) - 1
    odd_labels = eval_labels // (num_leaf_labels + 1) - 1

    odd_predictions = odd_predictions[odd_labels != -1].argmax(-1)
    even_predictions = even_predictions[even_labels != -1].argmax(-1)

    odd_labels = odd_labels[odd_labels != -1]
    even_labels = even_labels[even_labels != -1]

    odd_acc = (odd_predictions == odd_labels).mean()
    even_acc = (even_predictions == even_labels).mean()

    logging.info('odd_tags_accuracy: {}'.format(odd_acc))
    logging.info('even_tags_accuracy: {}'.format(even_acc))

    if use_tensorboard:
        writer.add_pr_curve('odd_tags_pr_curve', odd_labels, odd_predictions, 0)
        writer.add_pr_curve('even_tags_pr_curve', even_labels, even_predictions, 1)
    return even_acc, odd_acc


def get_dependency_from_lexicalized_tree(lex_tree, triple_dict, offset=0):
    # this recursion assumes projectivity
    # Input:
    #     root of lex-tree
    # Output:
    #     the global index of the dependency root
    if type(lex_tree) not in {str, dict} and len(lex_tree) == 1:
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

def is_punctuation(pos):
    punct_set = '.' '``' "''" ':' ','
    return (pos in punct_set) or (pos in ['PU', 'PUNCT']) # for chinese


def logits_to_dev_triples(arc_logits, rel_logits, tree, rev_rel_dict):
    triples = []
    for i, x in sorted(tree.nodes.items()):
        if x['word'] is None:
            continue
        head = arc_logits[i-1].argmax()
        label = rev_rel_dict[rel_logits[i-1].argmax()]
        tail, pos = i, x['tag']

        triples.append((tail, head, label, pos))
    return triples


def dep_graph_to_dev_triples(tree):
    # Input:
    #     a dependency tree
    # Output:
    #     a list of (head, tail, label) triples
    triples = []
    for i, x in sorted(tree.nodes.items()):
        if x['word'] is None:
            continue
        head, label, tail, pos = x['head'], x['rel'], i, x['tag']
        triples.append((tail, head, label, pos))
    return triples


def tree_to_dep_triples(lex_tree):
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
            lex_tree.pos()[head][1].split("^^^")[1].split("+")[0],
            lex_tree.pos()[head][1].split("^^^")[1].split("+")[1]
        ))
    return dep_triples


def dependency_eval(
    predictions, eval_labels, eval_dataset, tag_system, output_path,
    model_name, max_depth, keep_per_depth, is_greedy
) -> ParseMetrics:
    predicted_dev_triples, predicted_dev_triples_unlabeled = [], []
    gold_dev_triples, gold_dev_triples_unlabeled = [], []
    c_err = 0
    for i in tq(range(predictions.shape[0]), disable=True):
        logits = predictions[i]
        is_word = (eval_labels[i] != 0)

        original_tree = deepcopy(eval_dataset.trees[i])
        original_tree.collapse_unary(collapsePOS=True, collapseRoot=True)

        try:  # ignore the ones that failed in unchomsky_normal_form
            tree = tag_system.logits_to_tree(
                logits, original_tree.pos(),
                mask=is_word,
                max_depth=max_depth,
                keep_per_depth=keep_per_depth,
                is_greedy=is_greedy
            )
            tree.collapse_unary(collapsePOS=True, collapseRoot=True)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            c_err += 1
            predicted_dev_triples.append(create_dummy_tree(original_tree.pos()))
            continue
        if tree.leaves() != original_tree.leaves():
            c_err += 1
            predicted_dev_triples.append(create_dummy_tree(original_tree.pos()))
            continue

        gt_triples = tree_to_dep_triples(original_tree)
        pred_triples = tree_to_dep_triples(tree)
        assert len(gt_triples) == len(pred_triples), f"wrong length {len(gt_triples)} vs. {len(pred_triples)}!"

        for x, y in zip(sorted(gt_triples), sorted(pred_triples)):
            if is_punctuation(x[3]):
                # ignoring punctuations for evaluation
                continue
            assert x[0] == y[0], f"wrong tree {gt_triples} vs. {pred_triples}!"
            gold_dev_triples.append(f"{x[0]}-{x[1]}-{x[2].split(':')[0]}")
            gold_dev_triples_unlabeled.append(f"{x[0]}-{x[1]}")

            predicted_dev_triples.append(f"{y[0]}-{y[1]}-{y[2].split(':')[0]}")
            predicted_dev_triples_unlabeled.append(f"{y[0]}-{y[1]}")

    logging.warning("Number of binarization error: {}\n".format(c_err))
    las_recall, las_precision, las_fscore, _ = precision_recall_fscore_support(
        gold_dev_triples, predicted_dev_triples, average='micro'
    )
    uas_recall, uas_precision, uas_fscore, _ = precision_recall_fscore_support(
        gold_dev_triples_unlabeled, predicted_dev_triples_unlabeled, average='micro'
    )

    return (ParseMetrics(las_recall, las_precision, las_fscore, complete_match=1),
            ParseMetrics(uas_recall, uas_precision, uas_fscore, complete_match=1))



def calc_parse_eval(predictions, eval_labels, eval_dataset, tag_system, output_path,
                    model_name, max_depth, keep_per_depth, is_greedy) -> ParseMetrics:
    predicted_dev_trees = []
    gold_dev_trees = []
    c_err = 0
    for i in tq(range(predictions.shape[0])):
        logits = predictions[i]
        is_word = eval_labels[i] != 0
        original_tree = eval_dataset.trees[i]
        gold_dev_trees.append(original_tree)
        try:  # ignore the ones that failed in unchomsky_normal_form
            tree = tag_system.logits_to_tree(logits, original_tree.pos(), mask=is_word,
                                             max_depth=max_depth, keep_per_depth=keep_per_depth, is_greedy=is_greedy)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)

            c_err += 1
            predicted_dev_trees.append(create_dummy_tree(original_tree.pos()))
            continue
        if tree.leaves() != original_tree.leaves():
            c_err += 1
            predicted_dev_trees.append(create_dummy_tree(original_tree.pos()))
            continue
        predicted_dev_trees.append(tree)

    logging.warning("Number of binarization error: {}".format(c_err))

    return evalb("EVALB_SPMRL/", gold_dev_trees, predicted_dev_trees)


def save_predictions(predicted_trees, file_path):
    with open(file_path, 'w') as f:
        for tree in predicted_trees:
            f.write(' '.join(str(tree).split()) + '\n')


def evalb(evalb_dir, gold_trees, predicted_trees, ref_gold_path=None) -> ParseMetrics:
    # Code from: https://github.com/nikitakit/self-attentive-parser/blob/master/src/evaluate.py
    assert os.path.exists(evalb_dir)
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    evalb_spmrl_program_path = os.path.join(evalb_dir, "evalb_spmrl")
    assert os.path.exists(evalb_program_path) or os.path.exists(evalb_spmrl_program_path)

    if os.path.exists(evalb_program_path):
        evalb_param_path = os.path.join(evalb_dir, "nk.prm")
    else:
        evalb_program_path = evalb_spmrl_program_path
        evalb_param_path = os.path.join(evalb_dir, "spmrl.prm")

    assert os.path.exists(evalb_program_path)
    assert os.path.exists(evalb_param_path)

    assert len(gold_trees) == len(predicted_trees)
    for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
        gold_leaves = list(gold_tree.leaves())
        predicted_leaves = list(predicted_tree.leaves())

    temp_dir = tempfile.TemporaryDirectory(prefix="evalb-")
    gold_path = os.path.join(temp_dir.name, "gold.txt")
    predicted_path = os.path.join(temp_dir.name, "predicted.txt")
    output_path = os.path.join(temp_dir.name, "output.txt")

    with open(gold_path, "w") as outfile:
        if ref_gold_path is None:
            for tree in gold_trees:
                outfile.write(' '.join(str(tree).split()) + '\n')
        else:
            # For the SPMRL dataset our data loader performs some modifications
            # (like stripping morphological features), so we compare to the
            # raw gold file to be certain that we haven't spoiled the evaluation
            # in some way.
            with open(ref_gold_path) as goldfile:
                outfile.write(goldfile.read())

    with open(predicted_path, "w") as outfile:
        for tree in predicted_trees:
            outfile.write(' '.join(str(tree).split()) + '\n')

    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        gold_path,
        predicted_path,
        output_path,
    )
    subprocess.run(command, shell=True)

    fscore = ParseMetrics(math.nan, math.nan, math.nan, math.nan)
    with open(output_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.fscore = float(match.group(1))
            match = re.match(r"Complete match\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.complete_match = float(match.group(1))
            match = re.match(r"Tagging accuracy\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.tagging_accuracy = float(match.group(1))
                break

    success = (
            not math.isnan(fscore.fscore) or
            fscore.recall == 0.0 or
            fscore.precision == 0.0)

    if success:
        temp_dir.cleanup()
    else:
        print("Error reading EVALB results.")
        print("Gold path: {}".format(gold_path))
        print("Predicted path: {}".format(predicted_path))
        print("Output path: {}".format(output_path))

    return fscore
