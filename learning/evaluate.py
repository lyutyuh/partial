import torch
import numpy as np
from tqdm import tqdm as tq

import logging


def report_eval_loss(model, eval_dataloader, device, n_iter, writer):
    loss = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss.append(torch.mean(outputs[0]).cpu())

    mean_loss = np.mean(loss)
    logging.info("Eval Loss: {}".format(mean_loss))
    if writer is not None:
        writer.add_scalar('eval_loss', mean_loss, n_iter)
    return mean_loss


def predict(model, eval_dataloader, dataset_size, num_tags, device):
    model.eval()
    predictions = np.zeros((dataset_size, 256, num_tags))
    eval_labels = np.zeros((dataset_size, 256), dtype=int)
    idx = 0
    for batch in tq(eval_dataloader):
        if idx * 16 >= dataset_size:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs[1]
        predictions[idx * 16:(idx + 1) * 16, :, :] = logits.cpu().numpy()
        labels = batch['labels']
        eval_labels[idx * 16:(idx + 1) * 16, :] = labels.cpu().numpy()
        idx += 1

    return predictions, eval_labels


def calc_tag_accuracy(predictions, eval_labels, num_leaf_labels, writer, use_tensorboard):
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


def calc_parse_eval(predictions, eval_labels, eval_dataset, tag_system, output_path,
                    model_name, max_depth):
    predicted_dev_trees = []
    gold_dev_trees = []
    c_err = 0
    for i in tq(range(predictions.shape[0])):
        logits = predictions[i]
        is_word = eval_labels[i] != 0
        original_tree = eval_dataset.trees[i]
        try:  # ignore the ones that failed in unchomsky_normal_form
            tree = tag_system.logits_to_tree(logits, original_tree.pos(), mask=is_word, max_depth=max_depth)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
            c_err += 1
            continue
        if tree.leaves() != original_tree.leaves():
            c_err += 1
            continue
        predicted_dev_trees.append(tree)
        gold_dev_trees.append(original_tree)
    logging.warning("Number of binarization error: {}".format(c_err))
    save_predictions(predicted_dev_trees, output_path + model_name + "_predictions.txt")
    save_predictions(gold_dev_trees, output_path + model_name + "_gold.txt")


def save_predictions(predicted_trees, file_path):
    with open(file_path, 'w') as f:
        for tree in predicted_trees:
            f.write(' '.join(str(tree).split()) + '\n')
