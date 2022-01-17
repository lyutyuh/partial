## PUBLIC API: code from: https://github.com/nikitakit/tetra-tagging

import numpy as np


class Beam:
    def __init__(self, scores, stack_depths, prev, backptrs, labels):
        self.scores = scores
        self.stack_depths = stack_depths
        self.prev = prev
        self.backptrs = backptrs
        self.labels = labels


class BeamSearch:
    def __init__(
            self,
            initial_stack_depth,
            stack_depth_change_by_id,
            max_depth=12,
            min_depth=1,
            keep_per_depth=1,
            stack_depth_change_by_id_l2=None,
            stack_depth_change_by_id_after=None,
            crf_transitions=None,
            initial_label=None,
    ):
        # Save parameters
        self.stack_depth_change_by_id = stack_depth_change_by_id
        self.stack_depth_change_by_id_l2 = stack_depth_change_by_id_l2
        self.stack_depth_change_by_id_after = stack_depth_change_by_id_after
        self.valid_depths = np.arange(min_depth, max_depth)
        self.keep_per_depth = keep_per_depth
        self.max_depth = max_depth
        self.crf_transitions = crf_transitions

        # Initialize the beam
        scores = np.zeros(1, dtype=float)
        stack_depths = np.full(1, initial_stack_depth)
        prev = backptrs = labels = None
        if initial_label is not None:
            labels = np.full(1, initial_label)
        self.beam = Beam(scores, stack_depths, prev, backptrs, labels)

    def compute_new_scores(self, label_log_probs, is_last):
        if self.crf_transitions is None:
            return self.beam.scores[:, None] + label_log_probs
        else:
            if self.beam.labels is not None:
                all_new_scores = self.beam.scores[:, None] + label_log_probs + \
                                 self.crf_transitions["transitions"][self.beam.labels]
            else:
                all_new_scores = self.beam.scores[:, None] + label_log_probs + \
                                 self.crf_transitions["start_transitions"]
            if is_last:
                all_new_scores += self.crf_transitions["end_transitions"]
            return all_new_scores

    # This extra mask layer takes care of invalid reduce actions when there is not an empty
    # slot in the tree, which is needed in the top-down shift reduce tagging schema
    def extra_mask_layer(self, all_new_scores, all_new_stack_depths):
        depth_mask = np.zeros(all_new_stack_depths.shape)
        depth_mask[all_new_stack_depths < 0] = -np.inf
        depth_mask[all_new_stack_depths > self.max_depth] = -np.inf
        all_new_scores = all_new_scores + depth_mask

        all_new_stack_depths = (
                all_new_stack_depths
                + self.stack_depth_change_by_id
        )
        return all_new_scores, all_new_stack_depths

    def after_mask(self, all_new_scores, all_new_stack_depths):
        after_stack_depths = (
                all_new_stack_depths
                + self.stack_depth_change_by_id_after
        )
        depth_mask = np.zeros(all_new_stack_depths.shape)
        depth_mask[after_stack_depths < 1] = -np.inf
        all_new_scores = all_new_scores + depth_mask
        return all_new_scores

    def advance(self, label_logits, is_last=False):
        label_log_probs = label_logits

        all_new_scores = self.compute_new_scores(label_log_probs, is_last)

        if self.stack_depth_change_by_id_l2 is not None:
            all_new_stack_depths = (
                    self.beam.stack_depths[:, None]
                    + self.stack_depth_change_by_id_l2[None, :]
            )
            all_new_scores, all_new_stack_depths = self.extra_mask_layer(all_new_scores,
                                                                         all_new_stack_depths)
        else:
            all_new_stack_depths = (
                    self.beam.stack_depths[:, None]
                    + self.stack_depth_change_by_id[None, :]
            )

        if self.stack_depth_change_by_id_after is not None:
            all_new_scores = self.after_mask(all_new_scores, all_new_stack_depths)

        masked_scores = all_new_scores[None, :, :] + np.where(
            all_new_stack_depths[None, :, :]
            == self.valid_depths[:, None, None],
            0.0,
            -np.inf,
        )
        masked_scores = masked_scores.reshape(self.valid_depths.shape[0], -1)
        idxs = np.argsort(-masked_scores)[:, : self.keep_per_depth].flatten()
        backptrs, labels = np.unravel_index(idxs, all_new_scores.shape)

        transition_valid = all_new_stack_depths[
                               backptrs, labels
                           ] == self.valid_depths.repeat(self.keep_per_depth)

        backptrs = backptrs[transition_valid]
        labels = labels[transition_valid]

        self.beam = Beam(
            all_new_scores[backptrs, labels],
            all_new_stack_depths[backptrs, labels],
            self.beam,
            backptrs,
            labels,
        )

    def get_path(self, idx=0, required_stack_depth=1):
        if required_stack_depth is not None:
            assert self.beam.stack_depths[idx] == required_stack_depth
        score = self.beam.scores[idx]
        assert score > -np.inf

        beam = self.beam
        label_idxs = []
        while beam.prev is not None:
            label_idxs.insert(0, beam.labels[idx])
            idx = beam.backptrs[idx]
            beam = beam.prev

        return score, label_idxs
