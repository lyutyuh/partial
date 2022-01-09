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
            keep_per_depth=1,
            initial_label=None,
    ):
        # Save parameters
        self.stack_depth_change_by_id = stack_depth_change_by_id
        self.valid_depths = np.arange(1, max_depth)
        self.keep_per_depth = keep_per_depth

        # Initialize the beam
        scores = np.zeros(1, dtype=float)
        stack_depths = np.full(1, initial_stack_depth)
        prev = backptrs = labels = None
        if initial_label is not None:
            labels = np.full(1, initial_label)
        self.beam = Beam(scores, stack_depths, prev, backptrs, labels)

    def advance(self, label_logits):
        label_log_probs = label_logits

        all_new_scores = self.beam.scores[:, None] + label_log_probs

        all_new_stack_depths = (
                self.beam.stack_depths[:, None]
                + self.stack_depth_change_by_id[None, :]
        )

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
