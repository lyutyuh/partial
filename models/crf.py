import torch
import torch.nn as nn

MINUS_INF = -1e9


class CRF(nn.Module):
    """
    Linear-chain Conditional Random Field (CRF).
    Args:
        nb_labels (int): number of labels in your tagset, including special symbols.
        max_depth (int): max depth of the stack.
        bos_tag_id (int): integer representing the beginning of sentence symbol in
            your tagset.
        eos_tag_id (int): integer representing the end of sentence symbol in your tagset.
        pad_tag_id (int, optional): integer representing the pad symbol in your tagset.
            If None, the model will treat the PAD as a normal tag. Otherwise, the model
            will apply constraints for PAD transitions.
        batch_first (bool): Whether the first dimension represents the batch dimension.
    """

    def __init__(
            self, nb_labels, max_depth, bos_tag_id, eos_tag_id, tag_system, device, pad_tag_id=None,
            batch_first=True
    ):
        super().__init__()

        self.nb_labels = nb_labels
        self.max_depth = max_depth
        self.extra_label_size = 3
        self.state_size = nb_labels * max_depth + self.extra_label_size  # 1(bos)+1(eos)+1(padd)
        self.BOS_TAG_ID = bos_tag_id
        self.EOS_TAG_ID = eos_tag_id
        self.PAD_TAG_ID = pad_tag_id
        self.batch_first = batch_first
        self.tag_system = tag_system
        self.device = device

        self.transitions = nn.Parameter(torch.empty(self.state_size, self.state_size)).to(
            device)
        self.init_weights()

    def _get_index_range_by_depth(self, d):
        first_idx = d * self.nb_labels + self.extra_label_size
        last_idx = (d + 1) * self.nb_labels + self.extra_label_size
        return first_idx, last_idx

    def init_weights(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        # enforce contraints (rows=from, columns=to) with a big negative number
        # no transitions allowed to the beginning of sentence
        self.transitions.data[:, self.BOS_TAG_ID] = MINUS_INF
        # no transition alloed from the end of sentence
        self.transitions.data[self.EOS_TAG_ID, :] = MINUS_INF
        # no transitions allowed from the beginning to the length > 1
        self.transitions.data[self.BOS_TAG_ID,
        self.nb_labels + self.extra_label_size:] = MINUS_INF
        # no transitions allowed from the length > 1 to end
        self.transitions.data[self.nb_labels + self.extra_label_size:,
        self.EOS_TAG_ID] = MINUS_INF

        # the depth of the stack can change only one unit at each transition
        for d in range(self.max_depth):
            first_idx, last_idx = self._get_index_range_by_depth(d)
            self.transitions.data[first_idx:last_idx, first_idx:last_idx] = MINUS_INF
            if d > 0:
                d_m1_first_idx, _ = self._get_index_range_by_depth(d - 1)
                self.transitions.data[first_idx:last_idx, :d_m1_first_idx] = MINUS_INF
            if d < self.max_depth - 1:
                _, d_p1_last_idx = self._get_index_range_by_depth(d + 1)
                self.transitions.data[first_idx:last_idx, d_p1_last_idx:] = MINUS_INF

        if self.PAD_TAG_ID is not None:
            # no transitions from padding
            self.transitions.data[self.PAD_TAG_ID, :] = MINUS_INF
            self.transitions.data[self.nb_labels + self.extra_label_size:,
            self.PAD_TAG_ID] = MINUS_INF
            # except if the end of sentence is reached
            # or we are already in a pad position
            self.transitions.data[self.PAD_TAG_ID, self.EOS_TAG_ID] = 0.0
            self.transitions.data[self.PAD_TAG_ID, self.PAD_TAG_ID] = 0.0

    def tag_to_state(self, cur_state, tag_id):
        cur_depth = self.state_to_depth(cur_state)
        if tag_id < 1:
            return self.PAD_TAG_ID
        tag = self.tag_system.tag_vocab[tag_id - 1]
        sr = tag.split("/")[0]
        idx = tag.find("/")
        if idx == -1:
            label = ""
        else:
            label = tag[idx + 1:]
        if sr == 's':
            depth = cur_depth + 1
            if depth >= self.max_depth:  # TODO: hacky way to mark it is invalid
                return self.BOS_TAG_ID
        else:
            depth = cur_depth - 1
            if depth == -1:  # TODO: hacky way to mark it is invalid
                return self.BOS_TAG_ID

        return int(depth * self.nb_labels + self.tag_system.label_vocab.index(
            label) + self.extra_label_size)

    def state_to_depth(self, state):
        if state < self.extra_label_size:
            return -1
        else:
            return int((state - self.extra_label_size) / self.nb_labels)

    def state_to_label(self, cur_state):
        if cur_state < self.extra_label_size:
            return "ERR"
        else:
            return self.tag_system.label_vocab[
                int((cur_state - self.extra_label_size) % self.nb_labels)]

    def states_to_tag(self, s, t):
        s_depth = self.state_to_depth(s)
        t_depth = self.state_to_depth(t)
        t_label = self.state_to_label(t)
        if abs(s_depth - t_depth) != -1 or t_label == "ERR":
            return -1  # ERR
        elif s_depth - t_depth == 1:
            tag = "s" if t_label == "" else "s/" + t_label
            return self.tag_system.tag_vocab.index(tag)
        else:
            tag = "r" if t_label == "" else "r/" + t_label
            return self.tag_system.tag_vocab.index(tag)

    def emm_to_state(self, cur_state, emm):
        new_emm = torch.full((self.transitions.shape[1],), MINUS_INF).to(self.device)
        for i in range(emm.shape[0]):
            if i < 1:
                new_emm[self.PAD_TAG_ID] = emm[i]
            else:
                new_emm[self.tag_to_state(cur_state, i)] = emm[
                    i]  # TODO: what happens if the state is invalid?
        return new_emm

    def forward(self, emissions, tags, mask=None):
        """Compute the negative log-likelihood. See `log_likelihood` method."""
        nll = -self.log_likelihood(emissions, tags, mask=mask)
        return nll

    def log_likelihood(self, emissions, tags, mask=None):
        """Compute the probability of a sequence of tags given a sequence of
        emissions scores.

        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape of (batch_size, seq_len, nb_labels) if batch_first is True,
                (seq_len, batch_size, nb_labels) otherwise.
            tags (torch.LongTensor): Sequence of labels.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.
      mn       mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape of (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.

        Returns:
            torch.Tensor: the log-likelihoods for each sequence in the batch.
                Shape of (batch_size,)
        """

        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float).to(self.device)

        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)
        return torch.sum(scores - partition)

    def _compute_scores(self, emissions, tags, mask):
        """Compute the scores for a given batch of emissions with their tags.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_labels)
            tags (Torch.LongTensor): (batch_size, seq_len)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: Scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length = tags.shape
        scores = torch.zeros(batch_size).to(self.device)

        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()

        curr_state = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        next_state = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        for i, ftag in enumerate(first_tags):
            next_state[i] = self.tag_to_state(curr_state[i], ftag)

        curr_state = next_state

        t_scores = self.transitions[self.BOS_TAG_ID, curr_state]
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()
        scores += e_scores + t_scores

        for i in range(1, seq_length):
            is_valid = mask[:, i]

            current_tags = tags[:, i]
            next_state = torch.zeros(batch_size, dtype=torch.long)

            for i, tag in enumerate(current_tags):
                next_state[i] = self.tag_to_state(curr_state[i], tag)

            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self.transitions[curr_state, next_state]
            curr_state = next_state

            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid
            scores += e_scores + t_scores

        next_state = torch.zeros(batch_size, dtype=torch.long)
        for i, ltag in enumerate(last_tags):
            next_state[i] = self.tag_to_state(curr_state[i], ltag)
        scores += self.transitions[next_state, self.EOS_TAG_ID]

        return scores

    def _compute_log_partition(self, emissions, mask):
        """Compute the partition function in log-space using the forward-algorithm.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_tags)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: the partition scores for each batch.
                Shape of (batch_size,)
        """
        batch_size, seq_length, nb_tags = emissions.shape
        end_scores = torch.full((batch_size, self.state_size), MINUS_INF).to(self.device)

        for b in range(batch_size):
            alphas = self.transitions[self.BOS_TAG_ID, :] + self.emm_to_state(self.BOS_TAG_ID,
                                                                              emissions[b, 0])

            for i in range(1, seq_length):
                for s in range(self.state_size):

                    e_scores = self.emm_to_state(s, emissions[b, i])
                    t_scores = self.transitions[b, s]
                    a_scores = alphas[s]

                    scores = e_scores + t_scores + a_scores

                    if s == 0:
                        new_alphas = scores
                    else:
                        new_alphas = torch.logsumexp(
                            torch.cat((new_alphas.unsqueeze(0), scores.unsqueeze(0))), dim=0)

                is_valid = int(mask[b, i])
                alphas = is_valid * new_alphas + (1 - is_valid) * alphas

            last_transition = self.transitions[:, self.EOS_TAG_ID]
            end_scores[b] = alphas + last_transition.unsqueeze(0)

        return torch.logsumexp(end_scores, dim=1)

    def decode(self, emissions, mask=None):
        """Find the most probable sequence of labels given the emissions using
        the Viterbi algorithm.
        Args:
            emissions (torch.Tensor): Sequence of emissions for each label.
                Shape (batch_size, seq_len, nb_labels) if batch_first is True,
                (seq_len, batch_size, nb_labels) otherwise.
            mask (torch.FloatTensor, optional): Tensor representing valid positions.
                If None, all positions are considered valid.
                Shape (batch_size, seq_len) if batch_first is True,
                (seq_len, batch_size) otherwise.
        Returns:
            torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists: the best viterbi sequence of labels for each batch.
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float).to(self.device)

        scores, sequences = self._viterbi_decode(emissions, mask)
        return scores, sequences

    def _viterbi_decode(self, emissions, mask):
        """Compute the viterbi algorithm to find the most probable sequence of labels
        given a sequence of emissions.
        Args:
            emissions (torch.Tensor): (batch_size, seq_len, nb_tags)
            mask (Torch.FloatTensor): (batch_size, seq_len)
        Returns:
            torch.Tensor: the viterbi score for the for each batch.
                Shape of (batch_size,)
            list of lists of ints: the best viterbi sequence of labels for each batch
        """
        batch_size, seq_length, nb_tags = emissions.shape
        end_scores = torch.full((batch_size, self.state_size), MINUS_INF).to(self.device)
        backpointers = torch.zeros((seq_length, self.state_size, batch_size), dtype=int).to(
            self.device)
        print(mask.shape)

        for b in range(batch_size):
            alphas = self.transitions[self.BOS_TAG_ID, :] + self.emm_to_state(self.BOS_TAG_ID,
                                                                              emissions[b, 0])

            for i in range(1, seq_length):
                max_scores = torch.full((self.state_size,), MINUS_INF).to(self.device)
                max_score_tags = torch.full((self.state_size,), self.BOS_TAG_ID).to(self.device)

                print(max_scores)
                print(max_score_tags)

                for s in range(self.state_size):
                    e_scores = self.emm_to_state(s, emissions[b, i])
                    t_scores = self.transitions[b, s]
                    a_scores = alphas[s]
                    scores = e_scores + t_scores + a_scores

                    for t in range(self.state_size):
                        if s == 0 or max_score_tags[t] < scores[t]:
                            max_scores[t] = scores[t]
                            max_score_tags[t] = self.states_to_tag(s, t)

                is_valid = mask[b, i]
                alphas = is_valid * max_scores + (1 - is_valid) * alphas

                backpointers[i, :, b] = max_score_tags

            last_transition = self.transitions[:, self.EOS_TAG_ID]
            end_scores[b] = alphas + last_transition.unsqueeze(0)

        # get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)

        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)
        for i in range(batch_size):
            # recover the original sentence length for the i-th sample in the batch
            sample_length = emission_lengths[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences

    def _find_best_path(self, sample_id, best_tag, backpointers):
        """Auxiliary function to find the best path sequence for a specific sample.
            Args:
                sample_id (int): sample index in the range [0, batch_size)
                best_tag (int): tag which maximizes the final score
                backpointers (list of lists of tensors): list of pointers with
                shape (seq_len_i-1, nb_labels, batch_size) where seq_len_i
                represents the length of the ith sample in the batch
            Returns:
                list of ints: a list of tag indexes representing the bast path
        """

        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):
            # recover the best_tag at this timestep
            best_tag = backpointers_t[best_tag][sample_id].item()

            # append to the beginning of the list so we don't need to reverse it later
            best_path.insert(0, best_tag)

        return best_path