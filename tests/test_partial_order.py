"""CPU tests of the order-theoretic claims and algorithms in Liu et al. (EMNLP 2023).

Covers: token-split structures (Def. 3.13, Thm. 3.14, Eq. 1); 2-dimensionality of dependency trees (Assumption 3.12)
and its failure for general graphs; the pair-wise function F (Eq. 2); the linear-time aggregation (Alg. 1, Eqs. 5-6)
under min / max / log-sum-exp; greedy head decoding; and the repo's ModelForPartialOrder scoring + loss driven by an
exact realizer.

Run from the repo root: python -m pytest tests/ -q
"""
import itertools
import math
import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----------------------------------------------------------------------------------------------------------------------
# Structures
# ----------------------------------------------------------------------------------------------------------------------

def random_tree(n, rng):
    """Random (generally non-projective) dependency tree: heads[i] in {0..n}, word i in 1..n, 0 = ROOT."""
    order = list(range(1, n + 1))
    rng.shuffle(order)
    heads = [None] * (n + 1)
    heads[order[0]] = 0
    for i, w in enumerate(order[1:], start=1):
        heads[w] = order[rng.randrange(i)] if rng.random() < 0.95 else 0
    return heads[1:]  # heads of words 1..n


def token_split(edges):
    """Def. 3.13: (x, y) in E  ->  (x_r, y_b) in E_hat."""
    return {(("r", x), ("b", y)) for x, y in edges}


def recover(edges_hat):
    """Eq. 1: E = {(x, y) | x_r < y_b}."""
    return {(x[1], y[1]) for x, y in edges_hat if x[0] == "r" and y[0] == "b"}


def tree_realizer(heads):
    """Explicit K=2 realizer of a tree's token-split structure (constructive proof of Assumption 3.12 for trees).

    The token-split poset of a tree is a disjoint union of stars {children(y)_r < y_b}. Listing the stars in
    order (children, then center) gives L1; listing them in reverse order with reversed children gives L2. Same-star
    child/center pairs are below in both; everything else is ordered oppositely, hence incomparable.
    Returns dicts f1, f2: vertex -> position. ROOT is not a vertex (the repo scores it with a constant 0).
    """
    n = len(heads)
    groups = []
    root_kids = [x for x in range(1, n + 1) if heads[x - 1] == 0]
    if root_kids:
        groups.append([("r", x) for x in root_kids])
    for y in range(1, n + 1):
        groups.append([("r", x) for x in range(1, n + 1) if heads[x - 1] == y] + [("b", y)])
    l1 = [v for g in groups for v in g]
    l2 = [v for g in reversed(groups) for v in (g[:-1][::-1] + g[-1:] if g[-1][0] == "b" else g[::-1])]
    return {v: i for i, v in enumerate(l1)}, {v: i for i, v in enumerate(l2)}


def induced_order(realizers, vertices):
    """Def. 4.1: x < y iff f_k(x) < f_k(y) for all k (intersection of the total orders)."""
    return {(x, y) for x in vertices for y in vertices if x != y and all(f[x] < f[y] for f in realizers)}


# ----------------------------------------------------------------------------------------------------------------------
# Token-split structures (Def. 3.13, Thm. 3.14, Eq. 1)
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(20))
def test_token_split_is_partial_order_and_invertible(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 12)
    # arbitrary digraph incl. self-loops and cycles (e.g. coreference-like structures)
    edges = {(x, y) for x in range(n) for y in range(n) if rng.random() < 0.3}
    eh = token_split(edges)
    # irreflexive + antisymmetric
    assert all(a != b for a, b in eh)
    assert not any((b, a) in eh for a, b in eh)
    # transitive: no chain a<b<c exists since every edge goes r -> b
    assert not any(b == c for a, b in eh for c, d in eh)
    # Eq. 1 recovers the original structure, self-loops and cycles included
    assert recover(eh) == edges


# ----------------------------------------------------------------------------------------------------------------------
# Order dimension (Assumption 3.12)
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(50))
def test_trees_are_2_dimensional(seed):
    rng = random.Random(seed)
    heads = random_tree(rng.randint(1, 25), rng)
    n = len(heads)
    f1, f2 = tree_realizer(heads)
    verts = list(f1)
    rel = induced_order([f1, f2], verts)
    gold = token_split({(x, heads[x - 1]) for x in range(1, n + 1) if heads[x - 1] != 0})
    # the intersection of 2 total orders is EXACTLY the token-split structure (no extra comparabilities)
    assert rel == gold
    assert recover(rel) == {(x, heads[x - 1]) for x in range(1, n + 1) if heads[x - 1] != 0}


def test_general_graphs_need_more_than_2_orders():
    """The standard example S_3 (Dushnik-Miller) has dimension 3. It is the token-split structure of the graph
    x_i -> y_j for i != j, so K = 2 is a property of sparse/tree structures, not of all graphs (Sec. 3.3)."""
    a = [("r", i) for i in range(3)]
    b = [("b", j) for j in range(3)]
    target = token_split({(i, j) for i in range(3) for j in range(3) if i != j})
    verts = a + b
    exts = [p for p in itertools.permutations(verts)
            if all(p.index(u) < p.index(v) for u, v in target)]  # linear extensions
    pos = [{v: i for i, v in enumerate(p)} for p in exts]
    assert not any(induced_order([p, q], verts) == target for p, q in itertools.combinations_with_replacement(pos, 2))
    assert any(induced_order([p, q, r], verts) == target
               for p, q, r in itertools.combinations(pos, 3))


# ----------------------------------------------------------------------------------------------------------------------
# Eq. 2 and Algorithm 1
# ----------------------------------------------------------------------------------------------------------------------

def pairwise_F(f, g=None):
    """Eq. 2: F(x, y) = max_k f_k(x) - g_k(y); g = f for a single vertex set. f, g: (N, K) arrays."""
    g = f if g is None else g
    return (f[:, None, :] - g[None, :, :]).max(-1)


SEMIRINGS = {  # (oplus, zero element); real + distributes over each
    "min": (np.minimum, np.inf),
    "max": (np.maximum, -np.inf),
    "logsumexp": (np.logaddexp, -np.inf),
}


def algorithm1(a, b, key, plus, zero):
    """Algorithm 1 (generalised): oplus_x oplus_{y before x in sort(key)} (a(x) + b(y)), O(N) after sorting."""
    G, s = zero, zero
    for n in np.argsort(key, kind="stable"):
        q = a[n] + s        # line 5: G1(U_n)
        G = plus(G, q)      # line 6
        s = plus(s, b[n])   # line 7
    return G


@pytest.mark.parametrize("seed", range(10))
def test_fredman_trick_partitions_pairs(seed):
    """For y before x in the order of f1 - f2, F(x, y) = f1(x) - f1(y); for y after x, F(x, y) = f2(x) - f2(y)."""
    f = np.random.default_rng(seed).normal(size=(40, 2))
    F = pairwise_F(f)
    rank = np.argsort(np.argsort(f[:, 0] - f[:, 1]))
    for x in range(40):
        for y in range(40):
            if rank[y] < rank[x]:
                assert F[x, y] == pytest.approx(f[x, 0] - f[y, 0])
            elif rank[y] > rank[x]:
                assert F[x, y] == pytest.approx(f[x, 1] - f[y, 1])


@pytest.mark.parametrize("semiring", SEMIRINGS)
@pytest.mark.parametrize("sign", [1, -1])  # +F (decoding) and -F (first term of the training objective, Eq. 3)
@pytest.mark.parametrize("seed", range(5))
def test_algorithm1_matches_quadratic_aggregation(semiring, sign, seed):
    """Eq. (5): oplus over all ordered pairs x != y of sign*F(x, y) equals G1 oplus G2, each computed in O(N)."""
    plus, zero = SEMIRINGS[semiring]
    f = np.random.default_rng(seed).normal(size=(64, 2))
    F = sign * pairwise_F(f)
    naive = zero
    for x in range(64):
        for y in range(64):
            if x != y:
                naive = plus(naive, F[x, y])
    d = f[:, 0] - f[:, 1]
    G1 = algorithm1(sign * f[:, 0], -sign * f[:, 0], d, plus, zero)
    G2 = algorithm1(sign * f[:, 1], -sign * f[:, 1], -d, plus, zero)  # reverse order, by symmetry
    assert plus(G1, G2) == pytest.approx(naive, rel=1e-9, abs=1e-9)


def linear_greedy_decode(f, g):
    """argmin_y F(x, y) for every x (Sec. 5.1 decoding) via the sorted-prefix trick, O((N + M) log(N + M)).

    y is in S1(x) iff g1(y) - g2(y) <= f1(x) - f2(x); there F = f1(x) - g1(y), minimised by the prefix max of g1.
    The complementary suffix uses g2. Returns (min values, argmins).
    """
    dy = g[:, 0] - g[:, 1]
    order = np.argsort(dy, kind="stable")
    pre_val, pre_arg = np.maximum.accumulate(g[order, 0]), order[_running_argmax(g[order, 0])]
    rev = order[::-1]
    suf_val, suf_arg = np.maximum.accumulate(g[rev, 1])[::-1], rev[_running_argmax(g[rev, 1])][::-1]
    cut = np.searchsorted(dy[order], f[:, 0] - f[:, 1], side="right")  # |S1(x)|
    out_v, out_a = np.full(len(f), np.inf), np.zeros(len(f), dtype=int)
    for x, c in enumerate(cut):
        if c > 0 and f[x, 0] - pre_val[c - 1] < out_v[x]:
            out_v[x], out_a[x] = f[x, 0] - pre_val[c - 1], pre_arg[c - 1]
        if c < len(g) and f[x, 1] - suf_val[c] < out_v[x]:
            out_v[x], out_a[x] = f[x, 1] - suf_val[c], suf_arg[c]
    return out_v, out_a


def _running_argmax(v):
    idx = np.zeros(len(v), dtype=int)
    for i in range(1, len(v)):
        idx[i] = i if v[i] > v[idx[i - 1]] else idx[i - 1]
    return idx


@pytest.mark.parametrize("seed", range(10))
def test_linear_greedy_decoding_matches_quadratic(seed):
    rng = np.random.default_rng(seed)
    f, g = rng.normal(size=(50, 2)), rng.normal(size=(50, 2))  # x_r and y_b realizers (token-split)
    F = pairwise_F(f, g)
    vals, args = linear_greedy_decode(f, g)
    np.testing.assert_allclose(vals, F.min(1))
    np.testing.assert_array_equal(args, F.argmin(1))


# ----------------------------------------------------------------------------------------------------------------------
# The repo's model: scoring (learn.py) + loss, driven by an exact K=2 realizer
# ----------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from transformers import BertConfig, BertModel
    from learning.learn import ModelForPartialOrder

    path = str(tmp_path_factory.mktemp("tiny_bert"))
    cfg = BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, max_position_embeddings=64)
    BertModel(cfg).save_pretrained(path)
    cfg.task_specific_params = {"model_path": path, "pos_emb_dim": 4, "dropout": 0.0, "lstm_layers": 0,
                                "order_dim": 2, "num_rel_tags": 5}
    torch.manual_seed(0)
    return ModelForPartialOrder(cfg)


class _Fn(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, t):
        return self.fn(t)


def _batch(heads_list):
    """Fake one-subword-per-word batch: [CLS] w1 .. wn [SEP], padded."""
    L = max(len(h) for h in heads_list) + 2
    B = len(heads_list)
    input_ids = torch.zeros(B, L, dtype=torch.long)
    end_of_word = torch.zeros(B, L, dtype=torch.long)
    pos_ids = torch.zeros(B, L, dtype=torch.long)
    pair_ids = torch.zeros(B, L, dtype=torch.long)
    heads = torch.full((B, L - 2), -1, dtype=torch.long)
    for i, h in enumerate(heads_list):
        n = len(h)
        input_ids[i, : n + 2] = torch.randint(1, 50, (n + 2,))
        end_of_word[i, 1 : n + 1] = 1
        end_of_word[i, n] = 2
        pos_ids[i, 1 : n + 1] = 1
        pair_ids[i, 1 : n + 1] = torch.arange(1, n + 1)
        heads[i, :n] = torch.tensor(h)
    return dict(input_ids=input_ids, attention_mask=(input_ids != 0).float(), end_of_word=end_of_word,
                pos_ids=pos_ids, pair_ids=pair_ids, head_labels=heads, rel_labels=torch.zeros_like(heads))


def test_repo_model_recovers_tree_from_exact_realizer(tiny_model):
    """Feed realizer values from tree_realizer into the repo's forward (s_arc = -LSE_k(f_k(x_r) - f_k(y_b)), ROOT
    column = 0). Greedy argmax must return the gold heads, and the arc loss must vanish as the scale grows."""
    rng = random.Random(0)
    heads_list = [random_tree(n, rng) for n in (7, 12, 3, 12)]
    N = max(map(len, heads_list))
    real = torch.zeros(len(heads_list), N, 4)  # [f1(x_r), f2(x_r) | f1(y_b), f2(y_b)]
    for i, h in enumerate(heads_list):
        f1, f2 = tree_realizer(h)
        for w in range(1, len(h) + 1):
            real[i, w - 1] = torch.tensor([f1["r", w], f2["r", w], f1["b", w], f2["b", w]], dtype=torch.float)

    batch = _batch(heads_list)
    from learning.learn import ModelForPartialOrder

    model = ModelForPartialOrder(tiny_model.bert.config).train()  # fresh copy; heads are replaced below
    losses = []
    for scale in (1.0, 10.0, 100.0):
        model.realizer = _Fn(lambda t, s=scale: s * real[:, : t.shape[1]])
        model.rel_clf = _Fn(lambda t: torch.zeros(*t.shape[:2], 5).index_fill(-1, torch.tensor([0]), 1e4))
        loss, (s_arc, _) = model(**batch)
        for i, h in enumerate(heads_list):
            assert s_arc[i, : len(h), : len(h) + 1].argmax(-1).tolist() == h
        losses.append(loss.item())
    assert losses[0] > losses[1] > losses[2] and losses[2] < 1e-6


def test_repo_model_trains_on_cpu(tiny_model):
    """A real realizer (linear layer) on a tiny encoder: finite loss, gradients flow, a few steps reduce the loss."""
    from learning.learn import ModelForPartialOrder

    model = ModelForPartialOrder(tiny_model.bert.config).train()  # fresh realizer / rel classifier
    rng = random.Random(1)
    batch = _batch([random_tree(n, rng) for n in (5, 9, 6)])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        loss = model(**batch)[0]
        assert torch.isfinite(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    assert loss.item() < 0.5 * first
    assert math.isfinite(loss.item())
