"""Tests for the gapped-kmer hashing under non-DNA alphabets.

Companion to test_alphabet.py — focuses specifically on the
gapped_kmer._seqlet_to_gkmers hash plumbing that fans out from
affinitymat.cosine_similarity_from_seqlets.

Key invariants:
  - For DNA (alphabet_size=4) the hash base stays 5 and the CSR shape stays
    5**max_len — bit-identical to pre-fix behavior (covered transitively by
    tests/test_motif.py exact-PWM equality).
  - For protein (alphabet_size=20) the hash base is 21 so distinct
    (position, base) tuples never collide in the hash dictionary.
  - max_len is clamped automatically so hash_base ** max_len fits in int64.
"""

import math

import numpy as np
import pytest

from modiscolite.core import Seqlet
from modiscolite.gapped_kmer import _seqlet_to_gkmers, _safe_max_len


PROTEIN = "ACDEFGHIKLMNPQRSTVWY"


def _make_protein_seqlets(n, length, alphabet_size=20, seed=0):
    """Build n random protein seqlets, each length `length`."""
    rng = np.random.default_rng(seed)
    seqlets = []
    for i in range(n):
        idx = rng.integers(0, alphabet_size, size=length)
        oh = np.zeros((length, alphabet_size), dtype="float32")
        oh[np.arange(length), idx] = 1.0
        # Hypothetical contribs = random small signal so the gkmer "top n
        # positions" path has something to rank by.
        hyp = rng.standard_normal((length, alphabet_size)).astype("float32") * 0.1
        s = Seqlet(example_idx=i, start=0, end=length, is_revcomp=False)
        s.sequence = oh
        s.contrib_scores = oh * hyp
        s.hypothetical_contribs = hyp
        s.alphabet = PROTEIN
        seqlets.append(s)
    return seqlets


def test_safe_max_len_dna_no_clamp():
    # 5**15 ~ 3e10 fits in int64 → max_len stays 15.
    assert _safe_max_len(15, hash_base=5) == 15


def test_safe_max_len_protein_clamps():
    # 21**15 ~ 6.8e19 overflows int64 → max_len must clamp.
    out = _safe_max_len(15, hash_base=21)
    assert out < 15
    # The clamped value should still satisfy 21**out < 2**62.
    assert 21 ** out < 2 ** 62
    # And the next step up should overflow.
    assert 21 ** (out + 1) >= 2 ** 62


def test_safe_max_len_passes_small_max_len_through():
    # If the user already picks a small enough max_len it is preserved.
    assert _safe_max_len(4, hash_base=21) == 4


def test_protein_gkmer_uses_alphabet_size_in_shape():
    seqlets = _make_protein_seqlets(n=10, length=10)
    # Small max_len so we can predict the shape.
    csr = _seqlet_to_gkmers(
        seqlets, topn=8, min_k=3, max_k=4, max_gap=4, max_len=5,
        max_entries=100, take_fwd=True, sign=1, alphabet_size=20,
    )
    expected_cols = 21 ** 5
    assert csr.shape == (10, expected_cols)


def test_dna_gkmer_keeps_legacy_shape():
    """DNA call site must produce CSR with the historical 5**max_len shape."""
    rng = np.random.default_rng(0)
    seqlets = []
    for i in range(5):
        idx = rng.integers(0, 4, size=12)
        oh = np.zeros((12, 4), dtype="float32")
        oh[np.arange(12), idx] = 1.0
        hyp = rng.standard_normal((12, 4)).astype("float32") * 0.1
        s = Seqlet(example_idx=i, start=0, end=12, is_revcomp=False)
        s.sequence = oh
        s.contrib_scores = oh * hyp
        s.hypothetical_contribs = hyp
        s.alphabet = "ACGT"
        seqlets.append(s)
    csr = _seqlet_to_gkmers(
        seqlets, topn=8, min_k=3, max_k=4, max_gap=4, max_len=5,
        max_entries=100, take_fwd=True, sign=1, alphabet_size=4,
    )
    assert csr.shape == (5, 5 ** 5)


def test_protein_gkmer_no_overflow_at_default_max_len():
    """With alphabet_size=20 and max_len=15 the function must not raise.

    The bug: pre-fix, _extract_gkmers wrote keys up to ~21**14 ~ 1.4e18 into
    a CSR with shape 5**15 ~ 3e10 columns. scipy doesn't bounds-check on
    construction so it silently produced a broken matrix. After the fix the
    matrix shape is hash_base**clamped_max_len which fits in int64 and is
    consistent with the hash range.
    """
    seqlets = _make_protein_seqlets(n=20, length=20)
    csr = _seqlet_to_gkmers(
        seqlets, topn=10, min_k=4, max_k=6, max_gap=8, max_len=15,
        max_entries=200, take_fwd=True, sign=1, alphabet_size=20,
    )
    # max_len was clamped; max column index must be < csr.shape[1].
    if csr.nnz > 0:
        assert csr.indices.max() < csr.shape[1]
