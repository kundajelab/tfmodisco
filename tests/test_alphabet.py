"""Alphabet-plumbing tests for the protein-importance-scores generalization.

Covers the cheap, fast pieces:
  - TrackSet validates one_hot last-dim vs alphabet length
  - TrackSet stamps the alphabet onto every Seqlet it creates
  - SeqletSet inherits alphabet and sizes its buffers accordingly
  - save_hdf5 persists the alphabet attr; round-trips through h5py
  - Default-alphabet path (ACGT) is unchanged

Backward compatibility on the full DNA pipeline is covered by
`test_motif.py`, which asserts exact-equality on three reference PWMs.
"""

import os

import h5py
import numpy as np
import pytest

from modiscolite.core import Seqlet, SeqletSet, TrackSet
from modiscolite.io import save_hdf5


DNA = "ACGT"
PROTEIN = "ACDEFGHIKLMNPQRSTVWY"


def _make_one_hot(n, length, alphabet_size, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, alphabet_size, size=(n, length))
    oh = np.zeros((n, length, alphabet_size), dtype="float32")
    for i in range(n):
        oh[i, np.arange(length), idx[i]] = 1.0
    return oh


def test_trackset_default_alphabet_is_dna():
    oh = _make_one_hot(2, 10, 4)
    ts = TrackSet(one_hot=oh, contrib_scores=oh.copy(), hypothetical_contribs=oh.copy())
    assert ts.alphabet == DNA
    assert ts.length == 10


def test_trackset_rejects_alphabet_shape_mismatch():
    oh = _make_one_hot(2, 10, 4)
    with pytest.raises(ValueError, match="alphabet length"):
        TrackSet(
            one_hot=oh,
            contrib_scores=oh.copy(),
            hypothetical_contribs=oh.copy(),
            alphabet=PROTEIN,
        )


def test_trackset_protein_alphabet_accepted():
    oh = _make_one_hot(2, 10, 20)
    ts = TrackSet(
        one_hot=oh,
        contrib_scores=oh.copy(),
        hypothetical_contribs=oh.copy(),
        alphabet=PROTEIN,
    )
    assert ts.alphabet == PROTEIN


def test_create_seqlets_stamps_alphabet():
    oh = _make_one_hot(1, 30, 20)
    ts = TrackSet(
        one_hot=oh,
        contrib_scores=oh.copy(),
        hypothetical_contribs=oh.copy(),
        alphabet=PROTEIN,
    )
    raw_seqlets = [Seqlet(example_idx=0, start=5, end=15, is_revcomp=False)]
    seqlets = ts.create_seqlets(raw_seqlets)
    assert all(s.alphabet == PROTEIN for s in seqlets)
    # sequence slice has the right trailing dim
    assert seqlets[0].sequence.shape == (10, 20)


def test_seqletset_inherits_alphabet_and_buffer_shape():
    oh = _make_one_hot(1, 30, 20)
    ts = TrackSet(
        one_hot=oh,
        contrib_scores=oh.copy(),
        hypothetical_contribs=oh.copy(),
        alphabet=PROTEIN,
    )
    seqlets = ts.create_seqlets(
        [Seqlet(example_idx=0, start=5, end=15, is_revcomp=False)]
    )
    ss = SeqletSet(seqlets)
    assert ss.alphabet == PROTEIN
    assert ss.alphabet_size == 20
    assert ss.sequence.shape == (10, 20)
    assert ss.contrib_scores.shape == (10, 20)


def test_seqletset_falls_back_to_dna_for_unstamped_seqlets():
    """Pickles from before this PR have seqlet.alphabet == None."""
    seqlet = Seqlet(example_idx=0, start=0, end=5, is_revcomp=False)
    seqlet.sequence = np.eye(5, 4, dtype="float32")
    seqlet.contrib_scores = np.zeros((5, 4), dtype="float32")
    seqlet.hypothetical_contribs = np.zeros((5, 4), dtype="float32")
    assert seqlet.alphabet is None
    ss = SeqletSet([seqlet])
    assert ss.alphabet == DNA
    assert ss.alphabet_size == 4


def _toy_seqletset(alphabet, length=8):
    """Build a minimal SeqletSet usable by save_hdf5."""
    oh = _make_one_hot(1, length + 4, len(alphabet))
    ts = TrackSet(
        one_hot=oh,
        contrib_scores=oh.copy(),
        hypothetical_contribs=oh.copy(),
        alphabet=alphabet,
    )
    seqlets = ts.create_seqlets(
        [Seqlet(example_idx=0, start=2, end=2 + length, is_revcomp=False)]
    )
    return SeqletSet(seqlets)


def test_save_hdf5_persists_protein_alphabet(tmp_path):
    pattern = _toy_seqletset(PROTEIN)
    h5_path = tmp_path / "modisco.h5"
    save_hdf5(str(h5_path), pos_patterns=[pattern], neg_patterns=None, window_size=8)

    with h5py.File(h5_path, "r") as f:
        stored = f.attrs["alphabet"]
        if isinstance(stored, bytes):
            stored = stored.decode("ascii")
        assert stored == PROTEIN
        assert "pos_patterns" in f


def test_save_hdf5_persists_dna_alphabet(tmp_path):
    pattern = _toy_seqletset(DNA)
    h5_path = tmp_path / "modisco.h5"
    save_hdf5(str(h5_path), pos_patterns=[pattern], neg_patterns=None, window_size=8)

    with h5py.File(h5_path, "r") as f:
        stored = f.attrs["alphabet"]
        if isinstance(stored, bytes):
            stored = stored.decode("ascii")
        assert stored == DNA


def test_save_hdf5_falls_back_to_dna_when_alphabet_missing(tmp_path):
    """Old SeqletSets without an alphabet attribute should land as ACGT."""
    pattern = _toy_seqletset(DNA)
    # Simulate a pre-PR SeqletSet that lacks the alphabet attribute.
    delattr(pattern, "alphabet")

    h5_path = tmp_path / "modisco.h5"
    save_hdf5(str(h5_path), pos_patterns=[pattern], neg_patterns=None, window_size=8)

    with h5py.File(h5_path, "r") as f:
        stored = f.attrs["alphabet"]
        if isinstance(stored, bytes):
            stored = stored.decode("ascii")
        assert stored == DNA
