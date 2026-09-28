# test_gapped_kmer.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import scipy.sparse
import pytest

from modiscolite.gapped_kmer import _extract_gkmers
from modiscolite.gapped_kmer import _seqlet_to_gkmers

from modiscolite.core import Seqlet

from .synthetic import random_track_set
from .synthetic import make_seqlets

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


# The matrices returned by _seqlet_to_gkmers have 5**max_len columns, so
# these tests never densify, transpose or convert them to CSC.


def _random_X(nx, n, span=30, random_state=0):
	"""Return (nx, n, 3) rows of (position, base, attribution)."""

	rng = numpy.random.RandomState(random_state)
	X = numpy.zeros((nx, n, 3))
	for i in range(nx):
		X[i, :, 0] = numpy.sort(rng.choice(span, n, replace=False))
		X[i, :, 1] = rng.randint(4, size=n)
		X[i, :, 2] = rng.randn(n)

	return X


def _gkmer_reference(x, min_k, max_k, max_gap, max_len):
	"""Enumerate every gapped k-mer in x and sum attribution / k per key."""

	scores = {}
	n = len(x)

	def extend(idxs):
		k = len(idxs)
		if k >= max(min_k, 2):
			start = x[idxs[0], 0]
			key = sum((int(x[i, 1]) + 1) * 5 ** int(x[i, 0] - start)
				for i in idxs)
			scores[key] = scores.get(key, 0) + sum(x[i, 2] for i in idxs) / k

		if k == max_k:
			return

		for i in range(idxs[-1] + 1, n):
			if x[i, 0] - x[idxs[0], 0] >= max_len:
				break
			if x[i, 0] - x[idxs[-1], 0] > max_gap:
				continue
			extend(idxs + [i])

	for j in range(n):
		extend([j])

	return scores


def _as_dict(keys, scores):
	return {int(k): s for k, s in zip(keys, scores) if s != 0}


def _row_dict(csr, i):
	idxs = slice(csr.indptr[i], csr.indptr[i+1])
	return dict(zip(csr.indices[idxs].tolist(), csr.data[idxs].tolist()))


def _seqlet_X(seqlet, topn, take_fwd, sign):
	"""The (position, base, attribution) rows _seqlet_to_gkmers builds."""

	onehot = seqlet.sequence
	contrib = seqlet.hypothetical_contribs * onehot * sign
	if not take_fwd:
		onehot, contrib = onehot[::-1, ::-1], contrib[::-1, ::-1]

	imp = contrib.sum(axis=-1)
	positions = numpy.sort(numpy.argsort(-imp)[:topn])
	return numpy.array([[p, numpy.argmax(onehot[p]), imp[p]]
		for p in positions])


@pytest.fixture
def random_seqlets():
	track_set = random_track_set(n=10, length=60)
	return make_seqlets(track_set, [(i, 3*i, 3*i+25, bool(i % 3 == 0))
		for i in range(10)])


##


@pytest.mark.parametrize("min_k,max_k", [(2, 2), (2, 3), (3, 4), (4, 6),
	(3, 3)])
@pytest.mark.parametrize("max_gap", [1, 3, 15])
@pytest.mark.parametrize("max_len", [5, 10, 15])
def test_extract_gkmers(min_k, max_k, max_gap, max_len):
	X = _random_X(3, 12, span=20, random_state=max_gap*max_len)
	keys, scores = _extract_gkmers(X, min_k, max_k, max_gap, max_len, 10000)

	assert keys.shape == (3, 10000)
	assert scores.shape == (3, 10000)
	assert keys.dtype == numpy.int64
	assert scores.dtype == numpy.float64

	for i in range(3):
		expected = _gkmer_reference(X[i], min_k, max_k, max_gap, max_len)
		observed = _as_dict(keys[i], scores[i])

		assert observed.keys() == expected.keys()
		for key in expected:
			assert abs(observed[key] - expected[key]) < 1e-10


@pytest.mark.parametrize("nx,n", [(1, 3), (1, 20), (5, 8), (20, 10),
	(64, 6)])
def test_extract_gkmers_shapes(nx, n):
	X = _random_X(nx, n, span=25, random_state=n)
	keys, scores = _extract_gkmers(X, 2, 4, 5, 10, 3000)

	assert keys.shape == (nx, 3000)
	for i in range(nx):
		expected = _gkmer_reference(X[i], 2, 4, 5, 10)
		assert _as_dict(keys[i], scores[i]).keys() == expected.keys()


@pytest.mark.parametrize("max_entries", [1, 5, 20, 100])
def test_extract_gkmers_max_entries(max_entries):
	X = _random_X(4, 15, span=20, random_state=0)
	keys, scores = _extract_gkmers(X, 2, 4, 15, 15, max_entries)

	assert keys.shape == (4, max_entries)
	for i in range(4):
		expected = _gkmer_reference(X[i], 2, 4, 15, 15)
		top = sorted(expected.items(), key=lambda kv: -abs(kv[1]))
		top = dict(top[:max_entries])

		observed = _as_dict(keys[i], scores[i])
		assert observed.keys() == top.keys()
		assert numpy.all(numpy.diff(numpy.abs(scores[i])) <= 0)


def test_extract_gkmers_sorted():
	X = _random_X(6, 12, span=20, random_state=1)
	keys, scores = _extract_gkmers(X, 2, 5, 15, 15, 2000)

	for i in range(6):
		nonzero = scores[i][scores[i] != 0]
		assert numpy.all(numpy.diff(numpy.abs(nonzero)) <= 0)
		assert numpy.all(scores[i][len(nonzero):] == 0)
		assert numpy.all(keys[i][len(nonzero):] == 0)


def test_extract_gkmers_values():
	# Bases A, C, G at positions 0, 1, 3 with attributions 1, 2, 3.
	X = numpy.array([[[0, 0, 1.0], [1, 1, 2.0], [3, 2, 3.0]]])
	keys, scores = _extract_gkmers(X, 2, 3, 3, 10, 10)
	observed = _as_dict(keys[0], scores[0])

	assert observed == {
		1 + 2*5: 1.5,             # AC
		1 + 3*125: 2.0,           # A_G
		2 + 3*25: 2.5,            # C_G
		1 + 2*5 + 3*125: 2.0      # AC_G
	}


def test_extract_gkmers_max_gap():
	X = numpy.array([[[0, 0, 1.0], [1, 1, 2.0], [3, 2, 3.0]]])
	keys, scores = _extract_gkmers(X, 2, 3, 1, 10, 10)
	assert _as_dict(keys[0], scores[0]) == {1 + 2*5: 1.5}


def test_extract_gkmers_max_len():
	X = numpy.array([[[0, 0, 1.0], [1, 1, 2.0], [3, 2, 3.0]]])
	keys, scores = _extract_gkmers(X, 2, 3, 3, 3, 10)
	assert _as_dict(keys[0], scores[0]) == {1 + 2*5: 1.5, 2 + 3*25: 2.5}


def test_extract_gkmers_repeated():
	# The same gapped k-mer at two offsets sums both contributions.
	X = numpy.array([[[0, 0, 1.0], [1, 1, 1.0], [5, 0, 2.0], [6, 1, 4.0]]])
	keys, scores = _extract_gkmers(X, 2, 2, 1, 10, 10)
	assert _as_dict(keys[0], scores[0]) == {1 + 2*5: 1.0 + 3.0}


def test_extract_gkmers_min_k_above_max_k():
	X = _random_X(2, 8)
	keys, scores = _extract_gkmers(X, 5, 4, 5, 10, 50)
	assert numpy.all(scores == 0)
	assert numpy.all(keys == 0)


def test_extract_gkmers_rows_independent():
	X = _random_X(5, 10, random_state=3)
	keys, scores = _extract_gkmers(X, 2, 4, 5, 12, 400)

	for i in range(5):
		keys_i, scores_i = _extract_gkmers(X[i:i+1].copy(), 2, 4, 5, 12, 400)
		assert_array_equal(keys[i], keys_i[0])
		assert_array_equal(scores[i], scores_i[0])


def test_extract_gkmers_sign():
	X = _random_X(3, 10, random_state=4)
	Y = X.copy()
	Y[:, :, 2] *= -1

	keys0, scores0 = _extract_gkmers(X, 2, 4, 5, 12, 400)
	keys1, scores1 = _extract_gkmers(Y, 2, 4, 5, 12, 400)

	for i in range(3):
		assert _as_dict(keys0[i], scores0[i]) == {k: -v for k, v in
			_as_dict(keys1[i], scores1[i]).items()}


def test_extract_gkmers_translation():
	# Keys are relative to the first position, so shifting every position
	# leaves the output unchanged.
	X = _random_X(2, 10, random_state=5)
	Y = X.copy()
	Y[:, :, 0] += 17

	keys0, scores0 = _extract_gkmers(X, 2, 4, 5, 12, 400)
	keys1, scores1 = _extract_gkmers(Y, 2, 4, 5, 12, 400)

	assert_array_equal(keys0, keys1)
	assert_array_almost_equal(scores0, scores1)


def test_extract_gkmers_keys_in_range():
	X = _random_X(4, 15, span=40, random_state=6)
	keys, scores = _extract_gkmers(X, 2, 6, 15, 15, 5000)

	assert numpy.all(keys >= 0)
	assert numpy.all(keys < 5**15)


##


@pytest.mark.parametrize("take_fwd", [True, False])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("topn", [5, 10, 25])
def test_seqlet_to_gkmers(random_seqlets, take_fwd, sign, topn):
	X = _seqlet_to_gkmers(random_seqlets, topn, 2, 4, 5, 15, 500, take_fwd,
		sign)

	assert isinstance(X, scipy.sparse.csr_matrix)
	assert X.shape == (10, 5**15)

	for i, seqlet in enumerate(random_seqlets):
		x = _seqlet_X(seqlet, topn, take_fwd, sign)
		expected = _gkmer_reference(x, 2, 4, 5, 15)
		expected = dict(sorted(expected.items(),
			key=lambda kv: -abs(kv[1]))[:500])

		observed = _row_dict(X, i)
		assert observed.keys() == expected.keys()
		for key in expected:
			assert abs(observed[key] - expected[key]) < 1e-8


@pytest.mark.parametrize("max_entries", [1, 10, 50, 500])
def test_seqlet_to_gkmers_max_entries(random_seqlets, max_entries):
	X = _seqlet_to_gkmers(random_seqlets, 20, 4, 6, 15, 15, max_entries,
		True, 1)
	assert numpy.all(numpy.diff(X.indptr) <= max_entries)
	assert numpy.all(X.data != 0)


@pytest.mark.parametrize("max_len", [14, 15, 16])
def test_seqlet_to_gkmers_max_len(random_seqlets, max_len):
	X = _seqlet_to_gkmers(random_seqlets, 20, 2, 4, 5, max_len, 500, True, 1)

	assert X.shape == (10, 5**max_len)
	assert X.indices.dtype == numpy.int64
	assert numpy.all(X.indices < 5**max_len)


def test_seqlet_to_gkmers_revcomp(random_seqlets):
	X0 = _seqlet_to_gkmers(random_seqlets, 15, 2, 4, 5, 15, 500, False, 1)
	X1 = _seqlet_to_gkmers([s.revcomp() for s in random_seqlets], 15, 2, 4, 5,
		15, 500, True, 1)

	for i in range(10):
		assert _row_dict(X0, i) == _row_dict(X1, i)


def test_seqlet_to_gkmers_sign(random_seqlets):
	negated = []
	for seqlet in random_seqlets:
		s = Seqlet(seqlet.example_idx, seqlet.start, seqlet.end,
			seqlet.is_revcomp)
		s.sequence = seqlet.sequence
		s.hypothetical_contribs = -seqlet.hypothetical_contribs
		negated.append(s)

	X0 = _seqlet_to_gkmers(random_seqlets, 15, 2, 4, 5, 15, 500, True, 1)
	X1 = _seqlet_to_gkmers(negated, 15, 2, 4, 5, 15, 500, True, -1)

	for i in range(10):
		assert _row_dict(X0, i) == _row_dict(X1, i)


def test_seqlet_to_gkmers_topn_all(random_seqlets):
	# A topn of at least the seqlet length uses every position.
	X0 = _seqlet_to_gkmers(random_seqlets, 25, 2, 3, 3, 15, 5000, True, 1)
	X1 = _seqlet_to_gkmers(random_seqlets, 100, 2, 3, 3, 15, 5000, True, 1)

	for i in range(10):
		assert _row_dict(X0, i) == _row_dict(X1, i)


def test_seqlet_to_gkmers_uses_contribs_not_hypothetical(random_seqlets):
	# Only the observed base carries attribution, so changing the attribution
	# of the unobserved bases leaves the output unchanged.
	modified = []
	for seqlet in random_seqlets:
		s = Seqlet(seqlet.example_idx, seqlet.start, seqlet.end,
			seqlet.is_revcomp)
		s.sequence = seqlet.sequence
		s.hypothetical_contribs = seqlet.hypothetical_contribs + 5 * (
			1 - seqlet.sequence)
		modified.append(s)

	X0 = _seqlet_to_gkmers(random_seqlets, 15, 2, 4, 5, 15, 500, True, 1)
	X1 = _seqlet_to_gkmers(modified, 15, 2, 4, 5, 15, 500, True, 1)

	for i in range(10):
		assert _row_dict(X0, i) == _row_dict(X1, i)


def test_seqlet_to_gkmers_single(random_seqlets):
	X = _seqlet_to_gkmers(random_seqlets[:1], 10, 2, 4, 5, 15, 500, True, 1)
	assert X.shape == (1, 5**15)
	assert X.nnz > 0


def test_seqlet_to_gkmers_real(seqlets):
	X = _seqlet_to_gkmers(seqlets[:50], 20, 4, 6, 15, 15, 500, True, 1)

	assert X.shape == (50, 5**15)
	assert X.nnz == 50 * 500

	top = [numpy.sort(numpy.abs(list(_row_dict(X, i).values())))[::-1][:4]
		for i in range(2)]
	assert_array_almost_equal(top, [[2.5596, 2.0713, 2.0573, 2.0525],
		[1.6976, 1.567, 1.5533, 1.5262]], 4)
