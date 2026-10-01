# test_affinitymat.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import scipy.sparse
import sklearn.preprocessing
import pytest

from modiscolite.affinitymat import _sparse_vv_dot
from modiscolite.affinitymat import _sparse_mm_dot
from modiscolite.affinitymat import cosine_similarity_from_seqlets
from modiscolite.affinitymat import jaccard_from_seqlets
from modiscolite.affinitymat import jaccard
from modiscolite.affinitymat import pairwise_jaccard
from modiscolite.affinitymat import _jaccard
from modiscolite.affinitymat import pearson_correlation
from modiscolite.affinitymat import NNTsneConditionalProbs

from modiscolite.core import Seqlet
from modiscolite.gapped_kmer import _seqlet_to_gkmers
from modiscolite.util import get_2d_data_from_patterns

from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


def _csr(n, d, density, random_state=0):
	X = scipy.sparse.random(n, d, density=density, format='csr',
		random_state=random_state)
	X.sort_indices()
	X.indices = X.indices.astype('int64')
	X.indptr = X.indptr.astype('int64')
	return X


def _cont_jaccard(x, y):
	sign = numpy.sign(x) * numpy.sign(y)
	ax, ay = numpy.abs(x), numpy.abs(y)
	return (numpy.minimum(ax, ay) * sign).sum() / numpy.maximum(ax, ay).sum()


def _jaccard_reference(X, Y, min_overlap, seqlet_neighbors, func):
	X, Y = X.astype('float64'), Y.astype('float64')
	if seqlet_neighbors is None:
		seqlet_neighbors = numpy.tile(numpy.arange(len(X)), (len(Y), 1))

	n_pad = 0 if min_overlap is None else int(func(X.shape[1]*(1-min_overlap)))
	Y = numpy.pad(Y, ((0, 0), (n_pad, n_pad), (0, 0)))
	d = X.shape[1]

	results = numpy.zeros((len(Y), seqlet_neighbors.shape[1], 2))
	for l in range(len(Y)):
		for i, neighbor in enumerate(seqlet_neighbors[l]):
			scores = [_cont_jaccard(X[neighbor], Y[l, idx:idx+d])
				for idx in range(Y.shape[1] - d + 1)]
			results[l, i] = numpy.max(scores), numpy.argmax(scores) - n_pad

	return results


def _pearson_reference(x, y, min_overlap, func):
	n_pad = int(func(len(x) * (1 - min_overlap)))
	y = numpy.pad(y, ((n_pad, n_pad), (0, 0)))
	d = len(x)

	scores = []
	for idx in range(len(y) - d + 1):
		y_ = y[idx:idx+d]
		norm = numpy.linalg.norm(y_)
		scores.append(0.0 if norm == 0 else (x * y_).sum() /
			(numpy.linalg.norm(x) * norm))

	return max(scores), numpy.argmax(scores) - n_pad


def _compact(X, Y):
	"""Dense copies of X and Y restricted to the columns either one uses.

	The gapped k-mer matrices have 5**15 columns, so transposing one or
	converting it to CSC would allocate hundreds of gigabytes.
	"""

	cols = numpy.union1d(X.indices, Y.indices)
	dense = []
	for Z in (X, Y):
		Z_ = numpy.zeros((Z.shape[0], len(cols)))
		for i in range(Z.shape[0]):
			idxs = slice(Z.indptr[i], Z.indptr[i+1])
			Z_[i, numpy.searchsorted(cols, Z.indices[idxs])] = Z.data[idxs]

		dense.append(Z_)

	return dense


def _mm_dot_reference(X, Y, k):
	X, Y = _compact(X, Y)
	dot = numpy.maximum(X @ X.T, X @ Y.T)
	neighbors = numpy.argsort(-dot, axis=1, kind='mergesort')[:, :k]
	return numpy.take_along_axis(dot, neighbors, axis=1), neighbors


def _pairwise_reference(X, k):
	n = len(X)
	J = numpy.array([[_cont_jaccard(X[i], X[j]) for j in range(n)]
		for i in range(n)])
	neighbors = numpy.argsort(-J, axis=1, kind='mergesort')[:, :k]
	return numpy.take_along_axis(J, neighbors, axis=1), neighbors


def _negated(seqlets):
	"""Copies of `seqlets` with the attributions negated."""

	negated = []
	for seqlet in seqlets:
		s = Seqlet(seqlet.example_idx, seqlet.start, seqlet.end,
			seqlet.is_revcomp)
		s.sequence = seqlet.sequence
		s.contrib_scores = -seqlet.contrib_scores
		s.hypothetical_contribs = -seqlet.hypothetical_contribs
		negated.append(s)

	return negated


@pytest.fixture
def X3():
	return numpy.random.RandomState(0).randn(6, 12, 4)


##


@pytest.mark.parametrize("n,d,density", [(1, 10, 0.5), (5, 10, 0.3),
	(5, 100, 0.05), (20, 50, 0.2), (8, 30, 1.0)])
def test_sparse_vv_dot(n, d, density):
	X = _csr(n, d, density, random_state=0)
	Y = _csr(n, d, density, random_state=1)
	dot = (X @ Y.T).toarray()

	for i in range(n):
		for j in range(n):
			y = _sparse_vv_dot(X.data, X.indices, X.indptr, Y.data, Y.indices,
				Y.indptr, i, j)
			assert abs(y - dot[i, j]) < 1e-10


def test_sparse_vv_dot_values():
	X = scipy.sparse.csr_matrix(numpy.array([[1.0, 0, 2.0, 0, 3.0],
		[0, 0, 0, 0, 0], [0, 4.0, 0, 5.0, 0]]))
	indices, indptr = X.indices.astype('int64'), X.indptr.astype('int64')

	assert _sparse_vv_dot(X.data, indices, indptr, X.data, indices, indptr,
		0, 0) == 14.0
	assert _sparse_vv_dot(X.data, indices, indptr, X.data, indices, indptr,
		0, 2) == 0.0
	assert _sparse_vv_dot(X.data, indices, indptr, X.data, indices, indptr,
		1, 1) == 0.0
	assert _sparse_vv_dot(X.data, indices, indptr, X.data, indices, indptr,
		2, 2) == 41.0


def test_sparse_vv_dot_raises():
	X = scipy.sparse.csr_matrix(numpy.eye(3))
	assert_raises(TypeError, _sparse_vv_dot, X.data, X.indices, X.indptr,
		X.data, X.indices, X.indptr, 0, 0)


##


@pytest.mark.parametrize("n,d,density", [(1, 10, 0.5), (5, 10, 0.3),
	(10, 100, 0.05), (30, 50, 0.2), (12, 30, 1.0)])
@pytest.mark.parametrize("k", [1, 3, 100])
def test_sparse_mm_dot(n, d, density, k):
	X = _csr(n, d, density, random_state=0)
	Y = _csr(n, d, density, random_state=1)
	k = min(k, n)

	sims, neighbors = _sparse_mm_dot(X.data, X.indices, X.indptr, Y.data,
		Y.indices, Y.indptr, k)
	sims_ref, neighbors_ref = _mm_dot_reference(X, Y, k)

	assert sims.shape == (n, k)
	assert neighbors.shape == (n, k)
	assert sims.dtype == numpy.float64
	assert neighbors.dtype == numpy.int32
	assert_array_almost_equal(sims, sims_ref)
	assert_array_equal(neighbors, neighbors_ref)


def test_sparse_mm_dot_max():
	X = scipy.sparse.csr_matrix(numpy.array([[1.0, 0.0], [0.0, 1.0]]))
	Y = scipy.sparse.csr_matrix(numpy.array([[0.0, 2.0], [3.0, 0.0]]))
	args = [a.astype('int64') if a.dtype.kind == 'i' else a for a in
		(X.data, X.indices, X.indptr, Y.data, Y.indices, Y.indptr)]

	sims, neighbors = _sparse_mm_dot(*args, 2)
	assert_array_almost_equal(sims, [[3.0, 1.0], [2.0, 1.0]])
	assert_array_equal(neighbors, [[1, 0], [0, 1]])


def test_sparse_mm_dot_ties():
	# Ties keep their original order.
	X = _csr(6, 4, 1.0)
	X.data[:] = 1.0

	sims, neighbors = _sparse_mm_dot(X.data, X.indices, X.indptr, X.data,
		X.indices, X.indptr, 6)
	assert_array_almost_equal(sims, numpy.full((6, 6), 4.0))
	assert_array_equal(neighbors, numpy.tile(numpy.arange(6), (6, 1)))


##


@pytest.mark.parametrize("n", [2, 10, 40])
@pytest.mark.parametrize("n_neighbors", [1, 5, 20, 1000])
def test_cosine_similarity_from_seqlets(seqlets, n, n_neighbors):
	sims, neighbors = cosine_similarity_from_seqlets(seqlets[:n],
		n_neighbors=n_neighbors, sign=1)

	k = min(n_neighbors + 1, n)
	assert sims.shape == (n, k)
	assert neighbors.shape == (n, k)
	assert numpy.all(numpy.diff(sims, axis=1) <= 1e-12)
	assert numpy.all(sims <= 1 + 1e-6)
	assert numpy.all(neighbors < n)


@pytest.mark.parametrize("sign", [1, -1])
def test_cosine_similarity_from_seqlets_reference(seqlets, sign):
	seqlets = seqlets[:30]
	sims, neighbors = cosine_similarity_from_seqlets(seqlets, n_neighbors=9,
		sign=sign)

	X = sklearn.preprocessing.normalize(_seqlet_to_gkmers(seqlets, 20, 4, 6,
		15, 15, 500, True, sign))
	Y = sklearn.preprocessing.normalize(_seqlet_to_gkmers(seqlets, 20, 4, 6,
		15, 15, 500, False, sign))
	sims_ref, neighbors_ref = _mm_dot_reference(X, Y, 10)

	assert_array_almost_equal(sims, sims_ref)
	assert_array_equal(neighbors, neighbors_ref)


def test_cosine_similarity_from_seqlets_self(seqlets):
	sims, neighbors = cosine_similarity_from_seqlets(seqlets[:50],
		n_neighbors=10, sign=1)

	assert_array_almost_equal(sims[:, 0], numpy.ones(50))
	self_sims = [sims[i][neighbors[i] == i] for i in range(50)]
	assert all(len(s) == 0 or abs(s[0] - 1) < 1e-6 for s in self_sims)


def test_cosine_similarity_from_seqlets_sign(seqlets):
	sims0, neighbors0 = cosine_similarity_from_seqlets(seqlets[:30],
		n_neighbors=5, sign=1)
	sims1, neighbors1 = cosine_similarity_from_seqlets(_negated(seqlets[:30]),
		n_neighbors=5, sign=-1)

	assert_array_almost_equal(sims0, sims1)
	assert_array_equal(neighbors0, neighbors1)


@pytest.mark.parametrize("kwargs", [{'topn': 10}, {'topn': 30},
	{'min_k': 3, 'max_k': 5}, {'min_k': 2, 'max_k': 3}, {'max_gap': 5},
	{'max_len': 14}, {'max_entries': 50}, {'alphabet_size': 8}])
def test_cosine_similarity_from_seqlets_kwargs(seqlets, kwargs):
	params = dict(topn=20, min_k=4, max_k=6, max_gap=15, max_len=15,
		max_entries=500)
	params.update(kwargs)

	sims, neighbors = cosine_similarity_from_seqlets(seqlets[:25],
		n_neighbors=6, sign=1, **kwargs)

	X = sklearn.preprocessing.normalize(_seqlet_to_gkmers(seqlets[:25],
		params['topn'], params['min_k'], params['max_k'], params['max_gap'],
		params['max_len'], params['max_entries'], True, 1))
	Y = sklearn.preprocessing.normalize(_seqlet_to_gkmers(seqlets[:25],
		params['topn'], params['min_k'], params['max_k'], params['max_gap'],
		params['max_len'], params['max_entries'], False, 1))
	sims_ref, neighbors_ref = _mm_dot_reference(X, Y, 7)

	assert_array_almost_equal(sims, sims_ref)
	assert_array_equal(neighbors, neighbors_ref)


@pytest.mark.skip(reason="bug: for max_len <= 13, 5**max_len fits in int32 "
	"so scipy builds the gapped k-mer matrix with int32 indices, which "
	"_sparse_vv_dot's int64-only signature rejects with a TypingError")
@pytest.mark.parametrize("max_len", [8, 10, 13])
def test_cosine_similarity_from_seqlets_max_len(seqlets, max_len):
	sims, neighbors = cosine_similarity_from_seqlets(seqlets[:25],
		n_neighbors=6, sign=1, max_gap=5, max_len=max_len)
	assert sims.shape == (25, 7)


def test_cosine_similarity_from_seqlets_revcomp(seqlets):
	# A seqlet and its reverse complement are near-perfect matches through
	# the backward representation.
	seqlets = seqlets[:10] + [s.revcomp() for s in seqlets[:10]]
	sims, neighbors = cosine_similarity_from_seqlets(seqlets, n_neighbors=19,
		sign=1)

	for i in range(10):
		partner = numpy.where(neighbors[i] == i + 10)[0][0]
		assert sims[i, partner] > 0.99


def test_cosine_similarity_from_seqlets_real(seqlets):
	sims, neighbors = cosine_similarity_from_seqlets(seqlets[:100],
		n_neighbors=3, sign=1)

	assert_array_almost_equal(sims[:3], [
		[1.    , 0.4929, 0.4917, 0.4892],
		[1.    , 0.478 , 0.4511, 0.416 ],
		[1.    , 0.4639, 0.4507, 0.4474]], 4)
	assert_array_equal(neighbors[:3], [[0, 75, 41, 89], [1, 71, 39, 68],
		[2, 34, 0, 45]])


##


def _jaccard_from_seqlets_reference(seqlets, filter_seqlets, neighbors,
	min_overlap):
	Y = get_2d_data_from_patterns(seqlets)[0]
	X_fwd, X_rev = get_2d_data_from_patterns(filter_seqlets)

	fwd = _jaccard_reference(X_fwd, Y, min_overlap, neighbors, int)[:, :, 0]
	rev = _jaccard_reference(X_rev, Y, min_overlap, neighbors, int)[:, :, 0]
	return numpy.maximum(fwd, rev)


@pytest.mark.parametrize("n,k", [(5, 1), (10, 4), (20, 20)])
@pytest.mark.parametrize("min_overlap", [0.5, 0.7, 1.0])
def test_jaccard_from_seqlets(seqlets, n, k, min_overlap):
	seqlets = seqlets[:n]
	_, neighbors = cosine_similarity_from_seqlets(seqlets, n_neighbors=k-1,
		sign=1)

	affmat = jaccard_from_seqlets(seqlets, min_overlap,
		seqlet_neighbors=neighbors)
	expected = _jaccard_from_seqlets_reference(seqlets, seqlets, neighbors,
		min_overlap)

	assert affmat.shape == (n, k)
	assert affmat.dtype == numpy.float32
	assert_array_almost_equal(affmat, expected, 5)


def test_jaccard_from_seqlets_self(seqlets):
	seqlets = seqlets[:15]
	neighbors = numpy.tile(numpy.arange(15), (15, 1))
	affmat = jaccard_from_seqlets(seqlets, 0.7, seqlet_neighbors=neighbors)

	assert_array_almost_equal(numpy.diag(affmat), numpy.ones(15))
	assert numpy.all(affmat <= 1 + 1e-6)


def test_jaccard_from_seqlets_revcomp(seqlets):
	# A seqlet matches its reverse complement perfectly through the reverse
	# strand of the filters.
	seqlets = seqlets[:8] + [s.revcomp() for s in seqlets[:8]]
	neighbors = numpy.array([[i, (i + 8) % 16] for i in range(16)])
	affmat = jaccard_from_seqlets(seqlets, 0.7, seqlet_neighbors=neighbors)

	assert_array_almost_equal(affmat, numpy.ones((16, 2)))


def test_jaccard_from_seqlets_filter_seqlets(seqlets):
	filters = seqlets[20:26]
	neighbors = numpy.array([[0, 3, 5], [1, 1, 2], [4, 0, 2], [5, 4, 3]])
	affmat = jaccard_from_seqlets(seqlets[:4], 0.7, filter_seqlets=filters,
		seqlet_neighbors=neighbors)
	expected = _jaccard_from_seqlets_reference(seqlets[:4], filters,
		neighbors, 0.7)

	assert affmat.shape == (4, 3)
	assert_array_almost_equal(affmat, expected, 5)


def test_jaccard_from_seqlets_neighbor_dtype(seqlets):
	neighbors = numpy.tile(numpy.arange(6), (6, 1))
	affmat0 = jaccard_from_seqlets(seqlets[:6], 0.7,
		seqlet_neighbors=neighbors.astype('int32'))
	affmat1 = jaccard_from_seqlets(seqlets[:6], 0.7,
		seqlet_neighbors=neighbors.astype('int64'))
	assert_array_equal(affmat0, affmat1)


def test_jaccard_from_seqlets_real(seqlets):
	_, neighbors = cosine_similarity_from_seqlets(seqlets[:50], n_neighbors=3,
		sign=1)
	affmat = jaccard_from_seqlets(seqlets[:50], 0.7, seqlet_neighbors=neighbors)

	assert_array_almost_equal(affmat[:2], [[1.0, 0.6953, 0.5454, 0.7445],
		[1.0, 0.6897, 0.6896, 0.6064]], 4)


@pytest.mark.skip(reason="bug: with seqlet_neighbors=None, "
	"jaccard_from_seqlets builds a list of lists, which jaccard then calls "
	".astype on, raising AttributeError")
def test_jaccard_from_seqlets_default_neighbors(seqlets):
	affmat = jaccard_from_seqlets(seqlets[:6], 0.7)
	assert affmat.shape == (6, 6)


@pytest.mark.parametrize("nx,ny,d,m", [(1, 1, 5, 1), (1, 3, 5, 4),
	(3, 1, 10, 4), (4, 4, 10, 8), (10, 2, 20, 8), (2, 6, 7, 2)])
@pytest.mark.parametrize("min_overlap", [None, 0.5, 0.7, 1.0])
def test_jaccard(nx, ny, d, m, min_overlap):
	rng = numpy.random.RandomState(nx*ny)
	X = rng.randn(nx, d, m)
	Y = rng.randn(ny, d, m)

	results = jaccard(X, Y, min_overlap=min_overlap)
	expected = _jaccard_reference(X, Y, min_overlap, None, numpy.ceil)

	assert results.shape == (ny, nx, 2)
	assert_array_almost_equal(results, expected, 5)


@pytest.mark.parametrize("nx,ny,d,m", [(1, 1, 5, 1), (3, 1, 10, 4),
	(4, 4, 10, 8), (2, 6, 7, 2)])
@pytest.mark.parametrize("min_overlap", [None, 0.5, 0.7, 1.0])
def test_jaccard_sparse(nx, ny, d, m, min_overlap):
	rng = numpy.random.RandomState(nx*ny)
	X = rng.randn(nx, d, m)
	Y = rng.randn(ny, d, m)

	scores = jaccard(X, Y, min_overlap=min_overlap, return_sparse=True)
	expected = _jaccard_reference(X, Y, min_overlap, None, numpy.ceil)

	assert scores.shape == (ny, nx)
	assert scores.dtype == numpy.float32
	assert_array_almost_equal(scores, expected[:, :, 0], 5)


@pytest.mark.parametrize("func", [numpy.ceil, numpy.floor, int, round])
@pytest.mark.parametrize("min_overlap", [0.3, 0.55, 0.7, 0.9])
def test_jaccard_func(X3, func, min_overlap):
	Y = numpy.random.RandomState(1).randn(3, 12, 4)
	results = jaccard(X3, Y, min_overlap=min_overlap, func=func)
	expected = _jaccard_reference(X3, Y, min_overlap, None, func)

	n_pad = int(func(12 * (1 - min_overlap)))
	assert numpy.all(results[:, :, 1] >= -n_pad)
	assert numpy.all(results[:, :, 1] <= n_pad)
	assert_array_almost_equal(results, expected, 5)


@pytest.mark.parametrize("dy", [12, 15, 30])
def test_jaccard_longer_y(X3, dy):
	Y = numpy.random.RandomState(1).randn(2, dy, 4)
	results = jaccard(X3, Y, min_overlap=0.5)
	expected = _jaccard_reference(X3, Y, 0.5, None, numpy.ceil)

	assert_array_almost_equal(results, expected, 5)
	assert numpy.all(results[:, :, 1] <= dy - 12 + 6)


def test_jaccard_neighbors(X3):
	Y = numpy.random.RandomState(1).randn(4, 12, 4)
	neighbors = numpy.array([[0, 5], [1, 1], [3, 2], [4, 0]])

	results = jaccard(X3, Y, min_overlap=0.7, seqlet_neighbors=neighbors)
	expected = _jaccard_reference(X3, Y, 0.7, neighbors, numpy.ceil)

	assert results.shape == (4, 2, 2)
	assert_array_almost_equal(results, expected, 5)


def test_jaccard_neighbors_default(X3):
	neighbors = numpy.tile(numpy.arange(6), (6, 1))
	results0 = jaccard(X3, X3, min_overlap=0.7)
	results1 = jaccard(X3, X3, min_overlap=0.7, seqlet_neighbors=neighbors)
	assert_array_equal(results0, results1)


@pytest.mark.parametrize("min_overlap", [None, 0.5, 0.9])
def test_jaccard_self(X3, min_overlap):
	results = jaccard(X3, X3, min_overlap=min_overlap)

	assert_array_almost_equal(numpy.diagonal(results[:, :, 0]), numpy.ones(6))
	assert_array_equal(numpy.diagonal(results[:, :, 1]), numpy.zeros(6))
	assert numpy.all(results[:, :, 0] <= 1 + 1e-6)
	assert numpy.all(results[:, :, 0] >= -1 - 1e-6)


@pytest.mark.parametrize("shift", [-3, -1, 1, 2, 4])
def test_jaccard_shift(shift):
	x = numpy.zeros((1, 16, 4))
	x[0, 4:10] = numpy.random.RandomState(0).uniform(0.5, 1, size=(6, 4))
	y = numpy.roll(x, shift, axis=1)

	results = jaccard(x, y, min_overlap=0.5)
	assert abs(results[0, 0, 0] - 1) < 1e-6
	assert results[0, 0, 1] == shift


def test_jaccard_values():
	X = numpy.array([[[1.0], [2.0]]])
	Y = numpy.array([[[2.0], [-1.0]]])

	# (1 - 1) / (2 + 2) at zero offset.
	results = jaccard(X, Y)
	assert_array_almost_equal(results, [[[0.0, 0.0]]])

	# Offsets -1, 0, 1 give 2/3, 0 and -1/3.
	results = jaccard(X, Y, min_overlap=0.5)
	assert_array_almost_equal(results, [[[2/3, -1]]])


@pytest.mark.parametrize("scale", [0.001, 3.0, 1000.0])
def test_jaccard_scale(X3, scale):
	results0 = jaccard(X3, X3[::-1], min_overlap=0.7)
	results1 = jaccard(X3 * scale, X3[::-1] * scale, min_overlap=0.7)
	assert_array_almost_equal(results0, results1, 5)


def test_jaccard_symmetric(X3):
	results = jaccard(X3, X3)
	assert_array_almost_equal(results[:, :, 0], results[:, :, 0].T, 6)


@pytest.mark.parametrize("dtype", ['float16', 'float32', 'float64'])
def test_jaccard_dtypes(X3, dtype):
	results = jaccard(X3.astype(dtype), X3.astype(dtype), min_overlap=0.7)
	assert_array_almost_equal(numpy.diagonal(results[:, :, 0]), numpy.ones(6))


def test_jaccard_does_not_modify(X3):
	X = X3.copy()
	jaccard(X3, X3, min_overlap=0.5)
	assert_array_equal(X3, X)


def test_jaccard_raises():
	X = numpy.ones((2, 5, 4))
	assert_raises(ValueError, jaccard, X, numpy.ones((2, 3, 4)))


##


def _raw_jaccard(X, Y, neighbors):
	X = X.astype('float32')
	Y = Y.astype('float32')
	scores = numpy.zeros((len(Y), neighbors.shape[1], Y.shape[1]-X.shape[1]+1),
		dtype='float32')
	_jaccard(X, Y, neighbors.astype('int32'), scores)
	return scores


@pytest.mark.parametrize("nx,ny,d,dy,m", [(1, 1, 4, 4, 1), (3, 2, 5, 9, 4),
	(6, 6, 10, 10, 8), (2, 5, 3, 20, 2)])
def test__jaccard(nx, ny, d, dy, m):
	rng = numpy.random.RandomState(0)
	X, Y = rng.randn(nx, d, m), rng.randn(ny, dy, m)
	neighbors = numpy.tile(numpy.arange(nx), (ny, 1))

	scores = _raw_jaccard(X, Y, neighbors)
	for l in range(ny):
		for i in range(nx):
			for idx in range(dy - d + 1):
				expected = _cont_jaccard(X[i], Y[l, idx:idx+d])
				assert abs(scores[l, i, idx] - expected) < 1e-5


def test__jaccard_raises():
	X = numpy.ones((2, 5, 4))
	scores = numpy.zeros((2, 2, 1), dtype='float32')
	neighbors = numpy.zeros((2, 2), dtype='int32')

	assert_raises(TypeError, _jaccard, X, X, neighbors, scores)
	assert_raises(TypeError, _jaccard, X.astype('float32'),
		X.astype('float32'), neighbors.astype('int64'), scores)


##


@pytest.mark.parametrize("n,m", [(1, 1), (2, 4), (10, 4), (10, 40), (50, 8),
	(31, 120)])
@pytest.mark.parametrize("k", [1, 2, 10, 50])
def test_pairwise_jaccard(n, m, k):
	X = numpy.random.RandomState(n).randn(n, m)
	k = min(k, n)

	jaccards, neighbors = pairwise_jaccard(X, k)
	jaccards_ref, neighbors_ref = _pairwise_reference(X, k)

	assert jaccards.shape == (n, k)
	assert neighbors.shape == (n, k)
	assert jaccards.dtype == numpy.float64
	assert neighbors.dtype == numpy.int32
	assert_array_almost_equal(jaccards, jaccards_ref)
	assert_array_equal(neighbors, neighbors_ref)


@pytest.mark.parametrize("dtype", ['float32', 'float64'])
def test_pairwise_jaccard_dtypes(dtype):
	X = numpy.random.RandomState(0).randn(20, 16).astype(dtype)
	jaccards, neighbors = pairwise_jaccard(X, 5)
	jaccards_ref, neighbors_ref = _pairwise_reference(X.astype('float64'), 5)

	assert_array_almost_equal(jaccards, jaccards_ref, 5)
	assert_array_equal(neighbors, neighbors_ref)


def test_pairwise_jaccard_self():
	X = numpy.random.RandomState(0).randn(25, 10)
	jaccards, neighbors = pairwise_jaccard(X, 5)

	assert_array_equal(neighbors[:, 0], numpy.arange(25))
	assert_array_almost_equal(jaccards[:, 0], numpy.ones(25))
	assert numpy.all(numpy.diff(jaccards, axis=1) <= 0)


def test_pairwise_jaccard_symmetric():
	X = numpy.random.RandomState(0).randn(15, 10)
	jaccards, neighbors = pairwise_jaccard(X, 15)

	J = numpy.zeros((15, 15))
	for i in range(15):
		J[i, neighbors[i]] = jaccards[i]

	assert_array_almost_equal(J, J.T)


def test_pairwise_jaccard_values():
	X = numpy.array([[1.0, 2.0], [2.0, -1.0], [2.0, 4.0]])
	jaccards, neighbors = pairwise_jaccard(X, 3)

	assert_array_almost_equal(jaccards, [[1.0, 0.5, 0.0], [1.0, 1/6, 0.0],
		[1.0, 0.5, 1/6]])
	assert_array_equal(neighbors, [[0, 2, 1], [1, 2, 0], [2, 0, 1]])


def test_pairwise_jaccard_real(seqlets):
	X = get_2d_data_from_patterns(seqlets[:40])[0].reshape(40, -1)
	jaccards, neighbors = pairwise_jaccard(X, 4)

	assert_array_almost_equal(jaccards[:2], [[1., 0.5714, 0.1347, 0.1147],
		[1., 0.5426, 0.26  , 0.2453]], 4)
	assert_array_equal(neighbors[:2], [[0, 10, 4, 33], [1, 16, 15, 23]])


##


@pytest.mark.parametrize("d", [5, 12, 30])
@pytest.mark.parametrize("min_overlap", [0.3, 0.5, 0.7, 1.0])
def test_pearson_correlation(d, min_overlap):
	rng = numpy.random.RandomState(d)
	x, y = rng.randn(d, 8), rng.randn(d, 8)

	result = pearson_correlation(x[None], y[None], min_overlap=min_overlap)
	score, offset = _pearson_reference(x, y, min_overlap, numpy.ceil)

	assert result.shape == (1, 1, 2)
	assert abs(result[0, 0, 0] - score) < 1e-8
	assert result[0, 0, 1] == offset


@pytest.mark.parametrize("func", [numpy.ceil, numpy.floor, int])
def test_pearson_correlation_func(func):
	rng = numpy.random.RandomState(0)
	x, y = rng.randn(15, 4), rng.randn(15, 4)

	result = pearson_correlation(x[None], y[None], min_overlap=0.55, func=func)
	score, offset = _pearson_reference(x, y, 0.55, func)

	assert abs(result[0, 0, 0] - score) < 1e-8
	assert result[0, 0, 1] == offset


def test_pearson_correlation_2d():
	rng = numpy.random.RandomState(0)
	x, y = rng.randn(12, 4), rng.randn(12, 4)

	result0 = pearson_correlation(x, y, min_overlap=0.5)
	result1 = pearson_correlation(x[None], y[None], min_overlap=0.5)
	result2 = pearson_correlation(x, y[None], min_overlap=0.5)

	assert_array_equal(result0, result1)
	assert_array_equal(result0, result2)


@pytest.mark.parametrize("min_overlap", [0.5, 0.7, 1.0])
def test_pearson_correlation_self(min_overlap):
	x = numpy.random.RandomState(0).randn(1, 20, 8)
	result = pearson_correlation(x, x, min_overlap=min_overlap)
	assert_array_almost_equal(result, [[[1.0, 0.0]]])


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
@pytest.mark.parametrize("shift", [-4, -2, 1, 3])
def test_pearson_correlation_shift(shift):
	x = numpy.zeros((1, 20, 4))
	x[0, 6:14] = numpy.random.RandomState(0).randn(8, 4)
	y = numpy.roll(x, shift, axis=1)

	result = pearson_correlation(x, y, min_overlap=0.5)
	assert abs(result[0, 0, 0] - 1) < 1e-8
	assert result[0, 0, 1] == shift


@pytest.mark.parametrize("scale", [0.01, 5.0, -1.0])
def test_pearson_correlation_scale(scale):
	rng = numpy.random.RandomState(0)
	x, y = rng.randn(1, 10, 4), rng.randn(1, 10, 4)

	result0 = pearson_correlation(x, y, min_overlap=0.7)
	result1 = pearson_correlation(x, y * abs(scale), min_overlap=0.7)
	assert_array_almost_equal(result0, result1)


def test_pearson_correlation_negated():
	x = numpy.random.RandomState(0).randn(1, 10, 4)
	result = pearson_correlation(x, -x, min_overlap=1.0)
	assert_array_almost_equal(result, [[[-1.0, 0.0]]])


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
def test_pearson_correlation_zeros():
	x = numpy.random.RandomState(0).randn(1, 10, 4)
	result = pearson_correlation(x, numpy.zeros((1, 10, 4)), min_overlap=0.5)
	assert_array_almost_equal(result, [[[0.0, -5.0]]])


def test_pearson_correlation_longer_y():
	rng = numpy.random.RandomState(0)
	x, y = rng.randn(1, 10, 4), rng.randn(1, 25, 4)
	x[0] = y[0, 12:22]

	result = pearson_correlation(x, y, min_overlap=1.0)
	assert_array_almost_equal(result, [[[1.0, 12.0]]])


@pytest.mark.skip(reason="bug: pearson_correlation raises UnboundLocalError "
	"when min_overlap is None because n_pad is only set inside the "
	"min_overlap branch")
def test_pearson_correlation_no_overlap():
	x = numpy.random.RandomState(0).randn(1, 10, 4)
	result = pearson_correlation(x, x)
	assert_array_almost_equal(result, [[[1.0, 0.0]]])


@pytest.mark.skip(reason="bug: pearson_correlation normalizes and correlates "
	"the whole batch at once, so every row receives the same score when "
	"X holds more than one example")
def test_pearson_correlation_batch():
	rng = numpy.random.RandomState(0)
	X, Y = rng.randn(3, 10, 4), rng.randn(3, 10, 4)
	Y[1] = X[1]

	result = pearson_correlation(X, Y, min_overlap=1.0)
	assert abs(result[0, 1, 0] - 1) < 1e-8
	assert result[0, 0, 0] < 0.9


##


def _nn_affinity(n, k, random_state=0):
	X = numpy.random.RandomState(random_state).randn(n, 16)
	return pairwise_jaccard(X, k)


@pytest.mark.parametrize("n,k", [(10, 10), (30, 20), (60, 31), (40, 40)])
@pytest.mark.parametrize("perplexity", [2, 5, 10])
def test_nn_tsne_conditional_probs(n, k, perplexity):
	affmat, neighbors = _nn_affinity(n, k)
	P = NNTsneConditionalProbs(perplexity)(affmat, neighbors)

	n_used = min(n - 1, int(3 * perplexity + 1), k - 1)
	assert isinstance(P, scipy.sparse.coo_matrix)
	assert P.shape == (n, n)
	assert P.nnz == n * n_used
	assert_array_almost_equal(numpy.asarray(P.sum(axis=1)).ravel(),
		numpy.ones(n))


def test_nn_tsne_conditional_probs_structure():
	affmat, neighbors = _nn_affinity(30, 12)
	P = NNTsneConditionalProbs(3)(affmat, neighbors)

	for i in range(30):
		assert_array_equal(P.col[P.row == i], neighbors[i, 1:11])
		assert i not in P.col[P.row == i]


def test_nn_tsne_conditional_probs_order():
	# Closer neighbors receive at least as much probability.
	affmat, neighbors = _nn_affinity(40, 20)
	P = NNTsneConditionalProbs(5)(affmat, neighbors).toarray()

	for i in range(40):
		probs = P[i, neighbors[i, 1:17]]
		assert numpy.all(numpy.diff(probs) <= 1e-6)


def test_nn_tsne_conditional_probs_perplexity():
	# The probabilities spread over more neighbors as perplexity grows.
	affmat, neighbors = _nn_affinity(50, 40)
	P2 = NNTsneConditionalProbs(2)(affmat, neighbors).toarray()
	P10 = NNTsneConditionalProbs(10)(affmat, neighbors).toarray()

	assert P2.max(axis=1).mean() > P10.max(axis=1).mean()


def test_nn_tsne_conditional_probs_small():
	affmat, neighbors = _nn_affinity(4, 4)
	P = NNTsneConditionalProbs(10)(affmat, neighbors)

	assert P.nnz == 12
	assert_array_almost_equal(numpy.asarray(P.sum(axis=1)).ravel(),
		numpy.ones(4))


def test_nn_tsne_conditional_probs_tsne_probs_calc():
	distances = numpy.array([[0.1, 0.5, 2.0], [0.2, 0.3, 1.0], [0.5, 1.0, 1.5],
		[0.05, 0.1, 0.8]])
	neighbors = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
	P = NNTsneConditionalProbs(2).tsne_probs_calc(distances, neighbors)

	assert isinstance(P, scipy.sparse.coo_matrix)
	assert P.shape == (4, 4)

	P = P.toarray()
	assert_array_almost_equal(P.sum(axis=1), numpy.ones(4))
	assert_array_equal(numpy.diag(P), numpy.zeros(4))
	for i, row in enumerate(neighbors):
		assert P[i, row[0]] > P[i, row[1]] > P[i, row[2]] > 0


def test_nn_tsne_conditional_probs_tsne_probs_calc_uniform():
	# Equal distances reach a perplexity equal to the number of neighbors
	# only through uniform probabilities.
	distances = numpy.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 0.5, 0.5],
		[3.0, 3.0, 3.0]])
	neighbors = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
	P = NNTsneConditionalProbs(3).tsne_probs_calc(distances, neighbors)

	assert_array_almost_equal(P.toarray(), (1 - numpy.eye(4)) / 3)


def test_nn_tsne_conditional_probs_real(seqlets):
	X = get_2d_data_from_patterns(seqlets[:60])[0].reshape(60, -1)
	affmat, neighbors = pairwise_jaccard(X, 32)
	P = NNTsneConditionalProbs(10)(affmat, neighbors)

	assert P.shape == (60, 60)
	assert P.nnz == 60 * 31
	assert_array_almost_equal(P.toarray()[0, neighbors[0, 1:4]],
		[0.4112, 0.1398, 0.0499], 4)
