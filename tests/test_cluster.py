# test_cluster.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import scipy.linalg
import scipy.sparse
import pytest

import igraph
import leidenalg

from modiscolite.cluster import LeidenCluster

from numpy.testing import assert_raises
from numpy.testing import assert_array_equal


def _block_affinity(sizes, within=1.0, between=0.01, noise=0.0,
	random_state=0):
	"""Return a dense block-diagonal affinity matrix and the true labels."""

	labels = numpy.concatenate([numpy.full(size, i) for i, size in
		enumerate(sizes)])
	same = labels[:, None] == labels[None, :]
	X = numpy.where(same, within, between)

	if noise > 0:
		rng = numpy.random.RandomState(random_state)
		X = X + rng.uniform(0, noise, size=X.shape)
		X = (X + X.T) / 2

	numpy.fill_diagonal(X, 0)
	return X, labels


def _same_partition(y, labels):
	"""Whether two labelings define the same partition."""

	pairs = set(zip(y.tolist(), labels.tolist()))
	return len(pairs) == len(set(y.tolist())) == len(set(labels.tolist()))


def _reference(affinity_mat, n_seeds, n_leiden_iterations):
	n = affinity_mat.shape[0]
	sources = numpy.repeat(numpy.arange(n), numpy.diff(affinity_mat.indptr))
	g = igraph.Graph()
	g.add_vertices(n)
	g.add_edges(zip(sources, affinity_mat.indices))

	best, best_quality = None, None
	for seed in range(1, n_seeds+1):
		partition = leidenalg.find_partition(g,
			leidenalg.ModularityVertexPartition, weights=affinity_mat.data,
			n_iterations=n_leiden_iterations, seed=seed*100)

		if best_quality is None or partition.quality() > best_quality:
			best, best_quality = partition.membership, partition.quality()

	return numpy.array(best)


##


@pytest.mark.parametrize("k", [2, 3, 4, 6])
@pytest.mark.parametrize("size", [5, 10, 20])
def test_leiden_cluster_blocks(k, size):
	X, labels = _block_affinity([size] * k)
	y = LeidenCluster(scipy.sparse.csr_matrix(X))

	assert y.shape == (k * size,)
	assert _same_partition(y, labels)


@pytest.mark.parametrize("sizes", [[3, 30], [5, 10, 15], [2, 2, 2, 2],
	[25, 5, 12, 8]])
def test_leiden_cluster_uneven_blocks(sizes):
	X, labels = _block_affinity(sizes)
	y = LeidenCluster(scipy.sparse.csr_matrix(X))
	assert _same_partition(y, labels)


@pytest.mark.parametrize("n_seeds", [1, 2, 5, 10])
def test_leiden_cluster_n_seeds(n_seeds):
	X, labels = _block_affinity([8, 12, 10], noise=0.2)
	y = LeidenCluster(scipy.sparse.csr_matrix(X), n_seeds=n_seeds)
	assert _same_partition(y, labels)


@pytest.mark.parametrize("n_leiden_iterations", [-1, 1, 2, 10])
def test_leiden_cluster_iterations(n_leiden_iterations):
	X, labels = _block_affinity([8, 12, 10], noise=0.2)
	y = LeidenCluster(scipy.sparse.csr_matrix(X),
		n_leiden_iterations=n_leiden_iterations)
	assert _same_partition(y, labels)


@pytest.mark.parametrize("n_seeds", [1, 2, 3, 7])
@pytest.mark.parametrize("n_leiden_iterations", [-1, 2])
def test_leiden_cluster_reference(n_seeds, n_leiden_iterations):
	rng = numpy.random.RandomState(n_seeds)
	X = scipy.sparse.random(60, 60, density=0.2, random_state=rng)
	X = scipy.sparse.csr_matrix(X + X.T)

	y = LeidenCluster(X, n_seeds=n_seeds,
		n_leiden_iterations=n_leiden_iterations)
	assert_array_equal(y, _reference(X, n_seeds, n_leiden_iterations))


def test_leiden_cluster_labels():
	X, labels = _block_affinity([10, 10, 10, 10])
	y = LeidenCluster(scipy.sparse.csr_matrix(X))

	assert isinstance(y, numpy.ndarray)
	assert y.dtype.kind == 'i'
	assert_array_equal(numpy.unique(y), numpy.arange(4))


def test_leiden_cluster_largest_first():
	# Leiden numbers communities from largest to smallest.
	X, labels = _block_affinity([4, 20, 9])
	y = LeidenCluster(scipy.sparse.csr_matrix(X))
	assert_array_equal(numpy.bincount(y), [20, 9, 4])


def test_leiden_cluster_deterministic():
	rng = numpy.random.RandomState(0)
	X = scipy.sparse.random(80, 80, density=0.1, random_state=rng)
	X = scipy.sparse.csr_matrix(X + X.T)

	y0 = LeidenCluster(X, n_seeds=3)
	y1 = LeidenCluster(X, n_seeds=3)
	assert_array_equal(y0, y1)


def test_leiden_cluster_weights():
	# Two groups of five, fully connected. The within-group edges carry the
	# weight in the first matrix and the between-group edges in the second.
	labels = numpy.repeat([0, 1], 5)
	same = labels[:, None] == labels[None, :]
	X0 = numpy.where(same, 1.0, 0.01)
	X1 = numpy.where(same, 0.01, 1.0)
	numpy.fill_diagonal(X0, 0)
	numpy.fill_diagonal(X1, 0)

	y0 = LeidenCluster(scipy.sparse.csr_matrix(X0))
	y1 = LeidenCluster(scipy.sparse.csr_matrix(X1))

	assert _same_partition(y0, labels)
	assert not _same_partition(y1, labels)


def test_leiden_cluster_disconnected():
	X = scipy.sparse.csr_matrix(scipy.linalg.block_diag(numpy.ones((6, 6)),
		numpy.ones((4, 4)), numpy.ones((3, 3))))
	y = LeidenCluster(X)

	assert _same_partition(y, numpy.repeat([0, 1, 2], [6, 4, 3]))


def test_leiden_cluster_asymmetric():
	# Only the upper triangle is stored; the graph is still undirected.
	X, labels = _block_affinity([6, 6, 6])
	y = LeidenCluster(scipy.sparse.csr_matrix(numpy.triu(X)))
	assert _same_partition(y, labels)


def test_leiden_cluster_float32():
	X, labels = _block_affinity([7, 9])
	y = LeidenCluster(scipy.sparse.csr_matrix(X.astype('float32')))
	assert _same_partition(y, labels)


def test_leiden_cluster_raises():
	X, _ = _block_affinity([5, 5])
	assert_raises(AttributeError, LeidenCluster, scipy.sparse.coo_matrix(X))
