# cluster.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar

import sys
import itertools

import leidenalg
import numpy as np
import igraph as ig

from . import forked


# Edges handed to igraph per chunk in _graph (about 1.3 MB of Python ints).
_EDGE_CHUNK = 16384


def _leiden(g, weights, seed, n_leiden_iterations):
	"""One Leiden run from one seed: its quality and its membership."""

	partition = leidenalg.find_partition(
		graph=g,
		partition_type=leidenalg.ModularityVertexPartition,
		weights=weights,
		n_iterations=n_leiden_iterations,
		initial_membership=None,
		seed=seed*100)

	return np.array(partition.quality()), np.array(partition.membership)


def _n_workers(n_seeds, n_edges):
	"""The number of processes to spread the seeds over: one per seed, at most
	this process's thread budget (numba's thread count in the top-level
	process, a share of it in a forked worker; forked.budget). leidenalg
	holds the GIL for the whole of a run, so threads would not overlap; the
	extra processes are forked, which is done only on Linux. Below about
	10,000 edges a Leiden run takes no longer than forking a child and
	reaping it (~10 ms), so small graphs stay in this process."""

	if n_seeds < 2 or n_edges < 10000 or not sys.platform.startswith('linux'):
		return 1

	return max(1, min(n_seeds, forked.budget()))


def _leiden_forked(g, weights, seeds, n_leiden_iterations, n_workers):
	"""Run the seeds over this process and n_workers-1 forked children,
	worker k taking seeds[k::n_workers] (forked.forked_map, which runs a
	failed child's seeds again here). The children call only leidenalg,
	numpy and pickle, never numba. Returns {seed: (quality, membership)}."""

	def run(share):
		return {seed: _leiden(g, weights, seed, n_leiden_iterations)
			for seed in share}

	results = {}
	for share_results in forked.forked_map(run,
			[seeds[k::n_workers] for k in range(n_workers)]):
		results.update(share_results)

	return results


def _graph(affinity_mat):
	"""The undirected graph that Leiden runs on, and its edge weights.

	Every stored entry (i, j) of the CSR matrix is an undirected edge, so a
	symmetric matrix gives two parallel edges of weight w for each pair of
	distinct vertices and one self-loop for each diagonal entry. When the
	matrix is exactly symmetric, bit for bit, with sorted indices, each pair
	is added once instead, from the upper triangle in CSR order, with weight
	w + w (exact in floating point), and each diagonal entry once as before.
	The strengths, the total weight and the modularity of every partition
	are the same in exact arithmetic, and Leiden goes over half the edges.

	Leiden sums node strengths and community weights over the edges, and the
	two graphs sum them in different orders, so the doubles it compares can
	differ in the last bits. A move whose gain ties another's to within that
	rounding could then go the other way. A matrix that is not exactly
	symmetric is added entry by entry.
	"""

	n_vertices = affinity_mat.shape[0]
	indptr, indices, data = (affinity_mat.indptr, affinity_mat.indices,
		affinity_mat.data)

	transpose = affinity_mat.T.tocsr()
	symmetric = (np.array_equal(transpose.indptr, indptr) and
		np.array_equal(transpose.indices, indices) and
		_same_bytes(transpose.data, data))
	del transpose

	# The row of each entry, in the indices' own integer type, which holds
	# every vertex index.
	sources = np.repeat(np.arange(n_vertices, dtype=indices.dtype),
		np.diff(indptr))
	if symmetric:
		upper = indices >= sources
		sources, targets, weights = sources[upper], indices[upper], data[upper]
		del upper
		weights = np.where(sources == targets, weights, weights + weights)

		# Every index of a symmetric matrix is also a row, below n_vertices,
		# so the pairs can hold the Python ints of `vertices`, made once,
		# rather than two new ints per edge.
		vertices = np.arange(n_vertices).astype(object)
		to_ints = lambda chunk: vertices[chunk].tolist()
	else:
		targets, weights = indices, data
		to_ints = np.ndarray.tolist

	# python-igraph's abi3 build turns an ndarray edge list into a nested
	# Python list (memoryview.tolist()), about 140 bytes per edge, before it
	# reads a single pair. It reads any other iterable one pair at a time, so
	# pairs taken from a chunk at a time give the same edges, in the same
	# order, while only one chunk's pairs exist.
	edges = itertools.chain.from_iterable(
		zip(to_ints(sources[start:start+_EDGE_CHUNK]),
			to_ints(targets[start:start+_EDGE_CHUNK]))
		for start in range(0, len(sources), _EDGE_CHUNK))

	g = ig.Graph(n=n_vertices, edges=edges, directed=False)
	return g, weights


def _same_bytes(a, b):
	"""Whether two arrays of one dtype and shape hold the same bytes, as
	a.tobytes() == b.tobytes(), compared as unsigned integers of the item
	size, without copying either array."""

	if (a.dtype != b.dtype or a.shape != b.shape or a.dtype.kind not in
			'biufcmM' or a.dtype.itemsize not in (1, 2, 4, 8)):
		return a.tobytes() == b.tobytes()

	unsigned = np.dtype('u{}'.format(a.dtype.itemsize))
	return np.array_equal(a.view(unsigned), b.view(unsigned))


def LeidenCluster(affinity_mat, n_seeds=2, n_leiden_iterations=-1):
	g, weights = _graph(affinity_mat)

	seeds = list(range(1, n_seeds+1))
	n_workers = _n_workers(n_seeds, len(affinity_mat.data))
	if n_workers > 1:
		results = _leiden_forked(g, weights, seeds,
			n_leiden_iterations, n_workers)

	best_clustering = None
	best_quality = None

	for seed in seeds:
		if n_workers > 1:
			quality, membership = results[seed]
		else:
			quality, membership = _leiden(g, weights, seed,
				n_leiden_iterations)

		if best_quality is None or quality > best_quality:
			best_quality = quality
			best_clustering = membership

	return best_clustering
