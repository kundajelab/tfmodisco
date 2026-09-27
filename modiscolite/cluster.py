# cluster.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar

import sys

import leidenalg
import numpy as np
import igraph as ig

from . import forked


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


def LeidenCluster(affinity_mat, n_seeds=2, n_leiden_iterations=-1):
	n_vertices = affinity_mat.shape[0]
	n_cols = affinity_mat.indptr
	sources = np.concatenate([np.ones(n_cols[i+1] - n_cols[i], dtype='int32') * i for i in range(n_vertices)])

	g = ig.Graph(directed=None)
	g.add_vertices(n_vertices)
	g.add_edges(zip(sources, affinity_mat.indices))

	seeds = list(range(1, n_seeds+1))
	n_workers = _n_workers(n_seeds, len(affinity_mat.data))
	if n_workers > 1:
		results = _leiden_forked(g, affinity_mat.data, seeds,
			n_leiden_iterations, n_workers)

	best_clustering = None
	best_quality = None

	for seed in seeds:
		if n_workers > 1:
			quality, membership = results[seed]
		else:
			quality, membership = _leiden(g, affinity_mat.data, seed,
				n_leiden_iterations)

		if best_quality is None or quality > best_quality:
			best_quality = quality
			best_clustering = membership

	return best_clustering
