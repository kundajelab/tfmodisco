# cluster.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar

import os
import sys
import pickle
import signal
import warnings

import numba
import leidenalg
import numpy as np
import igraph as ig


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
	the numba thread budget. leidenalg holds the GIL for the whole of a run,
	so threads would not overlap; the extra processes are forked, which is
	done only on Linux. Below about 10,000 edges a Leiden run takes no longer
	than forking a child and reaping it (~10 ms), so small graphs stay in
	this process."""

	if n_seeds < 2 or n_edges < 10000 or not sys.platform.startswith('linux'):
		return 1

	return max(1, min(n_seeds, numba.config.NUMBA_NUM_THREADS))


def _leiden_forked(g, weights, seeds, n_leiden_iterations, n_workers):
	"""Run the seeds over this process and n_workers-1 forked children,
	worker k taking seeds[k::n_workers]. A child sends its results back
	pickled over a pipe. The seeds of a child that could not be forked or
	that failed are run again here, so the results never depend on whether
	a child succeeded. Returns {seed: (quality, membership)}."""

	children, results = [], {}
	try:
		for k in range(1, n_workers):
			read_fd, write_fd = os.pipe()
			try:
				with warnings.catch_warnings():
					# Python >= 3.12 warns when a multi-threaded process forks.
					# The child calls only leidenalg, numpy and pickle, never
					# numba or BLAS, and leaves through os._exit.
					warnings.simplefilter('ignore', DeprecationWarning)
					pid = os.fork()
			except OSError:
				os.close(read_fd)
				os.close(write_fd)
				children.append((None, None, k))
				continue

			if pid == 0:
				code = 1
				try:
					os.close(read_fd)
					child_results = {seed: _leiden(g, weights, seed,
						n_leiden_iterations) for seed in seeds[k::n_workers]}
					with os.fdopen(write_fd, 'wb') as handle:
						pickle.dump(child_results, handle,
							protocol=pickle.HIGHEST_PROTOCOL)
					code = 0
				finally:
					os._exit(code)

			os.close(write_fd)
			children.append((pid, os.fdopen(read_fd, 'rb'), k))

		for seed in seeds[0::n_workers]:
			results[seed] = _leiden(g, weights, seed, n_leiden_iterations)

		while children:
			pid, handle, k = children[0]
			child_results = None
			if pid is not None:
				data = handle.read()
				handle.close()
				_, status = os.waitpid(pid, 0)
				children[0] = (None, None, k)

				if os.waitstatus_to_exitcode(status) == 0:
					try:
						child_results = pickle.loads(data)
					except Exception:
						pass

			if child_results is None:
				child_results = {seed: _leiden(g, weights, seed,
					n_leiden_iterations) for seed in seeds[k::n_workers]}

			results.update(child_results)
			children.pop(0)

	finally:
		for pid, handle, k in children:
			if pid is not None:
				handle.close()
				try:
					os.kill(pid, signal.SIGKILL)
					os.waitpid(pid, 0)
				except OSError:
					pass

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
