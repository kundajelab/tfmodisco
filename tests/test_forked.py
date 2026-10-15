"""Regression tests for runtime thread limits and explicit worker budgets."""

import os

import numba
import pytest

from modiscolite import cluster
from modiscolite import forked
from modiscolite import precompile


@pytest.fixture
def runtime_budget(monkeypatch):
	threads = numba.get_num_threads()
	monkeypatch.setattr(forked, '_budget', None)
	try:
		yield
	finally:
		numba.set_num_threads(threads)


def test_budget_follows_runtime_thread_limit(runtime_budget):
	for limit in range(1, min(2, numba.config.NUMBA_NUM_THREADS) + 1):
		numba.set_num_threads(limit)
		assert forked.budget() == limit


def test_budget_does_not_initialize_threading_layer(runtime_budget, monkeypatch):
	def uninitialized_layer():
		raise ValueError('threading layer is not initialized')

	def unexpected_pool_start():
		pytest.fail('reading a cold budget must not initialize the thread pool')

	monkeypatch.setattr(numba, 'threading_layer', uninitialized_layer)
	monkeypatch.setattr(numba, 'get_num_threads', unexpected_pool_start)
	assert forked.budget() == numba.config.NUMBA_NUM_THREADS


def test_single_thread_does_not_fork_tasks(runtime_budget, monkeypatch):
	numba.set_num_threads(1)
	monkeypatch.setattr(forked, 'numba_fork_safe', lambda: True)

	def unexpected_fork(*args, **kwargs):
		pytest.fail('a one-thread runtime budget must not fork task workers')

	monkeypatch.setattr(forked, 'forked_map', unexpected_fork)
	assert forked.map_tasks(lambda task: task * task, [1, 2, 3],
		weights=[50000, 50000, 50000]) == {1: 1, 2: 4, 3: 9}
	assert numba.get_num_threads() == 1


def test_single_thread_does_not_fork_leiden(runtime_budget):
	numba.set_num_threads(1)
	assert cluster._n_workers(n_seeds=10, n_edges=10000) == 1


def test_single_thread_does_not_fork_compilers(runtime_budget, monkeypatch):
	numba.set_num_threads(1)

	@numba.njit
	def pending_kernel():
		return 0

	# Isolate scheduling from the platform's fork safety and cache location.
	monkeypatch.setattr(precompile, '_started', False)
	monkeypatch.setattr(precompile, '_children', [])
	monkeypatch.setattr(precompile, '_owner', None)
	monkeypatch.setattr(precompile, '_can_fork', lambda: True)
	monkeypatch.setattr(precompile, '_kernels', lambda: [[(pending_kernel, ())]])
	monkeypatch.setattr(pending_kernel, 'enable_caching', lambda: None)
	fork_attempts = []

	def unavailable_fork():
		fork_attempts.append(True)
		raise OSError('fork unavailable')

	monkeypatch.setattr(os, 'fork', unavailable_fork)
	precompile.start()
	assert fork_attempts == []
	assert precompile._children == []
	assert numba.get_num_threads() == 1


@pytest.mark.parametrize('raise_error', [False, True])
def test_explicit_budget_survives_internal_thread_limit(runtime_budget,
		raise_error):
	if numba.config.NUMBA_NUM_THREADS < 2:
		pytest.skip('requires two configured Numba threads')

	numba.set_num_threads(2)

	def scoped_work():
		with forked.budget_of(2):
			numba.set_num_threads(1)
			assert forked.budget() == 2
			with forked.budget_of(1):
				assert forked.budget() == 1
			assert forked.budget() == 2
			assert numba.get_num_threads() == 1
			if raise_error:
				raise ValueError('worker failed')

	if raise_error:
		with pytest.raises(ValueError, match='worker failed'):
			scoped_work()
	else:
		scoped_work()

	assert forked._budget is None
	assert forked.budget() == numba.get_num_threads() == 2
