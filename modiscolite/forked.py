# forked.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""Run independent pieces of work in forked child processes.

Each child inherits the parent's memory copy-on-write, so nothing is pickled
on the way in, and sends its result back pickled over a pipe. The work of a
child that could not be forked or that failed is done again in the parent,
so a result never depends on whether a child succeeded.

The processes share one budget of threads, numba's thread count
(`numba.config.NUMBA_NUM_THREADS`) in the top-level process. A child gets a
share of the budget: numba's parallel kernels run with that many threads in
it, and anything it forks in turn (the Leiden seeds) stays within it.
"""

import os
import sys
import pickle
import signal
import warnings
import threading

import numba


# The budget of this process: None in the top-level process, which has
# numba's whole thread count, and the share it was given in a forked child
# (or, in the parent, while it runs its own share).
_budget = None

# A child's share of the tasks in map_tasks is worth a fork only when it
# holds at least this much weight, in units of seqlets squared: about
# 40-80 ms of subclustering, against 10-20 ms to fork a child and reap it.
MIN_CHILD_WEIGHT = 100 ** 2


def budget():
	"""The number of threads this process may keep busy."""

	if _budget is None:
		return numba.config.NUMBA_NUM_THREADS
	return _budget


def numba_fork_safe():
	"""Whether a forked child may run numba's parallel kernels.

	On Linux, numba counts its TBB and workqueue threading layers as safe to
	use in a child forked after the parent has used them (its 'forksafe'
	layer choice), and not OpenMP, which is GNU OpenMP there. The layer must
	already be running, so that the check is about the layer the children
	inherit, and the fork must come from the main thread, which numba's TBB
	fork handler requires.
	"""

	if not sys.platform.startswith('linux'):
		return False

	if threading.current_thread() is not threading.main_thread():
		return False

	try:
		layer = numba.threading_layer()
	except ValueError:
		return False

	return layer in ('tbb', 'workqueue')


def _set_budget(n, n_threads=None):
	"""Set this process's budget to n (None: numba's whole thread count)
	and numba's thread count to n_threads, by default n."""

	global _budget
	_budget = n
	numba.set_num_threads(n if n_threads is None else n_threads)


def forked_map(func, shares, budgets=None, pack=None, unpack=None):
	"""Return [func(share) for share in shares], computing shares[1:] in
	forked children, one child each, while this process computes shares[0].

	A child returns pack(func(share)) over a pipe, and this process applies
	unpack to what it receives; both default to the identity, and neither is
	applied to a share this process computes itself. With budgets, child k
	sets its budget and numba's thread count to budgets[k] before it starts,
	and this process runs shares[0] with budgets[0]; without them (a child
	that never calls numba) nothing is changed. If a fork raises, or a child
	exits non-zero, is killed or sends a pickle that does not load, its share
	is computed again here with this process's own budget. On an exception
	in this process, every child still running is killed and reaped.
	"""

	children, results = [], [None] * len(shares)

	try:
		for k in range(1, len(shares)):
			read_fd, write_fd = os.pipe()
			try:
				with warnings.catch_warnings():
					# Python >= 3.12 warns when a multi-threaded process
					# forks. The child calls numba only when
					# numba_fork_safe() allows it, and leaves through
					# os._exit.
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
					if budgets is not None:
						_set_budget(budgets[k])

					result = func(shares[k])
					if pack is not None:
						result = pack(result)

					with os.fdopen(write_fd, 'wb') as handle:
						pickle.dump(result, handle,
							protocol=pickle.HIGHEST_PROTOCOL)
					code = 0
				finally:
					os._exit(code)

			os.close(write_fd)
			children.append((pid, os.fdopen(read_fd, 'rb'), k))

		if budgets is None:
			results[0] = func(shares[0])
		else:
			saved = _budget, numba.get_num_threads()
			try:
				_set_budget(budgets[0])
				results[0] = func(shares[0])
			finally:
				_set_budget(*saved)

		while children:
			pid, handle, k = children[0]
			result, received = None, False
			if pid is not None:
				data = handle.read()
				handle.close()
				_, status = os.waitpid(pid, 0)
				children[0] = (None, None, k)

				if os.waitstatus_to_exitcode(status) == 0:
					try:
						result = pickle.loads(data)
						received = True
					except Exception:
						pass

			if received:
				if unpack is not None:
					result = unpack(result)
			else:
				result = func(shares[k])

			results[k] = result
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


class Background:
	"""func() computed in a forked child beside this process's own work,
	for work this process would otherwise do after it.

	start(n_threads) forks the child, which sets its budget and numba's
	thread count to n_threads before it calls func, when n_threads is at
	least 1 and a child may run numba's parallel kernels (numba_fork_safe);
	otherwise it does nothing. join() waits for the child to finish and
	reaps it. result() returns unpack(pack(func())) as the child computed
	it (each defaults to the identity), or func() computed here with this
	process's own budget when no child was started, or the child could not
	be forked, exited non-zero, was killed or sent a pickle that does not
	load. kill() kills and reaps a child that is still running, so that
	result() computes func() here.
	"""

	def __init__(self, func, pack=None, unpack=None):
		self.func = func
		self.pack = pack
		self.unpack = unpack
		self._pid = None
		self._handle = None
		self._started = False
		self._data = None

	def start(self, n_threads):
		if self._started or n_threads < 1 or not numba_fork_safe():
			return
		self._started = True

		read_fd, write_fd = os.pipe()
		try:
			with warnings.catch_warnings():
				# As in forked_map.
				warnings.simplefilter('ignore', DeprecationWarning)
				pid = os.fork()
		except OSError:
			os.close(read_fd)
			os.close(write_fd)
			return

		if pid == 0:
			code = 1
			try:
				os.close(read_fd)
				_set_budget(n_threads)

				result = self.func()
				if self.pack is not None:
					result = self.pack(result)

				with os.fdopen(write_fd, 'wb') as handle:
					pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
				code = 0
			finally:
				os._exit(code)

		os.close(write_fd)
		self._pid, self._handle = pid, os.fdopen(read_fd, 'rb')

	def join(self):
		if self._pid is None:
			return

		data = self._handle.read()
		self._handle.close()
		_, status = os.waitpid(self._pid, 0)
		self._pid = self._handle = None

		if os.waitstatus_to_exitcode(status) == 0:
			self._data = data

	def kill(self):
		if self._pid is None:
			return

		self._handle.close()
		try:
			os.kill(self._pid, signal.SIGKILL)
			os.waitpid(self._pid, 0)
		except OSError:
			pass
		self._pid = self._handle = None

	def result(self):
		self.join()

		data, self._data = self._data, None
		if data is not None:
			try:
				result = pickle.loads(data)
			except Exception:
				pass
			else:
				return result if self.unpack is None else self.unpack(result)

		return self.func()


def _lpt(weights, order, n_bins):
	"""Longest processing time first: the items of `order`, heaviest first,
	each into the bin with the least weight so far (the lowest index on a
	tie). Returns the bins and their weights."""

	bins, loads = [[] for _ in range(n_bins)], [0] * n_bins
	for j in order:
		b = loads.index(min(loads))
		bins[b].append(j)
		loads[b] += weights[j]

	return bins, loads


def _plan(weights, n_budget):
	"""Group tasks into shares, one per process, heaviest task first.

	The heaviest task goes to bin 0, which the parent runs. The number of
	bins is the smallest that keeps every other bin at or below half the
	heaviest task's weight, so that the children finish first even though
	each gets only part of the thread budget and small tasks cost more than
	their weight says. There is at most one bin per task, at most one child
	for every MIN_CHILD_WEIGHT of weight outside the heaviest task (so a
	single bin when that weight is below it), and at most one child for
	every two threads of the budget, which leaves the parent at least half
	the budget's worth of what `_split_budget` hands out beyond one thread
	per bin: the heaviest task slows most when it has fewer threads, and
	its Leiden seeds are forked only when it has two or more.
	"""

	n = len(weights)
	order = sorted(range(n), key=lambda j: -weights[j])
	if n == 0:
		return [order], [0]

	heaviest = weights[order[0]]
	rest = sum(weights) - heaviest
	max_bins = min(n, 1 + n_budget // 2, 1 + int(rest // MIN_CHILD_WEIGHT))
	if max_bins < 2:
		return [order], [sum(weights)]

	for n_bins in range(2, max_bins + 1):
		bins, loads = _lpt(weights, order, n_bins)
		if max(loads[1:]) <= heaviest / 2:
			break

	return bins, loads


def _plan_even(weights, n_budget, min_child_weight):
	"""Group many small tasks into shares of about equal weight, one per
	process: as many processes as the budget has threads, but no more than
	there are tasks and no more than one for every `min_child_weight` of
	total weight (so a single share when the total is below twice that).
	The tasks are packed longest first (`_lpt`), and bin 0, which the
	parent runs, takes the heaviest."""

	n = len(weights)
	order = sorted(range(n), key=lambda j: -weights[j])
	n_bins = min(n, n_budget, int(sum(weights) // min_child_weight))
	if n_bins < 2:
		return [order], [sum(weights)]

	return _lpt(weights, order, n_bins)


def _split_budget(loads, n_budget):
	"""One thread for every bin, and the rest in proportion to the bins'
	weights, rounded down, with what the rounding leaves over going to bin 0.
	The shares sum to n_budget."""

	extra = n_budget - len(loads)
	total = sum(loads)
	budgets = [1 + int(extra * load / total) for load in loads]
	budgets[0] += n_budget - sum(budgets)
	return budgets


def map_tasks(func, tasks, weights, pack=None, unpack=None, before_fork=None,
		min_child_weight=None):
	"""Return {task: func(task)} for every task, spreading the tasks over
	forked children when that is worth it and safe.

	The tasks must be independent: func(task) may change only objects that
	belong to its own task, and its result must not depend on which process
	computes it or in what order. A child sends pack(task, func(task)) for
	each of its tasks and this process applies unpack to what it receives,
	keyed by task: unpack({task: packed}) returns {task: result} and may
	attach state to this process's objects, standing in for what func(task)
	changed in the child. When forking, before_fork() runs first, in
	this process (for example, to compile a numba kernel once rather than in
	every child).

	Tasks are grouped by `_plan` from their weights, or, when
	`min_child_weight` is given, by `_plan_even` (many small tasks spread
	evenly, each process getting at least that much weight), and the
	threads by `_split_budget`. When there is one group, or when a child
	could not run numba's parallel kernels safely (numba_fork_safe), every
	task runs here, in the order given.
	"""

	tasks = list(tasks)
	n_budget = budget()
	if min_child_weight is None:
		bins, loads = _plan(weights, n_budget)
	else:
		bins, loads = _plan_even(weights, n_budget, min_child_weight)

	if len(bins) < 2 or not numba_fork_safe():
		return {task: func(task) for task in tasks}

	if before_fork is not None:
		before_fork()

	budgets = _split_budget(loads, n_budget)
	shares = [[tasks[j] for j in b] for b in bins]

	def run(share):
		return {task: func(task) for task in share}

	def pack_share(share_results):
		return {task: pack(task, result) for task, result in share_results.items()}

	results = {}
	for share_results in forked_map(run, shares, budgets=budgets,
			pack=None if pack is None else pack_share, unpack=unpack):
		results.update(share_results)

	return results
