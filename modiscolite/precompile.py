# precompile.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""Compile the package's numba kernels in forked child processes, into
numba's on-disk cache, while this process gets on with the run.

A fresh process compiles every kernel on its first call, one after another,
and waits for each. `start()`, called once at the start of a run, forks
children that compile the kernels the run will need, with the argument types
it will pass, and write them to the cache. A child reports each kernel over a
pipe as soon as it is in the cache. When this process reaches a kernel for
the first time, it waits for that report, not for the child's other kernels,
and then loads the compiled code from the cache, which takes a few
milliseconds.

Nothing depends on a child succeeding. The first call compiles the kernel in
the ordinary way, and a child's work only turns that compile into a cache
hit. If a child failed, was killed, or compiled other argument types than the
call passes, the call compiles the kernel in this process, as it would have
without `start()`. When it is not safe to fork (not Linux, other threads
running, a threading layer that is not fork safe) or numba has nowhere to
write its cache (a read-only install with no writable NUMBA_CACHE_DIR or
home directory), `start()` does nothing.

Caching is turned on for the listed kernels only when `start()` forks, so
later runs load them from the cache. The cache is keyed on the kernel's own
source file: an edit to that file invalidates it, but an edit to an @njit
function it calls from another file would not. Each listed kernel calls
only functions in its own file and numba's own.

The children count against the thread budget (`forked.budget()`): each
compiles on one thread, and while they run this process's numba thread count
is lowered by the number still running.
"""

import os
import sys
import time
import atexit
import select
import signal
import warnings
import threading

import numba
from numba.core import types
from numba.core.dispatcher import Dispatcher
from numba.core.compiler_lock import global_compiler_lock

from . import forked


# A child that has not reported a kernel this process waits for, this many
# seconds after it was forked, is killed, and the kernel is compiled here
# instead.
TIMEOUT = 120.0

# Kernels compiled on first call for one signature only (`signature`), and
# the child compiling each kernel.
_signatures = {}
_jobs = {}

_started = False
_owner = None
_children = []
_threads = None


def signature(sig):
	"""Decorate an njit dispatcher so that it is compiled for `sig` on its
	first call, as njit(sig) would compile it at import, and for no other
	signature: calls whose types differ are converted to it, or raise, as
	with njit(sig)."""

	def decorate(dispatcher):
		if isinstance(dispatcher, Dispatcher):
			_signatures[dispatcher] = sig
			_hook(dispatcher)
		return dispatcher

	return decorate


def _kernels():
	"""The kernels a run of `TFMoDISco` calls, each with the signature its
	first call compiles: the one it was declared with, or the argument types
	the pipeline passes it.

	Kernels are grouped: one child compiles the kernels of a group, one
	after another, in the order listed, and reports each as it lands, so a
	first call never waits for a kernel queued behind its own. Groups are
	listed in the order the run first calls their first kernel. Every child
	holds about 100 MB while it compiles, so a kernel first called long
	after the others shares a child rather than add one to the start of the
	run."""

	from . import affinitymat, aggregator, gapped_kmer, util

	A = types.Array
	i8 = types.int64

	return [
		# _jaccard_all is first called in aggregator._detect_spurious_merging,
		# just after pairwise_jaccard's first call. Queued behind _gkmer_rows,
		# it compiles alongside pairwise_jaccard rather than after it, which
		# a small input, reaching both calls early, would wait for.
		[(gapped_kmer._gkmer_rows, (A(types.float64, 3, 'C'), i8, i8, i8, i8,
			i8, i8, A(types.int64, 2, 'C'), A(types.float64, 2, 'C'))),
			(affinitymat._jaccard_all, None)],
		[(affinitymat._sparse_mm_dot, (A(types.float64, 1, 'C'),
			A(types.int64, 1, 'C'), A(types.int64, 1, 'C'),
			A(types.int64, 1, 'C'), A(types.int32, 1, 'C'),
			A(types.float64, 1, 'C'), A(types.int64, 1, 'C'),
			A(types.int32, 1, 'C'), A(types.float64, 1, 'C'), i8))],
		[(affinitymat._fwd_rev_best, None)],
		# tfmodisco._density_adaptation's kernels, in the order it calls them.
		# binary_search_perplexity is compiled into _perplexity_betas.
		# _jaccard_best is first called in _patterns_from_clusters, after the
		# first main clustering, so it is queued behind them.
		[(util._transpose_triples, None), (util._transpose, None),
			(util._symmetrize, None), (util._perplexity_betas, None),
			(util._row_sums_plus_one, None), (util._add_diagonal, None),
			(affinitymat._jaccard_best, None)],
		# First called in aggregator._detect_spurious_merging, after both
		# main clusterings; _roc_corners in its SimilarPatternsCollapser.
		[(affinitymat._signs_magnitudes, (A(types.float32, 2, 'C'),
			A(types.float32, 2, 'C'), A(types.float32, 2, 'C'))),
			(affinitymat._pairwise_jaccard_rows, (A(types.float32, 2, 'C'),
			A(types.float32, 2, 'C'), A(types.float32, 2, 'C'), i8, i8, i8,
			A(types.float64, 2, 'C'), A(types.int32, 2, 'C'))),
			(aggregator._roc_corners, None)],
	]


class _Child:
	def __init__(self, pid, fd, kernels):
		self.pid = pid
		self.fd = fd
		self.kernels = kernels
		self.landed = 0
		self.forked = time.monotonic()
		self.finished = False


def _hook(dispatcher):
	"""Route the dispatcher's first compile through `_first_compile`. The
	dispatcher looks `_compile_for_args` up on itself whenever a call finds
	no compiled version, so an instance attribute takes its place."""

	if '_compile_for_args' in vars(dispatcher):
		return

	compile_for_args = dispatcher._compile_for_args

	def _compile_for_args(*args, **kws):
		sig = _first_compile(dispatcher)
		if sig is None:
			return compile_for_args(*args, **kws)

		# Dispatch again, now that the one signature is compiled and no
		# other may be; the dispatcher calls what this returns.
		return dispatcher

	dispatcher._compile_for_args = _compile_for_args


def _first_compile(dispatcher):
	"""Wait for the child compiling this dispatcher, if any. For a
	dispatcher with a fixed signature, compile it (from the cache, when a
	child put it there) and allow no other. Returns that signature."""

	with global_compiler_lock:
		child = _jobs.get(dispatcher)
		if child is not None:
			_wait(child, dispatcher)

		sig = _signatures.get(dispatcher)
		if sig is not None and dispatcher._can_compile:
			dispatcher.compile(sig)
			dispatcher.disable_compile()

	return sig


def _wait(child, dispatcher):
	"""Wait until the child has reported the dispatcher's kernel in the
	cache, or has exited (killing it once TIMEOUT has passed since it was
	forked), then reap any child that has exited and give their threads
	back. For the last kernel of its share, wait for the child to exit,
	which follows at once, so that a child is reaped by the first call of
	its last kernel."""

	if not child.finished:
		if os.getpid() != _owner:
			# A forked worker inherited this record; the child is not its own,
			# and its reports are this process's parent's to read.
			child.finished = True
		else:
			n = child.kernels.index(dispatcher) + 1
			if n == len(child.kernels):
				n = n + 1

			poll = select.poll()
			poll.register(child.fd, select.POLLIN)
			while not child.finished and child.landed < n:
				timeout = child.forked + TIMEOUT - time.monotonic()
				if timeout <= 0:
					_kill(child)
				elif poll.poll(1000 * timeout + 1):
					_read(child)

	_reap()


def _read(child):
	"""Count the kernels the child has reported since the last read. At the
	end of the pipe the child is exiting (its end closes as it exits, just
	before it can be reaped): reap it."""

	data = os.read(child.fd, 64)
	child.landed += len(data)
	if not data:
		try:
			os.waitpid(child.pid, 0)
		except ChildProcessError:
			pass

		_finish(child)


def _finish(child):
	child.finished = True
	if child.fd is not None:
		os.close(child.fd)
		child.fd = None


def _reap():
	"""Mark every child that has exited as finished, and set this process's
	numba thread count to its budget less the children still running."""

	global _threads

	if os.getpid() != _owner:
		return

	for child in _children:
		if not child.finished:
			try:
				pid, _ = os.waitpid(child.pid, os.WNOHANG)
			except ChildProcessError:
				pid = child.pid
			if pid != 0:
				_finish(child)

	# The thread count is per thread and belongs to the main thread; leave it
	# alone if anything else has changed it since.
	if (_threads is not None and threading.current_thread() is
			threading.main_thread() and numba.get_num_threads() == _threads[1]):
		running = sum(not child.finished for child in _children)
		n = max(1, _threads[0] - running)
		if n != _threads[1]:
			numba.set_num_threads(n)
			_threads = _threads[0], n


def _kill(child):
	try:
		os.kill(child.pid, signal.SIGKILL)
		os.waitpid(child.pid, 0)
	except OSError:
		pass

	_finish(child)


def _stop():
	"""Kill and reap the children still running, at exit."""

	if os.getpid() != _owner:
		return

	for child in _children:
		if not child.finished:
			_kill(child)


def _can_fork():
	"""Whether a child forked now can compile safely: Linux, no thread but
	this main one, and numba's threading layer either not started yet or
	one numba counts as fork safe (forked.numba_fork_safe)."""

	if not sys.platform.startswith('linux') or numba.config.DISABLE_JIT:
		return False

	if (threading.current_thread() is not threading.main_thread() or
			threading.active_count() > 1):
		return False

	try:
		numba.threading_layer()
	except ValueError:
		return True

	return forked.numba_fork_safe()


def start():
	"""Fork children that compile, into numba's cache, the kernels this
	process has not compiled yet, and return at once. Does nothing after its
	first call, when forking is not safe, or when the budget has no thread
	to spare."""

	global _started, _owner, _threads

	if _started:
		return
	_started = True

	if not _can_fork():
		return

	n_budget = forked.budget()
	todo = []
	for group in _kernels():
		jobs = []
		for dispatcher, sig in group:
			if not isinstance(dispatcher, Dispatcher):
				continue

			if sig is None:
				sig = _signatures.get(dispatcher)
			if sig is None or len(dispatcher.overloads) > 0:
				continue

			try:
				dispatcher.enable_caching()
			except Exception:
				# No writable cache location: compile in this process.
				continue

			jobs.append((dispatcher, sig))

		if jobs:
			todo.append(jobs)

	n_children = min(len(todo), n_budget - 1)
	if n_children < 1:
		return

	# Groups are dealt out in the order they are first called, so each
	# child compiles the kernel needed soonest first.
	shares = [[job for jobs in todo[k::n_children] for job in jobs]
		for k in range(n_children)]

	_owner = os.getpid()
	atexit.register(_stop)

	for share in shares:
		try:
			read_fd, write_fd = os.pipe()
		except OSError:
			break

		try:
			with warnings.catch_warnings():
				# Python >= 3.12 warns when a process with native threads
				# (a fork-safe threading layer, _can_fork) forks.
				warnings.simplefilter('ignore', DeprecationWarning)
				pid = os.fork()
		except OSError:
			os.close(read_fd)
			os.close(write_fd)
			continue

		if pid == 0:
			code = 1
			try:
				os.close(read_fd)
				for dispatcher, sig in share:
					# compile() has written the cache files when it returns,
					# each through a temporary file and a rename, so a
					# reported kernel is complete on disk.
					dispatcher.compile(sig)
					os.write(write_fd, b'.')
				code = 0
			finally:
				os._exit(code)

		os.close(write_fd)
		child = _Child(pid, read_fd, [dispatcher for dispatcher, sig in share])
		_children.append(child)
		for dispatcher, sig in share:
			_jobs[dispatcher] = child
			_hook(dispatcher)

	if _children:
		n = numba.get_num_threads()
		_threads = n, n
		_reap()
