# precompile.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

"""Compile the package's numba kernels in forked child processes, into
numba's on-disk cache, while this process gets on with the run.

A fresh process compiles every kernel on its first call, one after another,
and waits for each. `start()`, called once at the start of a run, forks
children that compile the kernels the run will need, with the argument types
it will pass, and write them to the cache. When this process reaches a kernel
for the first time, it waits for the child compiling it and then loads the
compiled code from the cache, which takes a few milliseconds.

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
import signal
import warnings
import threading

import numba
from numba.core import types
from numba.core.dispatcher import Dispatcher
from numba.core.compiler_lock import global_compiler_lock

from . import forked


# A child that has not finished this many seconds after it was forked is
# killed, and the kernel is compiled here instead.
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
	"""The kernels a run of `TFMoDISco` calls, in the order it first calls
	them, each with the signature its first call compiles: the one it was
	declared with, or the argument types the pipeline passes it."""

	from . import affinitymat, gapped_kmer, util

	A = types.Array
	i8 = types.int64

	return [
		(gapped_kmer._gkmer_rows, (A(types.float64, 3, 'C'), i8, i8, i8, i8,
			i8, i8, A(types.int64, 2, 'C'), A(types.float64, 2, 'C'))),
		(affinitymat._sparse_mm_dot, (A(types.float64, 1, 'C'),
			A(types.int64, 1, 'C'), A(types.int64, 1, 'C'),
			A(types.int64, 1, 'C'), A(types.int32, 1, 'C'),
			A(types.float64, 1, 'C'), A(types.int64, 1, 'C'),
			A(types.int32, 1, 'C'), A(types.float64, 1, 'C'), i8)),
		(affinitymat._jaccard_best, None),
		(util.binary_search_perplexity, None),
		(affinitymat.pairwise_jaccard, (A(types.float32, 2, 'C'), i8)),
	]


class _Child:
	def __init__(self, pid):
		self.pid = pid
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
			_wait(child)

		sig = _signatures.get(dispatcher)
		if sig is not None and dispatcher._can_compile:
			dispatcher.compile(sig)
			dispatcher.disable_compile()

	return sig


def _wait(child):
	"""Wait until the child has exited (killing it once TIMEOUT has passed
	since it was forked), then reap any other child that has exited and
	give their threads back."""

	if not child.finished:
		if os.getpid() != _owner:
			# A forked worker inherited this record; the child is not its own.
			child.finished = True
		else:
			while True:
				try:
					pid, _ = os.waitpid(child.pid, os.WNOHANG)
				except ChildProcessError:
					pid = child.pid

				if pid != 0:
					break

				if time.monotonic() - child.forked > TIMEOUT:
					_kill(child)
					break

				time.sleep(0.001)

			child.finished = True

	_reap()


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
			child.finished = pid != 0

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

	child.finished = True


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
	for dispatcher, sig in _kernels():
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

		todo.append((dispatcher, sig))

	n_children = min(len(todo), n_budget - 1)
	if n_children < 1:
		return

	# Kernels are dealt out in the order they are first called, so each
	# child compiles the kernel needed soonest first.
	shares = [todo[k::n_children] for k in range(n_children)]

	_owner = os.getpid()
	atexit.register(_stop)

	for share in shares:
		try:
			with warnings.catch_warnings():
				# Python >= 3.12 warns when a process with native threads
				# (a fork-safe threading layer, _can_fork) forks.
				warnings.simplefilter('ignore', DeprecationWarning)
				pid = os.fork()
		except OSError:
			continue

		if pid == 0:
			code = 1
			try:
				for dispatcher, sig in share:
					dispatcher.compile(sig)
				code = 0
			finally:
				os._exit(code)

		child = _Child(pid)
		_children.append(child)
		for dispatcher, sig in share:
			_jobs[dispatcher] = child
			_hook(dispatcher)

	if _children:
		n = numba.get_num_threads()
		_threads = n, n
		_reap()
