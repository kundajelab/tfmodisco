# aggregator.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import numpy as np

from numba import njit

from . import affinitymat
from . import core
from . import forked
from . import precompile
from . import util

from collections import OrderedDict


@precompile.signature('float64[:, ::1](float64[::1], float64[::1])')
@njit
def _roc_corners(neg_sorted, pos_sorted):
	"""The false and true positive counts (rows 0 and 1) of the ROC points
	that sklearn's `roc_curve` keeps, for scores sorted ascending, finite,
	and at least one of each class.

	sklearn takes one point per distinct score, from the highest down: the
	number of negatives and of positives scoring at least that much. Tied
	scores (-0.0 == 0.0) make one point. With more than two points, it
	keeps the first, the last, and each point where the step to the next
	(the numbers of negatives and positives at the next score) differs from
	the step into it: `np.diff(fps, 2)` or `np.diff(tps, 2)` is nonzero.
	Here the two sorted classes are merged from the top, one distinct score
	at a time, and the points are decided as the steps arrive. The counts
	are integers, so these are the float64 values sklearn computes. The
	first pass counts the points kept and the second writes them, so the
	output is only as large as what is kept."""

	n_neg, n_pos = neg_sorted.shape[0], pos_sorted.shape[0]
	out = np.empty((2, 0), dtype=np.float64)

	for write in range(2):
		k = 0
		a, b = n_neg - 1, n_pos - 1
		fp, tp = 0, 0
		n_points = 0
		# The last point seen, not yet decided, and the step into it.
		last_fp, last_tp, last_cn, last_cp = 0, 0, 0, 0

		while a >= 0 or b >= 0:
			if b < 0 or (a >= 0 and neg_sorted[a] >= pos_sorted[b]):
				score = neg_sorted[a]
			else:
				score = pos_sorted[b]

			cn = 0
			while a >= 0 and neg_sorted[a] == score:
				cn += 1
				a -= 1

			cp = 0
			while b >= 0 and pos_sorted[b] == score:
				cp += 1
				b -= 1

			fp += cn
			tp += cp

			# Keep the last point if it is the first, or if its step in
			# differs from its step out.
			if n_points > 0 and (n_points == 1 or cn != last_cn
					or cp != last_cp):
				if write:
					out[0, k] = last_fp
					out[1, k] = last_tp
				k += 1

			last_fp, last_tp, last_cn, last_cp = fp, tp, cn, cp
			n_points += 1

		# The last point is always kept.
		if write:
			out[0, k] = last_fp
			out[1, k] = last_tp
		else:
			out = np.empty((2, k + 1), dtype=np.float64)

	return out


def _roc_auc_score(neg_scores, pos_scores):
	"""The AUROC of telling `pos_scores` (label 1) from `neg_scores` (label 0).

	Returns the same float64 bits as sklearn 1.6.1's
	`roc_auc_score(y_true=[0]*len(neg_scores) + [1]*len(pos_scores),
	y_score=list(neg_scores) + list(pos_scores))`, without its input
	validation and without building Python lists: each class is sorted with
	numpy's quicksort and `_roc_auc_sorted` does the rest.
	"""

	neg_scores = np.asarray(neg_scores, dtype=np.float64).ravel()
	pos_scores = np.asarray(pos_scores, dtype=np.float64).ravel()
	return _roc_auc_sorted(np.sort(neg_scores), np.sort(pos_scores))


def _roc_auc_sorted(neg_sorted, pos_sorted):
	"""`_roc_auc_score(neg_sorted, pos_sorted)` for 1-D float64 arrays that
	are already sorted ascending, which it does not copy.

	The arithmetic is sklearn's, step for step: the false and true positive
	counts at each distinct score (`_binary_clf_curve`), the collinear
	intermediate points dropped (`roc_curve`), which `_roc_corners` finds in
	one merge of the two sorted classes, then (0, 0) prepended, both counts
	normalised by their last value, and scipy's trapezoid expression
	(`auc`) over the same arrays. sklearn orders the scores with a stable
	argsort; the counts at each distinct score do not depend on the order
	of tied scores, so any correct sort gives the same points.
	"""

	neg_sorted = np.ascontiguousarray(neg_sorted, dtype=np.float64)
	pos_sorted = np.ascontiguousarray(pos_sorted, dtype=np.float64)
	n_neg, n_pos = neg_sorted.shape[0], pos_sorted.shape[0]

	def _sklearn():
		from sklearn.metrics import roc_auc_score
		return roc_auc_score(
			y_true=np.r_[np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)],
			y_score=np.concatenate([neg_sorted, pos_sorted]))

	# One class only: sklearn warns and returns nan.
	if n_neg == 0 or n_pos == 0:
		return _sklearn()

	# NaN sorts last and -inf first; sklearn raises on either, so let it.
	if not (np.isfinite(neg_sorted[0]) and np.isfinite(neg_sorted[-1])
			and np.isfinite(pos_sorted[0]) and np.isfinite(pos_sorted[-1])):
		return _sklearn()

	fps, tps = _roc_corners(neg_sorted, pos_sorted)

	tps = np.r_[0, tps]
	fps = np.r_[0, fps]
	fpr = fps / fps[-1]
	tpr = tps / tps[-1]

	# sklearn.metrics.auc -> scipy.integrate.trapezoid. fpr never decreases,
	# so sklearn's direction is 1.
	d = fpr[1:] - fpr[:-1]
	return np.sum(d * (tpr[1:] + tpr[:-1]) / 2.0, axis=-1, dtype=np.float64)


def polish_pattern(pattern, min_frac, min_num, track_set, flank, window_size, bg_freq):
	polished = _polish_coordinates(pattern, min_frac, min_num, track_set,
		flank, window_size, bg_freq)
	if polished is not _NOT_APPLICABLE:
		return polished

	pattern = pattern.trim_to_support(min_frac=min_frac, min_num=min_num)

	pattern = _expand_seqlets_to_fill_pattern(pattern, 
		track_set=track_set, left_flank_to_add=flank,
		right_flank_to_add=flank)

	if pattern is None:
		return None

	# Trim by IC
	ppm = pattern.sequence
	per_pos_ic = util.compute_per_position_ic(
		ppm=ppm, background=bg_freq, pseudocount=0.001)

	start_idx = np.argmax(util.cpu_sliding_window_sum(
		arr=per_pos_ic, window_size=window_size))

	end_idx = start_idx + window_size
	pattern = pattern.trim_to_idx(start_idx=start_idx, end_idx=end_idx)

	pattern = _expand_seqlets_to_fill_pattern(pattern, 
		track_set=track_set, left_flank_to_add=flank,
		right_flank_to_add=flank)

	return pattern

def _expand_seqlets_to_fill_pattern(pattern, track_set, left_flank_to_add,
	right_flank_to_add):

	new_seqlets = []
	for seqlet in pattern.seqlets:
		left_expansion = left_flank_to_add 
		right_expansion = len(pattern) - len(seqlet) + right_flank_to_add

		if seqlet.is_revcomp == False:
			start = seqlet.start - left_expansion
			end = seqlet.end + right_expansion
		else:
			start = seqlet.start - right_expansion
			end = seqlet.end + left_expansion
		
		if start >= 0 and end <= track_set.length:
			new_seqlets.append(core.Seqlet(example_idx=seqlet.example_idx,
				start=start, end=end, is_revcomp=seqlet.is_revcomp))

	# One create_seqlets call for all of them: each seqlet gets the views it
	# would get from a call of its own.
	new_seqlets = track_set.create_seqlets(seqlets=new_seqlets)

	if len(new_seqlets) > 0:
		return core.SeqletSet(seqlets=new_seqlets, track_set=track_set)
	else:
		return None


# Pattern expansions and trims without the seqlets.
#
# Merging a seqlet into a pattern can expand the whole pattern, which makes
# every one of its seqlets again (_expand_seqlets_to_fill_pattern), and
# polishing trims, expands, trims and expands again; of all the seqlets
# these steps make, only the last step's are kept. Here those steps are
# taken on the seqlets' coordinates: a "shell" is a SeqletSet (made by
# SeqletSet._assemble) whose `seqlets` hold (example_idx, start, end,
# is_revcomp) tuples where the steps above would hold the Seqlet objects
# made at those coordinates, and whose unique_seqlets maps each one's string
# to its tuple. Its length, counts and running sums are the ones the
# SeqletSet would have, so its averages are too. A shell never leaves this
# module: _materialize makes it into the SeqletSet itself, with the Seqlet
# objects, made by create_seqlets as the steps above make them.
#
# The coordinates are computed by the same expressions on the same values as
# the steps they stand for (so they are the same objects of the same types),
# and kept as SeqletSet keeps seqlets: the first of each string, in order.
# Every seqlet of a shell spans it: an expansion makes every seqlet as long
# as the expanded pattern, and a trim as long as its window.

_NOT_APPLICABLE = object()


def _expanded(entries, length, track_length, left, right):
	"""The seqlets that _expand_seqlets_to_fill_pattern makes from a pattern
	`length` long holding `entries` (Seqlet objects, or the tuples of a
	shell, which span it), as SeqletSet keeps them: (kept, unique, all_kept),
	the coordinate tuples kept, the dict from each one's string to it, and
	whether every entry was kept (none fell off the track or repeated a
	string)."""

	kept, unique = [], {}
	for entry in entries:
		if type(entry) is tuple:
			example_idx, start, end, is_revcomp = entry
			n = length
		else:
			example_idx, start, end = entry.example_idx, entry.start, entry.end
			is_revcomp = entry.is_revcomp
			n = len(entry)

		right_expansion = length - n + right
		if is_revcomp == False:
			start = start - left
			end = end + right_expansion
		else:
			start = start - right_expansion
			end = end + left

		if start >= 0 and end <= track_length:
			key = f"{example_idx!s}_{start!s}_{end!s}"
			if key not in unique:
				coordinates = (example_idx, start, end, is_revcomp)
				unique[key] = coordinates
				kept.append(coordinates)

	return kept, unique, len(kept) == len(entries)


def _trimmed(entries, start_idx, end_idx):
	"""The seqlets that SeqletSet.trim_to_idx(start_idx, end_idx) keeps from
	`entries` (Seqlet objects or a shell's tuples), as _expanded returns
	them. Seqlet.trim's coordinates; trim_to_idx keeps the first of each
	string in both of its paths."""

	kept, unique = [], {}
	for entry in entries:
		if type(entry) is tuple:
			example_idx, start, end, is_revcomp = entry
		else:
			example_idx, start, end = entry.example_idx, entry.start, entry.end
			is_revcomp = entry.is_revcomp

		if is_revcomp == False:
			new_start = start + start_idx
			new_end = start + end_idx
		else:
			new_start = end - end_idx
			new_end = end - start_idx

		key = f"{example_idx!s}_{new_start!s}_{new_end!s}"
		if key not in unique:
			coordinates = (example_idx, new_start, new_end, is_revcomp)
			unique[key] = coordinates
			kept.append(coordinates)

	return kept, unique, len(kept) == len(entries)


def _flank_sums(track_set, kept, length, interior, left, right):
	"""The running sums of the seqlets at the coordinates `kept`, each
	`length` long, whose values at positions left .. length - right are
	those of seqlets whose running sums are `interior` (in the same order),
	or None when the kernel does not apply.

	SeqletSet adds each position's values one seqlet after another, from
	0.0, and the kernel (TrackSet._coordinate_sums) does the same; the
	additions at one position do not depend on any other position. So the
	interior positions take the same additions as `interior` did, and only
	the flanks are added up here: the left flank as seqlets `left` long and
	the right flank as seqlets `right` long, at the starts that give the
	same track positions (the reverse strand reads its seqlet backwards, so
	its left flank is the end of its window)."""

	example_idx = np.array([coordinates[0] for coordinates in kept],
		dtype=np.int64)
	starts = np.array([coordinates[1] for coordinates in kept], dtype=np.int64)
	revcomp = np.array([bool(coordinates[3]) for coordinates in kept],
		dtype=np.uint8)
	strand = revcomp.astype(bool)

	sums = tuple(np.zeros((length, 4), dtype='float') for _ in range(3))
	middle = length - right
	for total, inner in zip(sums, interior):
		total[left:middle] = inner

	for n, flank_starts, rows in (
			(left, np.where(strand, starts + (length - left), starts),
				slice(0, left)),
			(right, np.where(strand, starts, starts + middle),
				slice(middle, length))):
		if n == 0:
			continue

		flank = tuple(np.zeros((n, 4), dtype='float') for _ in range(3))
		if not track_set._coordinate_sums(example_idx, flank_starts, revcomp,
				n, flank):
			return None

		for total, part in zip(sums, flank):
			total[rows] = part

	return sums


def _coordinate_set(track_set, kept, unique, length, reuse=None):
	"""The shell of SeqletSet(seqlets, track_set=track_set) for seqlets of
	this length at the coordinates `kept` (as _expanded returns them), or
	_NOT_APPLICABLE when the kernel does not apply, which SeqletSet would
	then not use either. reuse: (interior sums, left, right) from the
	pattern these seqlets were expanded from (_flank_sums), when every one
	of its seqlets was kept, in order, and its sums are the kernel's for
	them."""

	if reuse is not None:
		interior, left, right = reuse
		if any(np.isnan(inner).any() for inner in interior):
			reuse = None

	sums = None
	if reuse is not None:
		sums = _flank_sums(track_set, kept, length, interior, left, right)

	if sums is None:
		sums = tuple(np.zeros((length, 4), dtype='float') for _ in range(3))
		if not track_set._coordinate_sums([coordinates[0] for coordinates in
				kept], [coordinates[1] for coordinates in kept],
				[coordinates[3] for coordinates in kept], length, sums):
			return _NOT_APPLICABLE

	per_position_counts = np.zeros((length,))
	per_position_counts += len(kept)
	return core.SeqletSet._assemble(kept, unique, length, per_position_counts,
		sums)


def _materialize(shell, track_set):
	"""The SeqletSet a shell stands for: each coordinate tuple made into a
	Seqlet with the views create_seqlets gives it, the Seqlet objects
	already there kept, and the same counts and sums."""

	made = iter(track_set.create_seqlets([core.Seqlet(example_idx=entry[0],
		start=entry[1], end=entry[2], is_revcomp=entry[3])
		for entry in shell.seqlets if type(entry) is tuple]))
	seqlets = [next(made) if type(entry) is tuple else entry
		for entry in shell.seqlets]

	unique_seqlets = dict(zip(shell.unique_seqlets, seqlets))
	assert len(unique_seqlets) == len(seqlets) == len(shell.unique_seqlets)
	return core.SeqletSet._assemble(seqlets, unique_seqlets, shell.length,
		shell.per_position_counts, shell._sums())


def _expand_shell(pattern, track_set, left_flank_to_add, right_flank_to_add,
	reusable):
	"""_expand_seqlets_to_fill_pattern(pattern, ...) as a shell: None where
	it returns None, and _NOT_APPLICABLE where its SeqletSet would not add
	its sums up with the kernel. `pattern` is a SeqletSet or a shell;
	`reusable` says that its running sums are the kernel's for its seqlets
	(each spanning it) at their coordinates, so that an expansion that
	keeps all of them adds up only the new flanks."""

	length = len(pattern)
	kept, unique, all_kept = _expanded(pattern.seqlets, length,
		track_set.length, left_flank_to_add, right_flank_to_add)
	if not kept:
		return None

	# Every seqlet made is length + left + right long; SeqletSet's length is
	# the longest one's len().
	new_length = len(core.Seqlet(*kept[0]))
	reuse = None
	if reusable and all_kept and new_length == (length + left_flank_to_add +
			right_flank_to_add):
		reuse = (pattern._sums(), left_flank_to_add, right_flank_to_add)

	return _coordinate_set(track_set, kept, unique, new_length, reuse)


def _polish_coordinates(pattern, min_frac, min_num, track_set, flank,
	window_size, bg_freq):
	"""polish_pattern's result, with its two trims and its first expansion
	taken on coordinates (shells), or _NOT_APPLICABLE when any step would
	not take the path these follow, for polish_pattern to run as it is."""

	start_idx, end_idx = pattern._support_window(min_frac, min_num)

	# trim_to_support's pattern: its seqlets, each end_idx - start_idx long.
	# trim_to_idx adds up that pattern's sums from the seqlets' arrays, which
	# fails where they are shorter than the window; these sums are never
	# read, so here the seqlets must span the pattern instead.
	if not (start_idx < end_idx and pattern.seqlets
			and pattern._spans_length()):
		return _NOT_APPLICABLE

	kept, _, _ = _trimmed(pattern.seqlets, start_idx, end_idx)
	length = len(core.Seqlet(*kept[0]))

	kept, unique, _ = _expanded(kept, length, track_set.length, flank, flank)
	if not kept:
		return None

	length = len(core.Seqlet(*kept[0]))
	expanded = _coordinate_set(track_set, kept, unique, length)
	if expanded is _NOT_APPLICABLE:
		return _NOT_APPLICABLE

	# Trim by IC
	ppm = expanded.sequence
	per_pos_ic = util.compute_per_position_ic(
		ppm=ppm, background=bg_freq, pseudocount=0.001)

	if len(per_pos_ic) < window_size:
		return _NOT_APPLICABLE

	start_idx = np.argmax(util.cpu_sliding_window_sum(
		arr=per_pos_ic, window_size=window_size))

	end_idx = start_idx + window_size
	if not (0 <= start_idx < end_idx <= len(expanded)):
		return _NOT_APPLICABLE

	kept, _, trim_kept = _trimmed(expanded.seqlets, start_idx, end_idx)
	length = len(core.Seqlet(*kept[0]))

	kept, unique, all_kept = _expanded(kept, length, track_set.length, flank,
		flank)
	if not kept:
		return None

	# The trim's seqlets are the expanded pattern's at its rows start_idx to
	# end_idx: when the trim and this expansion keep all of them, the
	# interior of this pattern's sums is those rows of the expanded one's.
	new_length = len(core.Seqlet(*kept[0]))
	reuse = None
	if trim_kept and all_kept and new_length == length + 2 * flank:
		reuse = (tuple(total[start_idx:end_idx] for total in
			expanded._sums()), flank, flank)

	polished = _coordinate_set(track_set, kept, unique, new_length, reuse)
	if polished is _NOT_APPLICABLE:
		return _NOT_APPLICABLE

	return _materialize(polished, track_set)


def _fwd_data(pattern, transformer, include_hypothetical):
	"""The forward-strand data of one pattern: the first array that
	`util.get_2d_data_from_patterns([pattern], transformer,
	include_hypothetical)` returns, computed the same way, without the
	reverse strand that an alignment never uses for the parent."""

	func = util.l1 if transformer == 'l1' else util.magnitude
	tracks = ['hypothetical_contribs', 'contrib_scores']
	if not include_hypothetical:
		tracks = tracks[1:]

	return np.array([np.concatenate([func(getattr(pattern, track))
		for track in tracks], axis=1)])


def _seqlets_data(seqlets, transformer, include_hypothetical):
	"""[util.get_2d_data_from_patterns([seqlet], transformer,
	include_hypothetical) for seqlet in seqlets], with the same bytes.

	For `l1`, when every seqlet's tracks have one shape and dtype, the data
	comes from a single call over the whole list, whose stacked path returns
	the per-seqlet loop's values bit for bit and otherwise falls back to that
	loop. Each seqlet's pair is then a (1, length, width) view of the stacked
	arrays.
	"""

	tracks = ['hypothetical_contribs', 'contrib_scores']
	if not include_hypothetical:
		tracks = tracks[1:]

	if transformer == 'l1' and len(seqlets) > 1:
		kinds = set(tuple((getattr(seqlet, track).shape,
			getattr(seqlet, track).dtype) for track in tracks)
			for seqlet in seqlets)

		if len(kinds) == 1:
			fwd_data, rev_data = util.get_2d_data_from_patterns(seqlets,
				transformer=transformer,
				include_hypothetical=include_hypothetical)
			return [(fwd_data[k:k+1], rev_data[k:k+1])
				for k in range(len(seqlets))]

	return [util.get_2d_data_from_patterns([seqlet], transformer=transformer,
		include_hypothetical=include_hypothetical) for seqlet in seqlets]


def _best_alignment(fwd_data_parent, fwd_data_child, rev_data_child, metric,
	min_overlap):
	"""The better of the child's forward and reverse strands against the
	parent's forward strand: (offset, whether the reverse strand won, its
	score).

	`jaccard` scores each row of its first argument against the second on
	its own: every row gets the same padding (it depends only on the rows'
	shared length), its own scores at each offset and its own argmax. So the
	two strands are stacked into one call, which returns what two calls
	would. `pearson_correlation` normalizes by the norm of its whole first
	argument, so each strand is scored on its own, against the parent's
	windows prepared once (`affinitymat.pearson_correlations`).
	"""

	if metric is affinitymat.jaccard:
		both_data_child = np.concatenate([fwd_data_child, rev_data_child])
		((best_crossmetric, best_crossmetric_argmax), (best_crossmetric_rev,
			best_crossmetric_argmax_rev)) = metric(both_data_child,
			fwd_data_parent, min_overlap)[0]
	elif metric is affinitymat.pearson_correlation:
		# The two calls below, with the parent's padded, normalized windows
		# computed once for both strands.
		fwd, rev = affinitymat.pearson_correlations([fwd_data_child,
			rev_data_child], fwd_data_parent, min_overlap)
		best_crossmetric, best_crossmetric_argmax = fwd.squeeze()
		best_crossmetric_rev, best_crossmetric_argmax_rev = rev.squeeze()
	else:
		best_crossmetric, best_crossmetric_argmax = metric(fwd_data_child,
			fwd_data_parent, min_overlap).squeeze()

		best_crossmetric_rev, best_crossmetric_argmax_rev = metric(
			rev_data_child, fwd_data_parent, min_overlap).squeeze()

	if best_crossmetric_rev > best_crossmetric:
		return int(best_crossmetric_argmax_rev), True, best_crossmetric_rev
	else:
		return int(best_crossmetric_argmax), False, best_crossmetric


def _align_patterns(parent_pattern, child_pattern, metric, min_overlap,
	transformer, include_hypothetical):

	fwd_data_parent = _fwd_data(parent_pattern, transformer,
		include_hypothetical)

	fwd_data_child, rev_data_child = util.get_2d_data_from_patterns(
		[child_pattern], transformer=transformer,
		include_hypothetical=include_hypothetical)

	return _best_alignment(fwd_data_parent, fwd_data_child, rev_data_child,
		metric, min_overlap)


def merge_in_seqlets_filledges(parent_pattern, seqlets_to_merge,
	track_set, metric, min_overlap, transformer='l1',
	include_hypothetical=True):

	parent_pattern = parent_pattern.copy()

	# Each seqlet is aligned, as it was given, against the parent pattern as
	# it stands when its turn comes. The seqlets' own data does not depend on
	# the parent, so it is computed for all of them up front; the parent's is
	# computed again only after the parent changes (fwd_data_parent = None).
	seqlets_to_merge = list(seqlets_to_merge)
	seqlets_data = _seqlets_data(seqlets_to_merge, transformer,
		include_hypothetical)
	fwd_data_parent = None

	# Once the parent has been expanded it is held as a shell (_expand_shell),
	# and its seqlets are made once, when it is returned. `reusable`: the
	# shell's sums are the kernel's for its seqlets, each spanning it, so its
	# next expansion adds up only the new flanks. A seqlet added below is
	# made by create_seqlets at coordinates inside the track, so it keeps
	# that true as long as it spans the parent too.
	shell, reusable = False, False

	for seqlet, (fwd_data_child, rev_data_child) in zip(seqlets_to_merge,
			seqlets_data):
		if fwd_data_parent is None:
			fwd_data_parent = _fwd_data(parent_pattern, transformer,
				include_hypothetical)

		alnmt, revcomp_match, alnmt_score = _best_alignment(fwd_data_parent,
			fwd_data_child, rev_data_child, metric, min_overlap)

		if revcomp_match:
			seqlet = seqlet.revcomp()

		preexpansion_seqletlen = len(seqlet)

		left_expansion = max(alnmt,0)
		right_expansion = max((len(parent_pattern) - (alnmt+len(seqlet))), 0)

		if seqlet.is_revcomp == False:
			start = seqlet.start - left_expansion
			end = seqlet.end + right_expansion
		else:
			start = seqlet.start - right_expansion
			end = seqlet.end + left_expansion

		if start >= 0 and end <= track_set.length:
			seqlet = track_set.create_seqlets(
				seqlets=[core.Seqlet(example_idx=seqlet.example_idx,
					start=start, end=end, is_revcomp=seqlet.is_revcomp)])[0] 
		else:
			continue #don't try adding this seqlet

		#also expand the pattern (if needed) so that the seqlet
		# doesn't go over the edge
		parent_left_expansion = max(0, -alnmt)
		parent_right_expansion = max(0, (alnmt+preexpansion_seqletlen)
										 - len(parent_pattern))

		if (parent_left_expansion > 0) or (parent_right_expansion > 0):
			candidate_parent_pattern = _expand_shell(parent_pattern,
				track_set, parent_left_expansion, parent_right_expansion,
				shell and reusable)

			if candidate_parent_pattern is _NOT_APPLICABLE:
				if shell:
					parent_pattern = _materialize(parent_pattern, track_set)
				shell, reusable = False, False

				candidate_parent_pattern = _expand_seqlets_to_fill_pattern(
					parent_pattern, track_set=track_set, 
					left_flank_to_add=parent_left_expansion,
					right_flank_to_add=parent_right_expansion)
			elif candidate_parent_pattern is not None:
				shell, reusable = True, True

			if candidate_parent_pattern is not None:
				parent_pattern = candidate_parent_pattern
				fwd_data_parent = None
			else:
				continue

		#add the seqlet in at alignment 0, assuming it's not already
		# part of the pattern
		if seqlet.string not in parent_pattern.unique_seqlets:
			if len(seqlet) != len(parent_pattern):
				reusable = False
			parent_pattern._add_seqlet(seqlet=seqlet)
			fwd_data_parent = None

	if shell:
		parent_pattern = _materialize(parent_pattern, track_set)
	return parent_pattern


class _Seqlets(object):
	"""A subsample's seqlets, as SimilarPatternsCollapser reads them."""

	def __init__(self, seqlets):
		self.seqlets = seqlets


class PatternMergeHierarchy(object):
	def __init__(self, root_nodes):
		self.root_nodes = root_nodes

	def add_level(self, level_arr):
		self.levels.append(level_arr)

class PatternMergeHierarchyNode(object):

	def __init__(self, pattern, child_nodes=None, parent_node=None,
					   indices_merged=None, submat_crosscontam=None,
					   submat_alignersim=None): 
		self.pattern = pattern 
		if (child_nodes is None):
			child_nodes = []
		self.child_nodes = child_nodes
		self.parent_node = parent_node
		self.indices_merged = indices_merged
		self.submat_crosscontam = submat_crosscontam
		self.submat_alignersim = submat_alignersim


def _detect_spurious_merging(patterns, track_set, perplexity,
	min_in_subcluster, min_overlap, prob_and_pertrack_sim_merge_thresholds,
	prob_and_pertrack_sim_dealbreaker_thresholds,
	min_frac, min_num, flank_to_add, window_size, bg_freq,
	n_seeds, max_seqlets_subsample=1000):

	def collapse(patterns, known_pairs=None):
		return SimilarPatternsCollapser(patterns=patterns, 
			track_set=track_set, min_overlap=min_overlap, 
			prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
			prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
			min_frac=min_frac, min_num=min_num, flank_to_add=flank_to_add, window_size=window_size, 
			bg_freq=bg_freq, max_seqlets_subsample=max_seqlets_subsample,
			known_pairs=known_pairs)

	def refine(i):
		pattern = patterns[i]
		pattern.compute_subpatterns(perplexity=perplexity, n_seeds=n_seeds,
			track_set=track_set)

		subpatterns = pattern.subcluster_to_subpattern.values()
		return collapse(subpatterns)[0]

	# Each pattern is subclustered and its subpatterns merged independently
	# of the others, so this runs in forked workers (forked.map_tasks). A
	# worker sends back the pattern's membership and each refined
	# subpattern's seqlet coordinates. Here the membership is attached as
	# compute_subpatterns would, and the seqlets are made again from the
	# track set: views of the same arrays with the same strides as the
	# worker's, so every later sum over them adds in the same order.
	def pack(i, refined):
		return patterns[i].subclusters, [[(seqlet.example_idx, seqlet.start,
			seqlet.end, seqlet.is_revcomp) for seqlet in pattern.seqlets]
			for pattern in refined]

	def rebuild(packed):
		return {i: [core.SeqletSet(track_set.create_seqlets([
			core.Seqlet(example_idx=example_idx, start=start, end=end,
				is_revcomp=is_revcomp)
			for example_idx, start, end, is_revcomp in coordinates]),
			track_set=track_set) for coordinates in refined]
			for i, (_, refined) in packed.items()}

	def unpack(packed):
		for i, (subclusters, _) in packed.items():
			patterns[i]._set_subclusters(subclusters, track_set=track_set)
		return rebuild(packed)

	def listed(refined, skip=None):
		"""The final collapser's input: each pattern's refined subpatterns,
		or the pattern itself when it was not refined, in pattern order,
		leaving out pattern `skip`."""

		to_return = []
		for i, pattern in enumerate(patterns):
			if i == skip:
				continue
			if i in refined:
				to_return.extend(refined[i]) 
			else:
				to_return.append(pattern)
		return to_return

	to_refine = [i for i, pattern in enumerate(patterns)
		if len(pattern.seqlets) > min_in_subcluster]
	weights = [len(patterns[i].seqlets) ** 2 for i in to_refine]

	# When nothing is forked, when the plan does not give the heaviest
	# pattern a share of its own, or when that share is not expected to
	# finish last (forked.finishes_last), forked.map_tasks runs the tasks
	# as planned.
	n_budget = forked.budget()
	bins, loads = forked._plan(weights, n_budget)
	if (len(bins) < 2 or len(bins[0]) > 1 or not forked.numba_fork_safe()
			or not forked.finishes_last(loads, forked._split_budget(loads,
				n_budget), n_seeds)):
		refined = forked.map_tasks(refine, to_refine, weights=weights,
			pack=pack, unpack=unpack, before_fork=lambda:
				patterns[to_refine[0]]._compile_subpattern_kernels())
		return collapse(listed(refined))

	# The heaviest pattern's subclustering is the last to finish (its Leiden
	# runs are the longest, twice as long as any other share's), and
	# forked.map_tasks's plan gives it a share of its own, run by this
	# process. Its affinity matrix is computed first, with the whole thread
	# budget, and then its Leiden runs and its merging with the share's
	# threads. The other shares run beside it as map_tasks would run them,
	# with the same threads, under one forked child (forked.Background)
	# that computes the first of them itself and forks the rest. That child
	# then also computes, while this process is still in its Leiden runs,
	# the final collapser's first entries for every pair of the patterns it
	# holds (each refined subpattern of the other patterns, and each pattern
	# too small to refine): an entry depends only on its two patterns, so
	# the collapser takes them as known_pairs and computes only the pairs
	# that involve the heaviest pattern's subpatterns. The child computes
	# them on the patterns as this process rebuilds them (rebuild), and if
	# it fails, result() computes the same here.
	budgets = forked._split_budget(loads, n_budget)
	critical = to_refine[bins[0][0]]
	others = [to_refine[j] for b in bins[1:] for j in b]

	def beside():
		refined = forked.map_shares(refine, [[to_refine[j] for j in b]
			for b in bins[1:]], budgets[1:], pack=pack, unpack=unpack)
		packed = {i: pack(i, refined[i]) for i in others}
		return packed, _first_pair_stats(listed(rebuild(packed),
			skip=critical), track_set, min_overlap, max_seqlets_subsample)

	affinity = patterns[critical]._subpattern_affinity(perplexity)

	helper = forked.Background(beside)
	helper.start(n_budget - budgets[0])
	try:
		with forked.budget_of(budgets[0]):
			patterns[critical].compute_subpatterns(perplexity=perplexity,
				n_seeds=n_seeds, affinity=affinity, track_set=track_set)
			del affinity

			refined_critical = collapse(
				patterns[critical].subcluster_to_subpattern.values())[0]
	except BaseException:
		helper.kill()
		raise

	packed, table = helper.result()
	refined = unpack(packed)
	refined[critical] = refined_critical

	# The index in the final collapser's input of each pattern the child
	# listed: those before the heaviest pattern's subpatterns keep theirs,
	# the others move up by the number of its subpatterns.
	n_before = sum(len(refined[i]) if i in refined else 1
		for i in range(critical))
	final = [a if a < n_before else a + len(refined_critical)
		for a in range(len(table))]
	known_pairs = {(final[a], final[b]): (table[a, b, 0], table[a, b, 1])
		for a in range(len(table)) for b in range(len(table)) if a != b}

	return collapse(listed(refined), known_pairs=known_pairs)

# The least estimated work, in seconds on one thread, that is worth giving
# a forked worker in SimilarPatternsCollapser: about four forks of a process
# of this size, which the parent makes one after another (8-13 ms each on a
# 20,000-seqlet run) before it starts its own share.
PAIR_MIN_CHILD_SECONDS = 0.05


def _pair_seconds(n_i, n_j, length_i):
	"""Estimated seconds on one thread to compute one (i, j) pair's entries
	in SimilarPatternsCollapser, for subsamples of n_i and n_j seqlets and
	pattern i of length length_i: a fixed cost (the alignment), shifting
	and stacking pattern j's seqlets, the between-pattern Jaccard (n_i x n_j
	rows of 8 * length_i values) and the AUROC's pass over the n_i ** 2
	within-pattern scores (sorted once per pattern, before the pairs).
	Fitted to the 2,146 pairs of the largest pattern's refine collapser and
	of the final collapser on a 20,000-seqlet run (the median pair within
	10% of its estimate); used only to balance the pairs across forked
	workers and to decide how many to fork."""

	return (1.8e-4 + 4.0e-6 * n_j + 1.1e-9 * n_i * n_j * length_i
		+ 1.9e-9 * n_i * n_i)


def _subsamples(patterns, max_seqlets_subsample):
	"""The seqlets SimilarPatternsCollapser scores for each pattern: all of
	them, or, for a pattern with more than max_seqlets_subsample, a
	subsample drawn from a fresh RandomState(1234). Only a subsample's
	seqlets are used: the ones core.SeqletSet(seqlets=subsample) would keep,
	the first of each string, without its running sums."""

	subsample_patterns = []
	for pattern in patterns:
		if len(pattern.seqlets) > max_seqlets_subsample:
			subsample = np.random.RandomState(1234).choice(
				a=pattern.seqlets, replace=False, size=max_seqlets_subsample)
			kept = {}
			for seqlet in subsample:
				kept.setdefault(seqlet.string, seqlet)
			pattern = _Seqlets(list(kept.values()))

		subsample_patterns.append(pattern)

	return subsample_patterns


def _pair_map(patterns, subsample_patterns, pattern1_cache, pairs, track_set,
	min_overlap):
	"""{(i, j): (aligner similarity, AUROC)} for each pair, the
	pairwise_sims and pairwise_aurocs entries SimilarPatternsCollapser
	fills in for it.

	Pattern i's flattened subsample data and its within-pattern Jaccard
	depend only on pattern i, so they are computed once per pattern object
	rather than once per (i, j) pair, and kept in pattern1_cache. The
	within-pattern scores are used only by the AUROC, which sorts them, so
	they are kept sorted, and a pair sorts only its between-pattern scores,
	in place: a forked worker then allocates no more than one copy of
	those. The key is the object itself (identity hash, and the dict keeps
	it alive so its id is not reused)."""

	def pattern1_data(i):
		if patterns[i] not in pattern1_cache:
			pattern1_fwdseqdata, _ =\
			  util.get_2d_data_from_patterns(subsample_patterns[i].seqlets)

			flat_pattern1_fwdseqdata = pattern1_fwdseqdata.reshape(
				(len(pattern1_fwdseqdata), -1))

			within_pattern1_sims = affinitymat.jaccard(
				flat_pattern1_fwdseqdata[:, :, None],
				flat_pattern1_fwdseqdata[:, :, None])[:, :, 0].flatten()
			within_pattern1_sims.sort()

			pattern1_cache[patterns[i]] = (flat_pattern1_fwdseqdata,
				within_pattern1_sims)

		return pattern1_cache[patterns[i]]

	def pair_stats(pair):
		"""The pairwise_sims and pairwise_aurocs entries of (i, j)."""

		i, j = pair

		#Compute best alignment between pattern pair
		alnmt, rc, aligner_sim =\
			_align_patterns(parent_pattern=patterns[i],
				child_pattern=patterns[j],
				metric=affinitymat.pearson_correlation,
				min_overlap=min_overlap,
				include_hypothetical=False,
				transformer='magnitude')

		#get realigned pattern2
		pattern2_coords = subsample_patterns[j].seqlets
		if rc: #flip strand if needed to align
			pattern2_coords  = [x.revcomp() for x in pattern2_coords]

		#now apply the alignment
		pattern2_coords = [x.shift((1 if x.is_revcomp else -1)*alnmt)
			for x in pattern2_coords]

		# Filter out bad seqlets
		pattern2_coords = [seqlet for seqlet in pattern2_coords
			if seqlet.start >= 0 and seqlet.end < track_set.length]

		if len(pattern2_coords) == 0:
			return 0.0, 0.5

		pattern2_shifted_seqlets = track_set.create_seqlets(
			seqlets=pattern2_coords)

		flat_pattern1_fwdseqdata, within_pattern1_sims = pattern1_data(i)

		pattern2_fwdseqdata, _ =\
		  util.get_2d_data_from_patterns(pattern2_shifted_seqlets)

		#Flatten, compute continjacc sim at this alignment
		flat_pattern2_fwdseqdata = pattern2_fwdseqdata.reshape(
			(len(pattern2_fwdseqdata), -1))

		between_pattern_sims = affinitymat.jaccard(
			flat_pattern1_fwdseqdata[:, :, None],
			flat_pattern2_fwdseqdata[:, :, None])[:, :, 0].flatten()
		between_pattern_sims.sort()

		auroc = _roc_auc_sorted(between_pattern_sims,
			within_pattern1_sims)

		return aligner_sim, auroc

	# Each pair's entries depend only on the two patterns, their
	# subsamples and the track set, so the pairs are computed in forked
	# workers (forked.map_tasks) when there is enough work. A worker
	# sends back each pair's two float64s as raw bytes. Pattern i's
	# cached data is computed here before forking, so that the workers
	# share it and it is kept for the next while iteration.
	return forked.map_tasks(pair_stats, pairs,
		weights=[_pair_seconds(len(subsample_patterns[i].seqlets),
			len(subsample_patterns[j].seqlets), len(patterns[i]))
			for i, j in pairs],
		pack=lambda pair, values: np.array(values,
			dtype=np.float64).tobytes(),
		unpack=lambda packed: {pair: tuple(np.frombuffer(values,
			dtype=np.float64)) for pair, values in packed.items()},
		before_fork=lambda: [pattern1_data(i) for i in
			sorted(set(i for i, _ in pairs))],
		min_child_weight=PAIR_MIN_CHILD_SECONDS)


def _first_pair_stats(patterns, track_set, min_overlap, max_seqlets_subsample):
	"""The entries SimilarPatternsCollapser computes for every pair (i, j),
	i != j, of `patterns` in its first iteration, as an (n, n, 2) float64
	array of (aligner similarity, AUROC); the diagonal is not used."""

	n = len(patterns)
	pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
	stats = _pair_map(patterns, _subsamples(patterns, max_seqlets_subsample),
		{}, pairs, track_set, min_overlap)

	table = np.zeros((n, n, 2), dtype=np.float64)
	for (i, j), values in stats.items():
		table[i, j] = values
	return table


def SimilarPatternsCollapser(patterns, track_set,
	min_overlap, prob_and_pertrack_sim_merge_thresholds,
	prob_and_pertrack_sim_dealbreaker_thresholds,
	min_frac, min_num, flank_to_add, window_size, bg_freq,
	max_seqlets_subsample=1000, known_pairs=None):
	patterns = [x.copy() for x in patterns]

	merge_hierarchy_levels = []        
	current_level_nodes = [
		PatternMergeHierarchyNode(pattern=x) for x in patterns]
	merge_hierarchy_levels.append(current_level_nodes)

	merge_occurred_last_iteration = True

	#negative numbers to indicate which
	# entries need to be filled (versus entries we can infer
	# from the previous iteration of the while loop)
	pairwise_aurocs = -np.ones((len(patterns), len(patterns)))
	pairwise_sims = np.zeros((len(patterns), len(patterns)))

	# Pattern i's subsample data and within-pattern scores, by pattern
	# object (_pair_map). An object's subsample is the same in every while
	# iteration: its seqlets are never changed once it is in `patterns`
	# (merging copies the parent first), and the subsample is drawn from a
	# fresh RandomState(1234) each time.
	pattern1_cache = {}

	#loop until no more patterns get merged
	while merge_occurred_last_iteration:
		merge_occurred_last_iteration = False

		# Drop the entries of patterns merged away in the last iteration.
		pattern1_cache = {pattern: pattern1_cache[pattern] for pattern in
			patterns if pattern in pattern1_cache}

		#Let's subsample 'patterns' to prevent runtime from being too
		# large in calculating pairwise sims. 
		subsample_patterns = _subsamples(patterns, max_seqlets_subsample)

		n = len(patterns)
		pairs = []
		for i in range(n):
			for j in range(n):
				#Note: I compute both i,j AND j,i because although
				# the result is the same for the sim, it can be different
				# for the auroc because a different motif is getting
				# shifted over.
				if j == i:
					pairwise_aurocs[i, j] = 0.5
					pairwise_sims[i, j] = 1.0
					continue

				if pairwise_aurocs[i, j] >= 0: #filled in from previous iter
					continue

				pairs.append((i, j))

		# Pairs of the first iteration whose entries the caller computed
		# already (known_pairs, keyed by the indices of `patterns` as given):
		# the same values _pair_map would compute here.
		if known_pairs is not None:
			for i, j in pairs:
				if (i, j) in known_pairs:
					pairwise_sims[i, j], pairwise_aurocs[i, j] =\
						known_pairs[(i, j)]
			pairs = [pair for pair in pairs if pair not in known_pairs]
			known_pairs = None

		stats = _pair_map(patterns, subsample_patterns, pattern1_cache, pairs,
			track_set, min_overlap)

		for i, j in pairs:
			#The symmetrization over i,j and j,i is done later
			pairwise_sims[i, j], pairwise_aurocs[i, j] = stats[(i, j)]


		#pairwise_sims is not symmetric; differ based on which pattern is
		# padded with zeros.
		patterns_to_patterns_aligner_sim = 0.5*(pairwise_sims+pairwise_sims.T)
		cross_contamination = 2*(1-np.maximum(pairwise_aurocs,0.5))

		indices_to_merge = []
		merge_partners_so_far = dict([(i, set([i])) for i in
									  range(len(patterns))])

		#merge patterns with highest similarity first
		sorted_pairs = sorted([(i,j,patterns_to_patterns_aligner_sim[i,j])
						for i in range(len(patterns))
						for j in range(len(patterns)) if (i < j)],
						key=lambda x: -x[2])

		#iterate over pairs
		for i, j, aligner_sim in sorted_pairs:
			#symmetrize asymmetric crosscontam
			# take min rather than avg to avoid aggressive merging
			cross_contam = min(cross_contamination[i,j],
								cross_contamination[j,i])

			collapse = any([(cross_contam >= x[0] and aligner_sim >= x[1])
				for x in prob_and_pertrack_sim_merge_thresholds])

			if collapse:
				collapse_passed = True
				#check compatibility for all indices that are
				#about to be merged
				merge_under_consideration = set(
					list(merge_partners_so_far[i])
					+list(merge_partners_so_far[j]))

				for m1 in merge_under_consideration:
					for m2 in merge_under_consideration:
						if (m1 < m2):
							cross_contam_here =\
								0.5*(cross_contamination[m1, m2]+
									 cross_contamination[m2, m1])
							aligner_sim_here =\
								patterns_to_patterns_aligner_sim[
									m1, m2]

							dealbreaker = any([(cross_contam_here <= x[0] and aligner_sim_here <= x[1])              
								for x in prob_and_pertrack_sim_dealbreaker_thresholds])

							if dealbreaker:
								collapse_passed = False
								break

				if collapse_passed:
					indices_to_merge.append((i,j))
					for an_idx in merge_under_consideration:
						merge_partners_so_far[an_idx]=\
							merge_under_consideration 

		for i, j in indices_to_merge:
			pattern1 = patterns[i]
			pattern2 = patterns[j]

			if pattern1 != pattern2: #if not the same object
				if len(pattern1.seqlets) < len(pattern2.seqlets):
					parent_pattern, child_pattern = pattern2, pattern1
				else:
					parent_pattern, child_pattern = pattern1, pattern2

				new_pattern = merge_in_seqlets_filledges(
					parent_pattern=parent_pattern,
					seqlets_to_merge=child_pattern.seqlets,
					include_hypothetical=False,
					metric=affinitymat.pearson_correlation,
					min_overlap=min_overlap,
					transformer='magnitude',
					track_set=track_set)

				new_pattern = polish_pattern(new_pattern, min_frac=min_frac, 
					min_num=min_num, track_set=track_set, flank=flank_to_add, 
					window_size=window_size, bg_freq=bg_freq)

				if new_pattern is not None:
					for k in range(len(patterns)):
						#Replace EVERY case where the parent or child
						# pattern is present with the new pattern. This
						# effectively does single-linkage.
						if (patterns[k]==parent_pattern or
							patterns[k]==child_pattern):
							patterns[k]=new_pattern

		merge_occurred_last_iteration = (len(indices_to_merge) > 0)

		if merge_occurred_last_iteration:
			#Once we are here, each element of 'patterns'
			#will have the new parent of the corresponding element
			#of 'old_patterns'
			old_to_new_pattern_mapping = patterns

			#sort by size and remove redundant patterns
			patterns = sorted(patterns, key=lambda x: -len(x.seqlets))
			patterns = list(OrderedDict([(x,1) for x in patterns]).keys())

			#let's figure out which indices don't require recomputation
			# and use it to repopulate pairwise_sims and pairwise_aurocs
			old_to_new_index_mappings = OrderedDict()
			for old_pattern_idx,(old_pattern_node, corresp_new_pattern)\
				in enumerate(zip(current_level_nodes,
								 old_to_new_pattern_mapping)):
				#if the old pattern was NOT changed in this iteration
				if old_pattern_node.pattern == corresp_new_pattern:
					new_idx = patterns.index(corresp_new_pattern) 
					old_to_new_index_mappings[old_pattern_idx] = new_idx

			new_pairwise_aurocs = -np.ones((len(patterns), len(patterns)))
			new_pairwise_sims = np.zeros((len(patterns), len(patterns)))
			for old_idx_i, new_idx_i in\
				old_to_new_index_mappings.items():
				for old_idx_j, new_idx_j in\
					old_to_new_index_mappings.items():
					new_pairwise_aurocs[new_idx_i, new_idx_j] =\
						pairwise_aurocs[old_idx_i, old_idx_j]
					new_pairwise_sims[new_idx_i, new_idx_j] =\
						pairwise_sims[old_idx_i, old_idx_j]
			pairwise_aurocs = new_pairwise_aurocs 
			pairwise_sims = new_pairwise_sims
				 

			#update the hierarchy
			#the current 'top level' will consist of all the current
			# nodes that didn't get a new parent, plus any new parents
			# created                
			next_level_nodes = []
			for frontier_pattern in patterns:
				#either this pattern is in old_pattern_nodes, in which
				# case take the old_pattern_node entry, or it's a completely
				# new pattern in which case make a node for it
				old_pattern_node_found = False
				for old_pattern_node in current_level_nodes:
					if (old_pattern_node.pattern==frontier_pattern):
						#sanity check..there should be only one node
						# per pattern
						next_level_nodes.append(old_pattern_node)
						old_pattern_node_found = True 
				if (old_pattern_node_found==False):
				   next_level_nodes.append(
					PatternMergeHierarchyNode(frontier_pattern)) 

			for next_level_node in next_level_nodes:
				#iterate over all the old patterns and their new parent
				# in order to set up the child nodes correctly
				for old_pattern_idx,(old_pattern_node, corresp_new_pattern)\
					in enumerate(zip(current_level_nodes,
									 old_to_new_pattern_mapping)):

					#if the node has a new parent
					if (old_pattern_node.pattern != corresp_new_pattern):
						if (next_level_node.pattern==corresp_new_pattern):

							
							#corresp_new_pattern should be comprised of a 
							# merging of all the old patterns at
							# indices_merged_with
							indices_merged = tuple(sorted(
								merge_partners_so_far[old_pattern_idx])) 
							#get the relevant slice         
							submat_crosscontam =\
							 cross_contamination[indices_merged,:][:,
												 indices_merged]
							submat_alignersim =\
							 patterns_to_patterns_aligner_sim[
								indices_merged, :][:,indices_merged]

							if (next_level_node.indices_merged is not None):
								assert (next_level_node.indices_merged
										==indices_merged),\
								 (next_level_node.indices_merged,
								  indices_merged)
							else:
								next_level_node.indices_merged =\
									indices_merged
								next_level_node.submat_crosscontam =\
									submat_crosscontam
								next_level_node.submat_alignersim =\
									submat_alignersim

							next_level_node.child_nodes.append(
											old_pattern_node) 

							old_pattern_node.parent_node = next_level_node

			current_level_nodes = next_level_nodes

	return patterns, PatternMergeHierarchy(root_nodes=current_level_nodes)
	