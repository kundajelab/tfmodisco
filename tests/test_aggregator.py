# test_aggregator.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import pytest

from modiscolite.aggregator import polish_pattern
from modiscolite.aggregator import _expand_seqlets_to_fill_pattern
from modiscolite.aggregator import _align_patterns
from modiscolite.aggregator import merge_in_seqlets_filledges
from modiscolite.aggregator import PatternMergeHierarchy
from modiscolite.aggregator import PatternMergeHierarchyNode
from modiscolite.aggregator import _detect_spurious_merging
from modiscolite.aggregator import SimilarPatternsCollapser

from modiscolite.affinitymat import jaccard
from modiscolite.affinitymat import pearson_correlation
from modiscolite.core import SeqletSet
from modiscolite.core import TrackSet

from .synthetic import planted_track_set
from .synthetic import two_motif_track_set
from .synthetic import make_seqlets
from .synthetic import make_pattern

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


BG = numpy.full(4, 0.25)

MERGE = [(0.8, 0.8), (0.5, 0.85), (0.2, 0.9)]
DEALBREAKER = [(0.4, 0.75), (0.2, 0.8), (0.1, 0.85), (0.0, 0.9)]

# SimilarPatternsCollapser assumes that every pattern is window_size +
# 2*flank_to_add long, which the patterns it receives in the pipeline always
# are. COLLAPSE fits 18 bp patterns around the 8 bp motif of motif_ts and
# COLLAPSE_TWO fits 20 bp patterns around the 10 bp motifs of
# two_motif_track_set.
COLLAPSE = dict(min_overlap=0.7, prob_and_pertrack_sim_merge_thresholds=MERGE,
	prob_and_pertrack_sim_dealbreaker_thresholds=DEALBREAKER, min_frac=0.2,
	min_num=30, flank_to_add=5, window_size=8, bg_freq=BG)

COLLAPSE_TWO = dict(COLLAPSE, window_size=10)


def _coords(pattern):
	return [(s.example_idx, s.start, s.end, s.is_revcomp)
		for s in pattern.seqlets]


def _assert_consistent(pattern, track_set):
	"""Every seqlet's data matches the track set at its coordinates."""

	for seqlet in pattern.seqlets:
		expected = make_seqlets(track_set, [(seqlet.example_idx, seqlet.start,
			seqlet.end, seqlet.is_revcomp)])[0]

		assert_array_equal(seqlet.sequence, expected.sequence)
		assert_array_equal(seqlet.contrib_scores, expected.contrib_scores)
		assert_array_equal(seqlet.hypothetical_contribs,
			expected.hypothetical_contribs)


@pytest.fixture
def motif_ts():
	"""30 examples of length 60 with an 8 bp motif at positions 26-34."""

	return planted_track_set(n=30, length=60, motif_length=8, position=26)


@pytest.fixture
def motif_pattern(motif_ts):
	coords = [(i, 16, 44, bool(i % 3 == 0)) for i in range(15)]
	return make_pattern(motif_ts, coords)


@pytest.fixture
def edge_ts(motif_ts):
	"""motif_ts with the motif of example 29 moved to positions 50-58."""

	one_hot = motif_ts.one_hot.copy()
	hyp = motif_ts.hypothetical_contribs.copy()
	one_hot[29] = numpy.roll(one_hot[29], 24, axis=0)
	hyp[29] = numpy.roll(hyp[29], 24, axis=0)
	return TrackSet(one_hot, one_hot*hyp, hyp)


##


def test_polish_pattern_planted(motif_ts, motif_pattern):
	pattern = polish_pattern(motif_pattern, min_frac=0.2, min_num=30,
		track_set=motif_ts, flank=5, window_size=8, bg_freq=BG)

	assert isinstance(pattern, SeqletSet)
	assert len(pattern) == 18
	assert _coords(pattern) == [(i, 21, 39, bool(i % 3 == 0))
		for i in range(15)]
	_assert_consistent(pattern, motif_ts)


@pytest.mark.parametrize("window_size", [4, 6, 8, 12])
@pytest.mark.parametrize("flank", [0, 2, 5, 10])
def test_polish_pattern(motif_ts, motif_pattern, window_size, flank):
	pattern = polish_pattern(motif_pattern, min_frac=0.2, min_num=30,
		track_set=motif_ts, flank=flank, window_size=window_size, bg_freq=BG)

	assert len(pattern) == window_size + 2*flank
	assert len(pattern.seqlets) == 15
	_assert_consistent(pattern, motif_ts)

	core = [(s.start + flank, s.end - flank) for s in pattern.seqlets]
	for start, end in core:
		assert end - start == window_size
		assert max(start, 26) < min(end, 34)
		if window_size >= 8:
			assert start <= 26 and end >= 34


def test_polish_pattern_real(pos_patterns, signed_track_set):
	for pattern in pos_patterns:
		polished = polish_pattern(pattern, min_frac=0.2, min_num=30,
			track_set=signed_track_set, flank=10, window_size=30,
			bg_freq=BG)

		assert len(polished) == 50
		assert len(polished.seqlets) <= len(pattern.seqlets)
		assert len(polished.seqlets) >= 0.9 * len(pattern.seqlets)
		_assert_consistent(polished, signed_track_set)


def test_polish_pattern_background(motif_ts, motif_pattern):
	# The background changes the information content but not the location
	# of a motif this strong.
	bg = numpy.array([0.4, 0.1, 0.1, 0.4])
	pattern0 = polish_pattern(motif_pattern, 0.2, 30, motif_ts, 5, 8, BG)
	pattern1 = polish_pattern(motif_pattern, 0.2, 30, motif_ts, 5, 8, bg)
	assert _coords(pattern0) == _coords(pattern1)


def test_polish_pattern_drops_edge(motif_ts):
	coords = [(0, 16, 44, False), (1, 16, 44, False), (2, 0, 28, False)]
	pattern = make_pattern(motif_ts, coords)
	polished = polish_pattern(pattern, 0.2, 30, motif_ts, 5, 8, BG)

	assert [s.example_idx for s in polished.seqlets] == [0, 1]


def test_polish_pattern_none(motif_ts):
	pattern = make_pattern(motif_ts, [(0, 0, 30, False), (1, 30, 60, True)])
	assert polish_pattern(pattern, 0.2, 30, motif_ts, 5, 8, BG) is None


def test_polish_pattern_does_not_modify(motif_ts, motif_pattern):
	coords = _coords(motif_pattern)
	polish_pattern(motif_pattern, 0.2, 30, motif_ts, 5, 8, BG)
	assert _coords(motif_pattern) == coords


##


@pytest.mark.parametrize("left", [0, 3, 10])
@pytest.mark.parametrize("right", [0, 3, 10])
def test_expand_seqlets_to_fill_pattern(motif_ts, motif_pattern, left, right):
	pattern = _expand_seqlets_to_fill_pattern(motif_pattern, motif_ts,
		left_flank_to_add=left, right_flank_to_add=right)

	assert len(pattern) == 28 + left + right
	assert len(pattern.seqlets) == 15
	for seqlet in pattern.seqlets:
		if seqlet.is_revcomp:
			assert (seqlet.start, seqlet.end) == (16 - right, 44 + left)
		else:
			assert (seqlet.start, seqlet.end) == (16 - left, 44 + right)

	_assert_consistent(pattern, motif_ts)


def test_expand_seqlets_to_fill_pattern_uneven(motif_ts):
	coords = [(0, 10, 30, False), (1, 10, 25, False), (2, 20, 30, True),
		(3, 10, 20, True)]
	pattern = make_pattern(motif_ts, coords)
	expanded = _expand_seqlets_to_fill_pattern(pattern, motif_ts, 2, 3)

	# Shorter seqlets grow on the right to fill the pattern, which is the left
	# of the genome for the reverse strand, so the last seqlet falls off.
	assert _coords(expanded) == [(0, 8, 33, False), (1, 8, 33, False),
		(2, 7, 32, True)]
	assert_array_equal(expanded.per_position_counts, numpy.full(25, 3))
	_assert_consistent(expanded, motif_ts)


def test_expand_seqlets_to_fill_pattern_zero(motif_ts, motif_pattern):
	pattern = _expand_seqlets_to_fill_pattern(motif_pattern, motif_ts, 0, 0)

	assert pattern is not motif_pattern
	assert _coords(pattern) == _coords(motif_pattern)
	assert_array_almost_equal(pattern.sequence, motif_pattern.sequence)


@pytest.mark.parametrize("left,right,expected", [(10, 10, 15), (11, 0, 5),
	(0, 11, 10), (22, 0, 5), (0, 22, 10), (23, 0, 0), (11, 11, 0)])
def test_expand_seqlets_to_fill_pattern_bounds(motif_ts, left, right,
	expected):
	# Seqlets span 10-38 of a length 60 track and one in three is on the
	# reverse strand. Forward seqlets have 10 bp of room on the left and 22 bp
	# on the right; reverse strand seqlets the opposite.
	coords = [(i, 10, 38, bool(i % 3 == 0)) for i in range(15)]
	pattern = make_pattern(motif_ts, coords)
	expanded = _expand_seqlets_to_fill_pattern(pattern, motif_ts, left, right)

	if expected == 0:
		assert expanded is None
	else:
		assert len(expanded.seqlets) == expected
		assert all(s.start >= 0 and s.end <= 60 for s in expanded.seqlets)


def test_expand_seqlets_to_fill_pattern_real(pos_patterns, signed_track_set):
	pattern = pos_patterns[0]
	expanded = _expand_seqlets_to_fill_pattern(pattern, signed_track_set, 5, 5)

	assert len(expanded) == len(pattern) + 10
	assert len(expanded.seqlets) <= len(pattern.seqlets)
	_assert_consistent(expanded, signed_track_set)


##


@pytest.mark.parametrize("metric,transformer,include_hypothetical", [
	(jaccard, 'l1', True), (jaccard, 'l1', False),
	(pearson_correlation, 'magnitude', False),
	(pearson_correlation, 'magnitude', True)])
def test_align_patterns_self(motif_pattern, metric, transformer,
	include_hypothetical):
	offset, rc, score = _align_patterns(motif_pattern, motif_pattern, metric,
		0.7, transformer, include_hypothetical)

	assert offset == 0
	assert rc is False
	assert abs(score - 1) < 1e-5


@pytest.mark.parametrize("metric,transformer", [(jaccard, 'l1'),
	(pearson_correlation, 'magnitude')])
def test_align_patterns_revcomp(motif_ts, metric, transformer):
	parent = make_pattern(motif_ts, [(i, 16, 44, False) for i in range(10)])
	child = make_pattern(motif_ts, [(i, 16, 44, True) for i in range(10)])

	offset, rc, score = _align_patterns(parent, child, metric, 0.7,
		transformer, False)

	assert offset == 0
	assert rc is True
	assert abs(score - 1) < 1e-5


@pytest.mark.parametrize("metric,transformer", [(jaccard, 'l1'),
	(pearson_correlation, 'magnitude')])
@pytest.mark.parametrize("start", [16, 18, 20, 22, 24])
def test_align_patterns_offset(motif_ts, metric, transformer, start):
	parent = make_pattern(motif_ts, [(i, 16, 44, False) for i in range(10)])
	child = make_pattern(motif_ts, [(i, start, start+20, False)
		for i in range(10, 20)])

	offset, rc, score = _align_patterns(parent, child, metric, 0.7,
		transformer, False)

	assert offset == start - 16
	assert rc is False


@pytest.mark.parametrize("start", [16, 20, 24])
def test_align_patterns_offset_revcomp(motif_ts, start):
	parent = make_pattern(motif_ts, [(i, 16, 44, False) for i in range(10)])
	child = make_pattern(motif_ts, [(i, start, start+20, True)
		for i in range(10, 20)])

	offset, rc, score = _align_patterns(parent, child, jaccard, 0.7, 'l1',
		False)

	# The flipped child is the forward strand of its window.
	assert rc is True
	assert offset == start - 16


def test_align_patterns_types(motif_pattern):
	offset, rc, score = _align_patterns(motif_pattern, motif_pattern, jaccard,
		0.7, 'l1', True)

	assert isinstance(offset, int)
	assert isinstance(rc, bool)
	assert numpy.isscalar(score)


def test_align_patterns_seqlet(motif_ts, motif_pattern):
	# A single seqlet can stand in for the child pattern.
	seqlet = make_seqlets(motif_ts, [(20, 20, 40, False)])[0]
	offset, rc, score = _align_patterns(motif_pattern, seqlet, jaccard, 0.7,
		'l1', True)
	assert offset == 4


def test_align_patterns_real(pos_patterns):
	offset, rc, score = _align_patterns(pos_patterns[0], pos_patterns[0],
		pearson_correlation, 0.7, 'magnitude', False)
	assert (offset, rc) == (0, False)
	assert abs(score - 1) < 1e-6

	offset, rc, score = _align_patterns(pos_patterns[0], pos_patterns[1],
		pearson_correlation, 0.7, 'magnitude', False)
	assert (offset, rc) == (11, True)
	assert abs(score - 0.7721) < 1e-4


##


def test_merge_in_seqlets_filledges(motif_ts, motif_pattern):
	seqlets = make_seqlets(motif_ts, [(i, 16, 44, bool(i % 2))
		for i in range(15, 25)])
	merged = merge_in_seqlets_filledges(motif_pattern, seqlets, motif_ts,
		jaccard, 0.7)

	assert merged is not motif_pattern
	assert len(merged.seqlets) == 25
	assert len(merged) == 28
	_assert_consistent(merged, motif_ts)


def test_merge_in_seqlets_filledges_strand(motif_ts):
	parent = make_pattern(motif_ts, [(i, 16, 44, False) for i in range(5)])
	seqlets = make_seqlets(motif_ts, [(i, 16, 44, True) for i in range(5, 10)])
	merged = merge_in_seqlets_filledges(parent, seqlets, motif_ts, jaccard, 0.7)

	# The merged seqlets were flipped to match the parent's orientation.
	assert _coords(merged)[5:] == [(i, 16, 44, False) for i in range(5, 10)]


def test_merge_in_seqlets_filledges_shifted(motif_ts):
	parent = make_pattern(motif_ts, [(i, 16, 44, False) for i in range(5)])
	seqlets = make_seqlets(motif_ts, [(5, 20, 40, False), (6, 18, 38, True),
		(7, 22, 42, False)])
	merged = merge_in_seqlets_filledges(parent, seqlets, motif_ts, jaccard, 0.7)

	# Each seqlet is expanded to cover the whole parent.
	assert _coords(merged)[5:] == [(5, 16, 44, False), (6, 16, 44, False),
		(7, 16, 44, False)]
	_assert_consistent(merged, motif_ts)


def test_merge_in_seqlets_filledges_expands_parent(motif_ts):
	parent = make_pattern(motif_ts, [(i, 26, 54, False) for i in range(5)])
	seqlets = make_seqlets(motif_ts, [(5, 20, 40, False)])
	merged = merge_in_seqlets_filledges(parent, seqlets, motif_ts, jaccard, 0.7)

	# The seqlet starts 6 bp before the parent, so the parent grows by 6 bp.
	assert len(merged) == 34
	assert _coords(merged) == [(i, 20, 54, False) for i in range(6)]
	_assert_consistent(merged, motif_ts)


def test_merge_in_seqlets_filledges_skips_edge(edge_ts):
	parent = make_pattern(edge_ts, [(i, 16, 44, False) for i in range(5)])
	seqlets = make_seqlets(edge_ts, [(29, 46, 60, False), (5, 16, 44, False)])
	merged = merge_in_seqlets_filledges(parent, seqlets, edge_ts, jaccard, 0.7)

	assert [s.example_idx for s in merged.seqlets] == [0, 1, 2, 3, 4, 5]


def test_merge_in_seqlets_filledges_duplicates(motif_ts, motif_pattern):
	merged = merge_in_seqlets_filledges(motif_pattern, motif_pattern.seqlets,
		motif_ts, jaccard, 0.7)

	assert _coords(merged) == _coords(motif_pattern)
	assert_array_almost_equal(merged.sequence, motif_pattern.sequence)


def test_merge_in_seqlets_filledges_empty(motif_ts, motif_pattern):
	merged = merge_in_seqlets_filledges(motif_pattern, [], motif_ts, jaccard,
		0.7)

	assert merged is not motif_pattern
	assert _coords(merged) == _coords(motif_pattern)


def test_merge_in_seqlets_filledges_does_not_modify(motif_ts, motif_pattern):
	coords = _coords(motif_pattern)
	seqlets = make_seqlets(motif_ts, [(20, 16, 44, False)])
	merge_in_seqlets_filledges(motif_pattern, seqlets, motif_ts, jaccard, 0.7)

	assert _coords(motif_pattern) == coords
	assert len(motif_pattern.unique_seqlets) == 15


@pytest.mark.parametrize("metric,transformer,include_hypothetical", [
	(jaccard, 'l1', True), (jaccard, 'magnitude', False),
	(pearson_correlation, 'magnitude', False)])
def test_merge_in_seqlets_filledges_metrics(motif_ts, motif_pattern, metric,
	transformer, include_hypothetical):
	seqlets = make_seqlets(motif_ts, [(i, 18, 38, bool(i % 2))
		for i in range(15, 25)])
	merged = merge_in_seqlets_filledges(motif_pattern, seqlets, motif_ts,
		metric, 0.7, transformer=transformer,
		include_hypothetical=include_hypothetical)

	assert len(merged.seqlets) == 25
	assert all((s.start, s.end) == (16, 44) for s in merged.seqlets)


@pytest.mark.parametrize("min_overlap", [0.3, 0.5, 0.9])
def test_merge_in_seqlets_filledges_min_overlap(motif_ts, motif_pattern,
	min_overlap):
	seqlets = make_seqlets(motif_ts, [(i, 16, 44, False) for i in range(15, 20)])
	merged = merge_in_seqlets_filledges(motif_pattern, seqlets, motif_ts,
		jaccard, min_overlap)
	assert len(merged.seqlets) == 20


def test_merge_in_seqlets_filledges_real(pos_patterns, signed_track_set):
	parent, child = pos_patterns
	merged = merge_in_seqlets_filledges(parent, child.seqlets,
		signed_track_set, jaccard, 0.7)

	assert len(merged.seqlets) > len(parent.seqlets)
	assert len(merged.seqlets) <= len(parent.seqlets) + len(child.seqlets)
	_assert_consistent(merged, signed_track_set)


##


def test_pattern_merge_hierarchy_node():
	node = PatternMergeHierarchyNode(pattern="p")

	assert node.pattern == "p"
	assert node.child_nodes == []
	assert node.parent_node is None
	assert node.indices_merged is None
	assert node.submat_crosscontam is None
	assert node.submat_alignersim is None


def test_pattern_merge_hierarchy_node_children():
	children = [PatternMergeHierarchyNode("a"), PatternMergeHierarchyNode("b")]
	node = PatternMergeHierarchyNode("p", child_nodes=children,
		indices_merged=(0, 1), submat_crosscontam=numpy.eye(2),
		submat_alignersim=numpy.ones((2, 2)))

	assert node.child_nodes is children
	assert node.indices_merged == (0, 1)
	assert_array_equal(node.submat_crosscontam, numpy.eye(2))


def test_pattern_merge_hierarchy_node_independent_children():
	node0 = PatternMergeHierarchyNode("a")
	node1 = PatternMergeHierarchyNode("b")
	node0.child_nodes.append("x")
	assert node1.child_nodes == []


def test_pattern_merge_hierarchy():
	nodes = [PatternMergeHierarchyNode("a"), PatternMergeHierarchyNode("b")]
	hierarchy = PatternMergeHierarchy(root_nodes=nodes)
	assert hierarchy.root_nodes is nodes


@pytest.mark.xfail(strict=True, reason="bug: PatternMergeHierarchy.add_level appends to "
	"self.levels, which is never initialized, so it raises AttributeError")
def test_pattern_merge_hierarchy_add_level():
	hierarchy = PatternMergeHierarchy(root_nodes=[])
	hierarchy.add_level(["a"])
	assert hierarchy.levels == [["a"]]


##


def test_similar_patterns_collapser_same(motif_ts):
	patterns = [make_pattern(motif_ts, [(i, 21, 39, False) for i in range(15)]),
		make_pattern(motif_ts, [(i, 21, 39, True) for i in range(15, 30)])]
	collapsed, hierarchy = SimilarPatternsCollapser(patterns, motif_ts,
		**COLLAPSE)

	assert len(collapsed) == 1
	assert len(collapsed[0].seqlets) == 30
	assert len(collapsed[0]) == 18
	_assert_consistent(collapsed[0], motif_ts)

	assert isinstance(hierarchy, PatternMergeHierarchy)
	assert len(hierarchy.root_nodes) == 1
	root = hierarchy.root_nodes[0]
	assert root.pattern is collapsed[0]
	assert root.indices_merged == (0, 1)
	assert len(root.child_nodes) == 2
	assert all(child.parent_node is root for child in root.child_nodes)
	assert root.submat_crosscontam.shape == (2, 2)
	assert root.submat_alignersim.shape == (2, 2)


def test_similar_patterns_collapser_different():
	track_set = two_motif_track_set(n=20)
	patterns = [make_pattern(track_set, [(i, 15, 35, False) for i in range(20)]),
		make_pattern(track_set, [(i, 15, 35, False) for i in range(20, 40)])]
	collapsed, hierarchy = SimilarPatternsCollapser(patterns, track_set,
		**COLLAPSE_TWO)

	assert len(collapsed) == 2
	assert _coords(collapsed[0]) == _coords(patterns[0])
	assert _coords(collapsed[1]) == _coords(patterns[1])
	assert len(hierarchy.root_nodes) == 2
	assert all(node.child_nodes == [] for node in hierarchy.root_nodes)


def test_similar_patterns_collapser_three(motif_ts):
	# Three copies of one motif collapse into a single pattern.
	patterns = [make_pattern(motif_ts, [(i, 21, 39, False)
		for i in range(j, 30, 3)]) for j in range(3)]
	collapsed, hierarchy = SimilarPatternsCollapser(patterns, motif_ts,
		**COLLAPSE)

	assert len(collapsed) == 1
	assert len(collapsed[0].seqlets) == 30


def test_similar_patterns_collapser_mixed():
	track_set = two_motif_track_set(n=20)
	patterns = [
		make_pattern(track_set, [(i, 15, 35, False) for i in range(10)]),
		make_pattern(track_set, [(i, 15, 35, False) for i in range(20, 30)]),
		make_pattern(track_set, [(i, 15, 35, True) for i in range(10, 20)]),
	]
	collapsed, hierarchy = SimilarPatternsCollapser(patterns, track_set,
		**COLLAPSE_TWO)

	assert len(collapsed) == 2
	assert len(collapsed[0].seqlets) == 20
	assert set(s.example_idx for s in collapsed[0].seqlets) == set(range(20))
	assert set(s.example_idx for s in collapsed[1].seqlets) == set(range(20, 30))


def test_similar_patterns_collapser_single(motif_pattern, motif_ts):
	collapsed, hierarchy = SimilarPatternsCollapser([motif_pattern], motif_ts,
		**COLLAPSE)

	assert len(collapsed) == 1
	assert collapsed[0] is not motif_pattern
	assert _coords(collapsed[0]) == _coords(motif_pattern)
	assert len(hierarchy.root_nodes) == 1


def test_similar_patterns_collapser_no_thresholds(motif_ts):
	patterns = [make_pattern(motif_ts, [(i, 21, 39, False) for i in range(15)]),
		make_pattern(motif_ts, [(i, 21, 39, False) for i in range(15, 30)])]
	kwargs = dict(COLLAPSE, prob_and_pertrack_sim_merge_thresholds=[])
	collapsed, hierarchy = SimilarPatternsCollapser(patterns, motif_ts,
		**kwargs)
	assert len(collapsed) == 2


def test_similar_patterns_collapser_dealbreaker(motif_ts):
	patterns = [make_pattern(motif_ts, [(i, 21, 39, False) for i in range(15)]),
		make_pattern(motif_ts, [(i, 21, 39, False) for i in range(15, 30)])]
	kwargs = dict(COLLAPSE,
		prob_and_pertrack_sim_dealbreaker_thresholds=[(1.0, 1.0)])
	collapsed, hierarchy = SimilarPatternsCollapser(patterns, motif_ts,
		**kwargs)
	assert len(collapsed) == 2


@pytest.mark.parametrize("max_seqlets_subsample", [2, 5, 14, 1000])
def test_similar_patterns_collapser_subsample(motif_ts, max_seqlets_subsample):
	patterns = [make_pattern(motif_ts, [(i, 21, 39, False) for i in range(15)]),
		make_pattern(motif_ts, [(i, 21, 39, True) for i in range(15, 30)])]
	collapsed, hierarchy = SimilarPatternsCollapser(patterns, motif_ts,
		max_seqlets_subsample=max_seqlets_subsample, **COLLAPSE)

	assert len(collapsed) == 1
	assert len(collapsed[0].seqlets) == 30


@pytest.mark.parametrize("window_size,flank_to_add", [(6, 0), (8, 2),
	(10, 5)])
def test_similar_patterns_collapser_polish(motif_ts, window_size,
	flank_to_add):
	patterns = [make_pattern(motif_ts, [(i, 16, 44, False) for i in range(15)]),
		make_pattern(motif_ts, [(i, 16, 44, False) for i in range(15, 30)])]
	kwargs = dict(COLLAPSE, window_size=window_size, flank_to_add=flank_to_add)
	collapsed, hierarchy = SimilarPatternsCollapser(patterns, motif_ts,
		**kwargs)

	assert len(collapsed) == 1
	assert len(collapsed[0]) == window_size + 2*flank_to_add


def test_similar_patterns_collapser_sorted(motif_ts):
	track_set = two_motif_track_set(n=20)
	patterns = [
		make_pattern(track_set, [(i, 15, 35, False) for i in range(20, 25)]),
		make_pattern(track_set, [(i, 15, 35, False) for i in range(8)]),
		make_pattern(track_set, [(i, 15, 35, False) for i in range(8, 20)]),
	]
	collapsed, hierarchy = SimilarPatternsCollapser(patterns, track_set,
		**COLLAPSE_TWO)

	assert [len(p.seqlets) for p in collapsed] == [20, 5]


def test_similar_patterns_collapser_does_not_modify(motif_ts):
	patterns = [make_pattern(motif_ts, [(i, 21, 39, False) for i in range(15)]),
		make_pattern(motif_ts, [(i, 21, 39, False) for i in range(15, 30)])]
	coords = [_coords(p) for p in patterns]
	SimilarPatternsCollapser(patterns, motif_ts, **COLLAPSE)

	assert [_coords(p) for p in patterns] == coords


def test_similar_patterns_collapser_real(pos_patterns, signed_track_set):
	kwargs = dict(COLLAPSE, min_num=30, flank_to_add=10, window_size=30)
	collapsed, hierarchy = SimilarPatternsCollapser(pos_patterns,
		signed_track_set, **kwargs)

	assert len(collapsed) == 2
	assert [len(p.seqlets) for p in collapsed] == [74, 26]


##


def test_detect_spurious_merging_passthrough(motif_ts):
	patterns = [make_pattern(motif_ts, [(i, 21, 39, False) for i in range(15)]),
		make_pattern(motif_ts, [(i, 21, 39, False) for i in range(15, 30)])]
	merged, hierarchy = _detect_spurious_merging(patterns, motif_ts,
		perplexity=10, min_in_subcluster=1000, n_seeds=1, **COLLAPSE)
	expected, _ = SimilarPatternsCollapser(patterns, motif_ts, **COLLAPSE)

	assert [_coords(p) for p in merged] == [_coords(p) for p in expected]
	assert isinstance(hierarchy, PatternMergeHierarchy)


def test_detect_spurious_merging_split():
	# A pattern mixing two motifs is split by subclustering.
	track_set = two_motif_track_set(n=20)
	pattern = make_pattern(track_set, [(i, 15, 35, False) for i in range(40)])
	merged, hierarchy = _detect_spurious_merging([pattern], track_set,
		perplexity=10, min_in_subcluster=10, n_seeds=2, **COLLAPSE_TWO)

	assert len(merged) == 2
	assert sorted(len(p.seqlets) for p in merged) == [20, 20]
	for p in merged:
		assert len(set(s.example_idx < 20 for s in p.seqlets)) == 1


@pytest.mark.parametrize("n_seeds", [1, 2, 3])
def test_detect_spurious_merging_n_seeds(n_seeds):
	track_set = two_motif_track_set(n=20)
	pattern = make_pattern(track_set, [(i, 15, 35, False) for i in range(40)])
	merged, hierarchy = _detect_spurious_merging([pattern], track_set,
		perplexity=10, min_in_subcluster=10, n_seeds=n_seeds, **COLLAPSE_TWO)
	assert len(merged) == 2


def test_detect_spurious_merging_computes_subpatterns():
	track_set = two_motif_track_set(n=20)
	pattern = make_pattern(track_set, [(i, 15, 35, False) for i in range(40)])
	_detect_spurious_merging([pattern], track_set, perplexity=10,
		min_in_subcluster=10, n_seeds=2, **COLLAPSE_TWO)

	assert pattern.subclusters is not None
	assert len(pattern.subcluster_to_subpattern) >= 2


def test_detect_spurious_merging_real(pos_patterns, signed_track_set):
	kwargs = dict(COLLAPSE, flank_to_add=10, window_size=30)
	merged, hierarchy = _detect_spurious_merging(pos_patterns,
		signed_track_set, perplexity=50, min_in_subcluster=50, n_seeds=2,
		**kwargs)

	assert [len(p.seqlets) for p in merged] == [74, 26]
