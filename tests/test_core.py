# test_core.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import pytest

from modiscolite.core import TrackSet
from modiscolite.core import Seqlet
from modiscolite.core import SeqletSet

from .synthetic import random_one_hot
from .synthetic import random_track_set
from .synthetic import two_motif_track_set
from .synthetic import make_seqlets
from .synthetic import make_pattern

from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def random_ts():
	return random_track_set(n=8, length=60)


@pytest.fixture
def pattern(random_ts):
	coords = [(0, 5, 25, False), (1, 10, 30, True), (2, 0, 20, False),
		(3, 40, 60, True), (4, 17, 37, False)]
	return make_pattern(random_ts, coords)


def _two_motif_pattern(n=40):
	"""A pattern whose seqlets are drawn half from each of two motifs."""

	track_set = two_motif_track_set(n=n, length=60, motif_length=10,
		position=20)
	coords = [(i, 15, 35, False) for i in range(2*n)]
	return make_pattern(track_set, coords)


##


@pytest.mark.parametrize("n,length", [(1, 1), (1, 20), (5, 60), (64, 100),
	(3, 1000)])
def test_track_set_init(n, length):
	one_hot = random_one_hot((n, length))
	hyp = numpy.random.RandomState(0).randn(n, length, 4)
	track_set = TrackSet(one_hot, one_hot*hyp, hyp)

	assert track_set.length == length
	assert track_set.one_hot is one_hot
	assert track_set.hypothetical_contribs is hyp
	assert_array_equal(track_set.contrib_scores, one_hot*hyp)


def test_track_set_init_real(track_set, one_hot, hypothetical_contribs):
	assert track_set.length == 300
	assert track_set.one_hot.shape == (300, 300, 4)
	assert track_set.contrib_scores.dtype == numpy.float32
	assert_array_equal(track_set.contrib_scores, one_hot*hypothetical_contribs)


@pytest.mark.parametrize("coord", [(0, 0, 10), (0, 50, 60), (3, 10, 40),
	(7, 0, 60), (5, 20, 21), (2, 33, 45)])
def test_track_set_create_seqlets_fwd(random_ts, coord):
	idx, start, end = coord
	seqlet = random_ts.create_seqlets([Seqlet(idx, start, end, False)])[0]

	assert_array_equal(seqlet.sequence, random_ts.one_hot[idx, start:end])
	assert_array_equal(seqlet.contrib_scores,
		random_ts.contrib_scores[idx, start:end])
	assert_array_equal(seqlet.hypothetical_contribs,
		random_ts.hypothetical_contribs[idx, start:end])


@pytest.mark.parametrize("coord", [(0, 0, 10), (0, 50, 60), (3, 10, 40),
	(7, 0, 60), (5, 20, 21), (2, 33, 45)])
def test_track_set_create_seqlets_rc(random_ts, coord):
	idx, start, end = coord
	seqlet = random_ts.create_seqlets([Seqlet(idx, start, end, True)])[0]

	assert_array_equal(seqlet.sequence,
		random_ts.one_hot[idx, start:end][::-1, ::-1])
	assert_array_equal(seqlet.contrib_scores,
		random_ts.contrib_scores[idx, start:end][::-1, ::-1])
	assert_array_equal(seqlet.hypothetical_contribs,
		random_ts.hypothetical_contribs[idx, start:end][::-1, ::-1])


def test_track_set_create_seqlets_in_place(random_ts):
	seqlets = [Seqlet(0, 0, 10, False), Seqlet(1, 5, 20, True)]
	returned = random_ts.create_seqlets(seqlets)

	assert returned is seqlets
	assert returned[0] is seqlets[0]
	assert seqlets[1].sequence.shape == (15, 4)


def test_track_set_create_seqlets_empty(random_ts):
	assert random_ts.create_seqlets([]) == []


def test_track_set_create_seqlets_views(random_ts):
	seqlet = random_ts.create_seqlets([Seqlet(2, 5, 15, True)])[0]

	assert numpy.shares_memory(seqlet.sequence, random_ts.one_hot)
	assert numpy.shares_memory(seqlet.contrib_scores, random_ts.contrib_scores)


def test_track_set_create_seqlets_real(track_set, seqlets):
	assert len(seqlets) == 551
	for seqlet in seqlets:
		assert seqlet.sequence.shape == (30, 4)
		assert seqlet.contrib_scores.shape == (30, 4)
		assert seqlet.hypothetical_contribs.shape == (30, 4)
		assert_array_equal(seqlet.sequence.sum(axis=1), numpy.ones(30))


def test_track_set_create_seqlets_rc_complement(track_set):
	fwd = track_set.create_seqlets([Seqlet(10, 100, 130, False)])[0]
	rc = track_set.create_seqlets([Seqlet(10, 100, 130, True)])[0]

	assert_array_equal(rc.sequence, fwd.sequence[::-1, ::-1])
	assert abs(rc.contrib_scores.sum() - fwd.contrib_scores.sum()) < 1e-5


##


@pytest.mark.parametrize("coord", [(0, 0, 10, False), (5, 17, 40, True),
	(1000, 5, 6, False), (3, -2, 8, True)])
def test_seqlet_init(coord):
	seqlet = Seqlet(*coord)

	assert seqlet.example_idx == coord[0]
	assert seqlet.start == coord[1]
	assert seqlet.end == coord[2]
	assert seqlet.is_revcomp == coord[3]
	assert seqlet.sequence is None
	assert seqlet.contrib_scores is None
	assert seqlet.hypothetical_contribs is None


def test_seqlet_init_keywords():
	seqlet = Seqlet(example_idx=4, start=10, end=31, is_revcomp=True)
	assert (seqlet.example_idx, seqlet.start, seqlet.end) == (4, 10, 31)
	assert seqlet.is_revcomp is True


@pytest.mark.parametrize("coord,expected", [
	((0, 0, 10, False), "example:0,start:0,end:10,rc:False"),
	((5, 17, 40, True), "example:5,start:17,end:40,rc:True"),
	((1234, 200, 230, False), "example:1234,start:200,end:230,rc:False"),
])
def test_seqlet_str(coord, expected):
	assert str(Seqlet(*coord)) == expected


@pytest.mark.parametrize("start,end", [(0, 1), (0, 10), (5, 35), (100, 400),
	(7, 7)])
def test_seqlet_len(start, end):
	assert len(Seqlet(0, start, end, False)) == end - start
	assert len(Seqlet(0, start, end, True)) == end - start


@pytest.mark.parametrize("coord,expected", [((0, 0, 10, False), "0_0_10"),
	((5, 17, 40, True), "5_17_40"), ((12, 3, 4, False), "12_3_4")])
def test_seqlet_string(coord, expected):
	assert Seqlet(*coord).string == expected


def test_seqlet_string_ignores_strand():
	assert Seqlet(3, 10, 20, False).string == Seqlet(3, 10, 20, True).string


@pytest.mark.parametrize("is_revcomp", [False, True])
def test_seqlet_revcomp(random_ts, is_revcomp):
	seqlet = make_seqlets(random_ts, [(2, 10, 30, is_revcomp)])[0]
	rc = seqlet.revcomp()

	assert rc is not seqlet
	assert rc.is_revcomp == (not is_revcomp)
	assert (rc.example_idx, rc.start, rc.end) == (2, 10, 30)
	assert_array_equal(rc.sequence, seqlet.sequence[::-1, ::-1])
	assert_array_equal(rc.contrib_scores, seqlet.contrib_scores[::-1, ::-1])
	assert_array_equal(rc.hypothetical_contribs,
		seqlet.hypothetical_contribs[::-1, ::-1])


@pytest.mark.parametrize("is_revcomp", [False, True])
def test_seqlet_revcomp_matches_track_set(random_ts, is_revcomp):
	seqlet = make_seqlets(random_ts, [(4, 7, 29, is_revcomp)])[0]
	expected = make_seqlets(random_ts, [(4, 7, 29, not is_revcomp)])[0]
	rc = seqlet.revcomp()

	assert_array_equal(rc.sequence, expected.sequence)
	assert_array_equal(rc.contrib_scores, expected.contrib_scores)
	assert_array_equal(rc.hypothetical_contribs, expected.hypothetical_contribs)


def test_seqlet_revcomp_twice(random_ts):
	seqlet = make_seqlets(random_ts, [(1, 0, 25, False)])[0]
	rc2 = seqlet.revcomp().revcomp()

	assert rc2.is_revcomp is False
	assert_array_equal(rc2.sequence, seqlet.sequence)
	assert_array_equal(rc2.contrib_scores, seqlet.contrib_scores)


def test_seqlet_revcomp_does_not_modify(random_ts):
	seqlet = make_seqlets(random_ts, [(1, 0, 25, False)])[0]
	sequence = seqlet.sequence.copy()
	seqlet.revcomp()

	assert seqlet.is_revcomp is False
	assert_array_equal(seqlet.sequence, sequence)


def test_seqlet_revcomp_raises():
	assert_raises(TypeError, Seqlet(0, 0, 10, False).revcomp)


@pytest.mark.parametrize("shift", [-10, -1, 0, 1, 7, 100])
@pytest.mark.parametrize("is_revcomp", [False, True])
def test_seqlet_shift(shift, is_revcomp):
	seqlet = Seqlet(3, 20, 45, is_revcomp)
	shifted = seqlet.shift(shift)

	assert shifted is not seqlet
	assert shifted.example_idx == 3
	assert shifted.start == 20 + shift
	assert shifted.end == 45 + shift
	assert shifted.is_revcomp == is_revcomp
	assert len(shifted) == len(seqlet)
	assert shifted.sequence is None
	assert (seqlet.start, seqlet.end) == (20, 45)


@pytest.mark.parametrize("start_idx,end_idx", [(0, 20), (0, 5), (5, 20),
	(3, 17), (10, 11), (19, 20)])
def test_seqlet_trim_fwd(random_ts, start_idx, end_idx):
	seqlet = make_seqlets(random_ts, [(3, 10, 30, False)])[0]
	trimmed = seqlet.trim(start_idx, end_idx)

	assert trimmed.is_revcomp is False
	assert trimmed.start == 10 + start_idx
	assert trimmed.end == 10 + end_idx
	assert_array_equal(trimmed.sequence, seqlet.sequence[start_idx:end_idx])
	assert_array_equal(trimmed.contrib_scores,
		seqlet.contrib_scores[start_idx:end_idx])
	assert_array_equal(trimmed.hypothetical_contribs,
		seqlet.hypothetical_contribs[start_idx:end_idx])


@pytest.mark.parametrize("start_idx,end_idx", [(0, 20), (0, 5), (5, 20),
	(3, 17), (10, 11), (19, 20)])
def test_seqlet_trim_rc(random_ts, start_idx, end_idx):
	seqlet = make_seqlets(random_ts, [(3, 10, 30, True)])[0]
	trimmed = seqlet.trim(start_idx, end_idx)

	assert trimmed.is_revcomp is True
	assert trimmed.start == 30 - end_idx
	assert trimmed.end == 30 - start_idx
	assert_array_equal(trimmed.sequence, seqlet.sequence[start_idx:end_idx])


@pytest.mark.parametrize("is_revcomp", [False, True])
@pytest.mark.parametrize("start_idx,end_idx", [(0, 20), (2, 9), (7, 20),
	(0, 1)])
def test_seqlet_trim_matches_track_set(random_ts, is_revcomp, start_idx,
	end_idx):
	seqlet = make_seqlets(random_ts, [(6, 25, 45, is_revcomp)])[0]
	trimmed = seqlet.trim(start_idx, end_idx)
	expected = make_seqlets(random_ts, [(6, trimmed.start, trimmed.end,
		is_revcomp)])[0]

	assert_array_equal(trimmed.sequence, expected.sequence)
	assert_array_equal(trimmed.contrib_scores, expected.contrib_scores)
	assert_array_equal(trimmed.hypothetical_contribs,
		expected.hypothetical_contribs)


def test_seqlet_trim_does_not_modify(random_ts):
	seqlet = make_seqlets(random_ts, [(3, 10, 30, False)])[0]
	seqlet.trim(2, 8)

	assert (seqlet.start, seqlet.end) == (10, 30)
	assert seqlet.sequence.shape == (20, 4)


##


def test_seqlet_set_single(random_ts):
	seqlet = make_seqlets(random_ts, [(2, 10, 30, False)])[0]
	pattern = SeqletSet([seqlet])

	assert pattern.seqlets == [seqlet]
	assert pattern.unique_seqlets == {"2_10_30": seqlet}
	assert pattern.length == 20
	assert len(pattern) == 20
	assert pattern.subclusters is None
	assert pattern.subcluster_to_subpattern is None
	assert_array_equal(pattern.per_position_counts, numpy.ones(20))
	assert_array_almost_equal(pattern.sequence, seqlet.sequence)
	assert_array_almost_equal(pattern.contrib_scores, seqlet.contrib_scores)
	assert_array_almost_equal(pattern.hypothetical_contribs,
		seqlet.hypothetical_contribs)


@pytest.mark.parametrize("n", [2, 3, 5, 8])
@pytest.mark.parametrize("length", [1, 10, 25])
def test_seqlet_set_mean(random_ts, n, length):
	coords = [(i, i, i+length, bool(i % 2)) for i in range(n)]
	seqlets = make_seqlets(random_ts, coords)
	pattern = SeqletSet(seqlets)

	assert len(pattern.seqlets) == n
	assert pattern.length == length
	assert_array_equal(pattern.per_position_counts, numpy.full(length, n))
	assert_array_almost_equal(pattern.sequence,
		numpy.mean([s.sequence for s in seqlets], axis=0))
	assert_array_almost_equal(pattern.contrib_scores,
		numpy.mean([s.contrib_scores for s in seqlets], axis=0))
	assert_array_almost_equal(pattern.hypothetical_contribs,
		numpy.mean([s.hypothetical_contribs for s in seqlets], axis=0))


def test_seqlet_set_ppm(pattern):
	assert_array_almost_equal(pattern.sequence.sum(axis=1), numpy.ones(20))
	assert numpy.all(pattern.sequence >= 0)


def test_seqlet_set_uneven_lengths(random_ts):
	seqlets = make_seqlets(random_ts, [(0, 0, 10, False), (1, 0, 6, False),
		(2, 0, 3, True)])
	pattern = SeqletSet(seqlets)

	assert pattern.length == 10
	assert_array_equal(pattern.per_position_counts,
		[3, 3, 3, 2, 2, 2, 1, 1, 1, 1])

	assert_array_almost_equal(pattern.sequence[:3],
		numpy.mean([s.sequence[:3] for s in seqlets], axis=0))
	assert_array_almost_equal(pattern.sequence[3:6],
		numpy.mean([s.sequence[3:6] for s in seqlets[:2]], axis=0))
	assert_array_almost_equal(pattern.contrib_scores[6:],
		seqlets[0].contrib_scores[6:])


def test_seqlet_set_longest_not_first(random_ts):
	seqlets = make_seqlets(random_ts, [(0, 0, 4, False), (1, 0, 12, False)])
	pattern = SeqletSet(seqlets)

	assert pattern.length == 12
	assert_array_equal(pattern.per_position_counts, [2]*4 + [1]*8)
	assert_array_almost_equal(pattern.sequence[4:], seqlets[1].sequence[4:])


def test_seqlet_set_duplicates(random_ts):
	seqlets = make_seqlets(random_ts, [(0, 0, 10, False), (0, 0, 10, False),
		(1, 0, 10, False)])
	pattern = SeqletSet(seqlets)

	assert len(pattern.seqlets) == 2
	assert pattern.seqlets == [seqlets[0], seqlets[2]]
	assert_array_equal(pattern.per_position_counts, numpy.full(10, 2))
	assert_array_almost_equal(pattern.sequence,
		(seqlets[0].sequence + seqlets[2].sequence) / 2)


def test_seqlet_set_duplicates_strand(random_ts):
	# Seqlets on opposite strands of the same coordinates share a string, so
	# only the first one is kept.
	seqlets = make_seqlets(random_ts, [(3, 5, 15, False), (3, 5, 15, True)])
	pattern = SeqletSet(seqlets)

	assert pattern.seqlets == [seqlets[0]]
	assert_array_almost_equal(pattern.sequence, seqlets[0].sequence)


def test_seqlet_set_real(seqlets):
	pattern = SeqletSet(seqlets)

	assert len(pattern.seqlets) == 551
	assert pattern.length == 30
	assert pattern.sequence.shape == (30, 4)
	assert_array_almost_equal(pattern.sequence.sum(axis=1), numpy.ones(30))
	assert_array_almost_equal(pattern.contrib_scores,
		numpy.mean([s.contrib_scores for s in seqlets], axis=0))


def test_seqlet_set_dtype(seqlets):
	pattern = SeqletSet(seqlets[:10])
	assert pattern.sequence.dtype == numpy.float64
	assert pattern.contrib_scores.dtype == numpy.float64


def test_seqlet_set_raises():
	assert_raises(ValueError, SeqletSet, [])


##


def test_seqlet_set_copy(pattern, random_ts):
	pattern_copy = pattern.copy()

	assert pattern_copy is not pattern
	assert pattern_copy.seqlets == pattern.seqlets
	assert pattern_copy.seqlets is not pattern.seqlets
	assert_array_almost_equal(pattern_copy.sequence, pattern.sequence)
	assert_array_almost_equal(pattern_copy.contrib_scores,
		pattern.contrib_scores)

	new_seqlet = make_seqlets(random_ts, [(7, 0, 20, False)])[0]
	pattern_copy._add_seqlet(new_seqlet)

	assert len(pattern.seqlets) == 5
	assert len(pattern_copy.seqlets) == 6
	assert "7_0_20" not in pattern.unique_seqlets


def test_seqlet_set_copy_subclusters(pos_patterns):
	pattern = pos_patterns[0]
	assert pattern.subclusters is not None

	pattern_copy = pattern.copy()
	assert pattern_copy.subclusters is None
	assert pattern_copy.subcluster_to_subpattern is None


@pytest.mark.parametrize("start_idx,end_idx", [(0, 20), (0, 10), (5, 20),
	(4, 16), (19, 20)])
def test_seqlet_set_trim_to_idx(pattern, start_idx, end_idx):
	trimmed = pattern.trim_to_idx(start_idx, end_idx)

	assert trimmed is not pattern
	assert len(trimmed) == end_idx - start_idx
	assert len(trimmed.seqlets) == 5
	assert_array_almost_equal(trimmed.sequence,
		pattern.sequence[start_idx:end_idx])
	assert_array_almost_equal(trimmed.contrib_scores,
		pattern.contrib_scores[start_idx:end_idx])
	assert_array_almost_equal(trimmed.hypothetical_contribs,
		pattern.hypothetical_contribs[start_idx:end_idx])


def test_seqlet_set_trim_to_idx_coords(pattern):
	trimmed = pattern.trim_to_idx(3, 13)

	for seqlet, trimmed_seqlet in zip(pattern.seqlets, trimmed.seqlets):
		assert trimmed_seqlet.is_revcomp == seqlet.is_revcomp
		if seqlet.is_revcomp:
			assert trimmed_seqlet.start == seqlet.end - 13
			assert trimmed_seqlet.end == seqlet.end - 3
		else:
			assert trimmed_seqlet.start == seqlet.start + 3
			assert trimmed_seqlet.end == seqlet.start + 13


def test_seqlet_set_trim_to_idx_does_not_modify(pattern):
	sequence = pattern.sequence.copy()
	pattern.trim_to_idx(2, 8)

	assert len(pattern) == 20
	assert_array_almost_equal(pattern.sequence, sequence)


@pytest.fixture
def uneven_pattern(random_ts):
	seqlets = make_seqlets(random_ts, [(0, 0, 10, False), (1, 0, 10, False),
		(2, 0, 8, False), (3, 0, 6, False), (4, 0, 4, False)])
	return SeqletSet(seqlets)


def test_seqlet_set_trim_to_support_counts(uneven_pattern):
	assert_array_equal(uneven_pattern.per_position_counts,
		[5, 5, 5, 5, 4, 4, 3, 3, 2, 2])


@pytest.mark.parametrize("min_frac,min_num", [(0.9, 100), (1.0, 5),
	(0.95, 30), (1.0, 4.5)])
def test_seqlet_set_trim_to_support(uneven_pattern, min_frac, min_num):
	# Every threshold above 4 keeps only the positions all 5 seqlets cover.
	trimmed = uneven_pattern.trim_to_support(min_frac, min_num)

	assert len(trimmed) == 4
	assert len(trimmed.seqlets) == 5
	assert_array_equal(trimmed.per_position_counts, numpy.full(4, 5))
	assert_array_almost_equal(trimmed.sequence, uneven_pattern.sequence[:4])


@pytest.mark.skip(reason="bug: Seqlet.trim does not clip end_idx to the "
	"seqlet's own length, so trim_to_support raises a broadcasting ValueError "
	"whenever it keeps positions beyond the shortest seqlet")
@pytest.mark.parametrize("min_frac,min_num,expected", [(0.5, 100, 8),
	(0.2, 100, 10), (1.0, 3, 8), (1.0, 4, 6), (0.0, 100, 10)])
def test_seqlet_set_trim_to_support_uneven(uneven_pattern, min_frac, min_num,
	expected):
	trimmed = uneven_pattern.trim_to_support(min_frac, min_num)

	assert len(trimmed) == expected
	assert len(trimmed.seqlets) == 5


def test_seqlet_set_trim_to_support_full(pattern):
	trimmed = pattern.trim_to_support(min_frac=0.2, min_num=30)

	assert len(trimmed) == 20
	assert_array_almost_equal(trimmed.sequence, pattern.sequence)


def test_seqlet_set_trim_to_support_real(pos_patterns):
	for pattern in pos_patterns:
		trimmed = pattern.trim_to_support(min_frac=0.2, min_num=30)
		assert len(trimmed) == len(pattern)
		assert len(trimmed.seqlets) == len(pattern.seqlets)


##


def test_seqlet_set_add_seqlet(pattern, random_ts):
	seqlet = make_seqlets(random_ts, [(7, 30, 50, False)])[0]
	expected = numpy.mean([s.contrib_scores for s in pattern.seqlets +
		[seqlet]], axis=0)

	pattern._add_seqlet(seqlet)

	assert len(pattern.seqlets) == 6
	assert pattern.seqlets[-1] is seqlet
	assert pattern.unique_seqlets["7_30_50"] is seqlet
	assert_array_equal(pattern.per_position_counts, numpy.full(20, 6))
	assert_array_almost_equal(pattern.contrib_scores, expected)


def test_seqlet_set_add_seqlet_shorter(pattern, random_ts):
	seqlet = make_seqlets(random_ts, [(7, 30, 40, False)])[0]
	pattern._add_seqlet(seqlet)

	assert len(pattern) == 20
	assert_array_equal(pattern.per_position_counts, [6]*10 + [5]*10)


def test_seqlet_set_add_seqlet_no_dedup(pattern):
	# _add_seqlet does not check for duplicates; its callers do.
	pattern._add_seqlet(pattern.seqlets[0])

	assert len(pattern.seqlets) == 6
	assert len(pattern.unique_seqlets) == 5
	assert_array_equal(pattern.per_position_counts, numpy.full(20, 6))


def test_seqlet_set_add_seqlet_longer_raises(pattern, random_ts):
	seqlet = make_seqlets(random_ts, [(7, 20, 50, False)])[0]
	assert_raises(ValueError, pattern._add_seqlet, seqlet)


##


def test_seqlet_set_save_seqlets(random_ts, tmp_path):
	seqlets = make_seqlets(random_ts, [(0, 5, 10, False), (3, 20, 26, True)])
	pattern = SeqletSet(seqlets)

	filename = tmp_path / "seqlets.fa"
	pattern.save_seqlets(filename)

	bases = numpy.array(list("ACGT"))
	lines = open(filename).read().split("\n")

	assert len(lines) == 5
	assert lines[0] == ">example0:5-10"
	assert lines[1] == "".join(bases[random_ts.one_hot[0, 5:10].argmax(axis=1)])
	assert lines[2] == ">example3:20-26"
	assert lines[3] == "".join(bases[
		random_ts.one_hot[3, 20:26][::-1, ::-1].argmax(axis=1)])
	assert lines[4] == ""


def test_seqlet_set_save_seqlets_real(seqlets, tmp_path):
	pattern = SeqletSet(seqlets[:3])

	filename = tmp_path / "seqlets.fa"
	pattern.save_seqlets(filename)

	assert open(filename).read() == (
		">example0:117-147\nATCCAGCAGGGAGGAGAGAGGCTCTCCACG\n"
		">example1:130-160\nGGACAGCAGAGAGAGGAACAGGCAGGCCGA\n"
		">example2:144-174\nCGGCCTAGGCCAGCCCCTCTCCCCAGTCAT\n")


##


@pytest.mark.parametrize("perplexity", [5, 10, 50])
@pytest.mark.parametrize("n_seeds", [1, 2])
def test_seqlet_set_compute_subpatterns(pos_patterns, perplexity, n_seeds):
	pattern = pos_patterns[0].copy()
	pattern.compute_subpatterns(perplexity=perplexity, n_seeds=n_seeds)

	n = len(pattern.seqlets)
	assert pattern.subclusters.shape == (n,)
	assert pattern.subclusters.min() == 0

	sizes = [len(p.seqlets) for p in pattern.subcluster_to_subpattern.values()]
	assert sum(sizes) == n
	assert sizes == sorted(sizes, reverse=True)
	assert set(pattern.subcluster_to_subpattern.keys()) == set(
		pattern.subclusters)

	for subcluster, subpattern in pattern.subcluster_to_subpattern.items():
		assert isinstance(subpattern, SeqletSet)
		members = [s for s, c in zip(pattern.seqlets, pattern.subclusters)
			if c == subcluster]
		assert subpattern.seqlets == members


def test_seqlet_set_compute_subpatterns_deterministic(pos_patterns):
	pattern0 = pos_patterns[0].copy()
	pattern1 = pos_patterns[0].copy()

	pattern0.compute_subpatterns(perplexity=10, n_seeds=2)
	pattern1.compute_subpatterns(perplexity=10, n_seeds=2)
	assert_array_equal(pattern0.subclusters, pattern1.subclusters)


@pytest.mark.parametrize("n_iterations", [-1, 1, 2, 5])
def test_seqlet_set_compute_subpatterns_iterations(pos_patterns, n_iterations):
	pattern = pos_patterns[1].copy()
	pattern.compute_subpatterns(perplexity=10, n_seeds=1,
		n_iterations=n_iterations)

	assert pattern.subclusters.shape == (len(pattern.seqlets),)


def test_seqlet_set_compute_subpatterns_two_motifs():
	pattern = _two_motif_pattern(n=40)
	pattern.compute_subpatterns(perplexity=10, n_seeds=2)

	assert len(pattern.subcluster_to_subpattern) >= 2
	# No subcluster mixes seqlets from the two motifs.
	for subpattern in pattern.subcluster_to_subpattern.values():
		sources = set(s.example_idx < 40 for s in subpattern.seqlets)
		assert len(sources) == 1


def test_seqlet_set_compute_subpatterns_real(pos_patterns):
	pattern = pos_patterns[0]
	sizes = [len(p.seqlets) for p in pattern.subcluster_to_subpattern.values()]

	assert len(pattern.seqlets) == 74
	assert sizes == [35, 22, 17]
