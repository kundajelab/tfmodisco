# test_extract_seqlets.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import pytest

from modiscolite.extract_seqlets import _bin_mode
from modiscolite.extract_seqlets import _laplacian_null
from modiscolite.extract_seqlets import _iterative_extract_seqlets
from modiscolite.extract_seqlets import _smooth_and_split
from modiscolite.extract_seqlets import _isotonic_thresholds
from modiscolite.extract_seqlets import _refine_thresholds
from modiscolite.extract_seqlets import extract_seqlets

from modiscolite.core import Seqlet

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


KWARGS = dict(target_fdr=0.05, min_passing_windows_frac=0.03,
	max_passing_windows_frac=0.2, weak_threshold_for_counting_sign=0.8)


def _coords(seqlets):
	return [(s.example_idx, s.start, s.end, s.is_revcomp) for s in seqlets]


def _track(d, peaks, n=1):
	"""Return an (n, d) track of -inf with the given {position: value} peaks."""

	track = numpy.full((n, d), -numpy.inf)
	for (i, position), value in peaks.items():
		track[i, position] = value

	return track


@pytest.fixture
def attributions(contrib_scores):
	return contrib_scores.sum(axis=2)


@pytest.fixture
def smoothed(attributions):
	return _smooth_and_split(attributions, 20)[2]


##


@pytest.mark.parametrize("bins", [10, 100, 1000])
@pytest.mark.parametrize("loc", [-3.0, 0.0, 2.5])
def test_bin_mode(bins, loc):
	values = numpy.random.RandomState(0).laplace(loc, 0.5, size=50000)
	l_edge, r_edge, top = _bin_mode(values, bins=bins)

	hist, edges = numpy.histogram(values, bins=bins)
	peak = numpy.argmax(hist)

	assert l_edge == edges[peak]
	assert r_edge == edges[peak+1]
	assert abs((r_edge - l_edge) - (values.max() - values.min()) / bins) < 1e-9
	assert numpy.all(top > l_edge) and numpy.all(top < r_edge)
	assert_array_equal(top, values[(values > l_edge) & (values < r_edge)])


@pytest.mark.parametrize("loc", [-3.0, 0.0, 2.5])
def test_bin_mode_location(loc):
	values = numpy.random.RandomState(0).laplace(loc, 0.5, size=100000)
	l_edge, r_edge, top = _bin_mode(values, bins=100)
	assert abs((l_edge + r_edge) / 2 - loc) < 0.2


def test_bin_mode_spike():
	values = numpy.concatenate([numpy.random.RandomState(0).uniform(-10, 10,
		size=1000), numpy.full(500, 3.3)])
	l_edge, r_edge, top = _bin_mode(values, bins=50)

	assert l_edge < 3.3 < r_edge
	assert numpy.sum(top == 3.3) == 500


def test_bin_mode_two_level():
	# The second pass on the first pass's values narrows the estimate.
	values = numpy.random.RandomState(0).laplace(0.7, 0.3, size=200000)
	l0, r0, top = _bin_mode(values)
	l1, r1, _ = _bin_mode(top)

	assert l0 <= l1 < r1 <= r0
	assert (r1 - l1) < (r0 - l0)


def test_bin_mode_default_bins():
	values = numpy.random.RandomState(0).randn(10000)
	l_edge, r_edge, _ = _bin_mode(values)
	assert abs((r_edge - l_edge) - (values.max() - values.min()) / 1000) < 1e-9


##


@pytest.mark.parametrize("num_to_samp", [10, 100, 1000, 10000])
def test_laplacian_null(smoothed, num_to_samp):
	pos, neg = _laplacian_null(smoothed, 20, num_to_samp)

	assert len(pos) + len(neg) == num_to_samp
	assert numpy.all(pos >= 0)
	assert numpy.all(neg < 0)


def test_laplacian_null_deterministic(smoothed):
	pos0, neg0 = _laplacian_null(smoothed, 20, 1000)
	pos1, neg1 = _laplacian_null(smoothed, 20, 1000)

	assert_array_equal(pos0, pos1)
	assert_array_equal(neg0, neg1)


@pytest.mark.parametrize("random_seed", [0, 1, 99])
def test_laplacian_null_seed(smoothed, random_seed):
	pos0, neg0 = _laplacian_null(smoothed, 20, 1000)
	pos1, neg1 = _laplacian_null(smoothed, 20, 1000, random_seed=random_seed)

	assert len(pos1) + len(neg1) == 1000
	assert not numpy.array_equal(numpy.sort(numpy.concatenate([pos0, neg0])),
		numpy.sort(numpy.concatenate([pos1, neg1])))


@pytest.mark.parametrize("window_size", [1, 10, 50])
def test_laplacian_null_window_size_unused(smoothed, window_size):
	pos0, neg0 = _laplacian_null(smoothed, 20, 500)
	pos1, neg1 = _laplacian_null(smoothed, window_size, 500)

	assert_array_equal(pos0, pos1)
	assert_array_equal(neg0, neg1)


def test_laplacian_null_symmetric():
	# A symmetric Laplace track gives a symmetric null centered at zero.
	track = numpy.random.RandomState(0).laplace(0, 1, size=(200, 500))
	pos, neg = _laplacian_null(track, 20, 10000)

	assert abs(len(pos) / 10000 - 0.5) < 0.03
	assert abs(numpy.median(pos) + numpy.median(neg)) < 0.1


def test_laplacian_null_scale():
	track = numpy.random.RandomState(0).laplace(0, 1, size=(100, 500))
	pos0, neg0 = _laplacian_null(track, 20, 5000)
	pos1, neg1 = _laplacian_null(track * 10, 20, 5000)

	assert abs(numpy.median(pos1) / numpy.median(pos0) - 10) < 1.0


def test_laplacian_null_real(smoothed):
	pos, neg = _laplacian_null(smoothed, 20, 10000)

	assert len(pos) == 5822
	assert len(neg) == 4178
	assert_array_almost_equal(pos[:4], [0.2991, 0.4741, 0.4988, 0.2131], 4)
	assert_array_almost_equal(neg[:4], [-0.0656, -0.4195, -0.2515, -0.4306], 4)


##


def test_iterative_extract_seqlets_values():
	track = _track(100, {(0, 30): 5.0, (0, 70): 3.0})
	seqlets = _iterative_extract_seqlets(track, window_size=10, flank=5,
		suppress=10)

	assert _coords(seqlets) == [(0, 25, 45, False), (0, 65, 85, False)]
	assert all(isinstance(s, Seqlet) for s in seqlets)


def test_iterative_extract_seqlets_order():
	track = _track(200, {(0, 30): 1.0, (0, 100): 3.0, (0, 150): 2.0})
	seqlets = _iterative_extract_seqlets(track, window_size=10, flank=5,
		suppress=10)

	assert [s.start for s in seqlets] == [95, 145, 25]


@pytest.mark.parametrize("gap,expected", [(10, 1), (11, 2), (5, 1), (30, 2)])
def test_iterative_extract_seqlets_suppress(gap, expected):
	track = _track(100, {(0, 40): 2.0, (0, 40+gap): 1.0})
	seqlets = _iterative_extract_seqlets(track, window_size=5, flank=2,
		suppress=10)

	assert len(seqlets) == expected
	assert seqlets[0].start == 38


@pytest.mark.parametrize("position,expected", [(0, 0), (4, 0), (5, 1),
	(94, 1), (95, 0), (99, 0)])
def test_iterative_extract_seqlets_edges(position, expected):
	track = _track(100, {(0, position): 1.0})
	seqlets = _iterative_extract_seqlets(track, window_size=10, flank=5,
		suppress=10)
	assert len(seqlets) == expected


def test_iterative_extract_seqlets_edge_suppresses():
	# A peak too close to the edge is dropped but still suppresses neighbors.
	track = _track(100, {(0, 2): 5.0, (0, 9): 1.0})
	seqlets = _iterative_extract_seqlets(track, window_size=10, flank=5,
		suppress=10)
	assert seqlets == []


def test_iterative_extract_seqlets_examples():
	track = _track(100, {(0, 50): 1.0, (1, 20): 2.0, (1, 80): 1.5,
		(3, 60): 0.5}, n=4)
	seqlets = _iterative_extract_seqlets(track, window_size=6, flank=3,
		suppress=8)

	assert _coords(seqlets) == [(0, 47, 59, False), (1, 17, 29, False),
		(1, 77, 89, False), (3, 57, 69, False)]


def test_iterative_extract_seqlets_empty():
	track = numpy.full((3, 50), -numpy.inf)
	assert _iterative_extract_seqlets(track, 5, 2, 5) == []


def test_iterative_extract_seqlets_in_place():
	track = _track(100, {(0, 30): 5.0})
	_iterative_extract_seqlets(track, window_size=10, flank=5, suppress=10)
	assert numpy.all(track == -numpy.inf)


@pytest.mark.parametrize("suppress", [3, 7, 15])
def test_iterative_extract_seqlets_dense(suppress):
	track = numpy.random.RandomState(suppress).uniform(0, 1, size=(2, 200))
	seqlets = _iterative_extract_seqlets(track, window_size=4, flank=2,
		suppress=suppress)

	for i in range(2):
		starts = sorted(s.start for s in seqlets if s.example_idx == i)
		assert numpy.all(numpy.diff(starts) > suppress)


@pytest.mark.parametrize("window_size,flank", [(1, 0), (10, 0), (5, 3),
	(20, 5)])
def test_iterative_extract_seqlets_length(window_size, flank):
	track = numpy.random.RandomState(0).uniform(0, 1, size=(3, 150))
	seqlets = _iterative_extract_seqlets(track, window_size, flank, 10)

	assert len(seqlets) > 0
	assert all(len(s) == window_size + 2*flank for s in seqlets)
	assert all(s.start >= 0 for s in seqlets)


##


@pytest.mark.parametrize("n,length", [(1, 10), (5, 50), (20, 300), (3, 1000)])
@pytest.mark.parametrize("window_size", [1, 5, 10])
def test_smooth_and_split(n, length, window_size):
	tracks = numpy.random.RandomState(n).randn(n, length)
	pos, neg, smoothed = _smooth_and_split(tracks, window_size)

	assert smoothed.shape == (n, length - window_size + 1)
	for i in range(n):
		assert_array_almost_equal(smoothed[i], numpy.convolve(tracks[i],
			numpy.ones(window_size), 'valid'))

	assert len(pos) + len(neg) == smoothed.size
	assert numpy.all(pos >= 0)
	assert numpy.all(neg < 0)
	assert numpy.all(numpy.diff(pos) >= 0)
	assert numpy.all(numpy.diff(neg) <= 0)


def test_smooth_and_split_values():
	tracks = numpy.array([[1.0, -2.0, 3.0, -4.0, 0.0]])
	pos, neg, smoothed = _smooth_and_split(tracks, 2)

	assert_array_almost_equal(smoothed, [[-1.0, 1.0, -1.0, -4.0]])
	assert_array_almost_equal(pos, [1.0])
	assert_array_almost_equal(neg, [-1.0, -1.0, -4.0])


def test_smooth_and_split_zero():
	# Zero counts as positive.
	pos, neg, smoothed = _smooth_and_split(numpy.zeros((2, 5)), 3)
	assert_array_equal(pos, numpy.zeros(6))
	assert len(neg) == 0


@pytest.mark.parametrize("subsample_cap", [10, 100, 1000])
def test_smooth_and_split_subsample(subsample_cap):
	tracks = numpy.random.RandomState(0).randn(10, 200)
	pos, neg, smoothed = _smooth_and_split(tracks, 5,
		subsample_cap=subsample_cap)

	assert smoothed.shape == (10, 196)
	assert len(pos) + len(neg) == subsample_cap
	assert numpy.isin(pos, smoothed).all()
	assert numpy.isin(neg, smoothed).all()


def test_smooth_and_split_subsample_deterministic():
	tracks = numpy.random.RandomState(0).randn(10, 200)
	pos0, neg0, _ = _smooth_and_split(tracks, 5, subsample_cap=500)
	pos1, neg1, _ = _smooth_and_split(tracks, 5, subsample_cap=500)

	assert_array_equal(pos0, pos1)
	assert_array_equal(neg0, neg1)


def test_smooth_and_split_does_not_modify():
	tracks = numpy.random.RandomState(0).randn(4, 30)
	tracks0 = tracks.copy()
	_smooth_and_split(tracks, 5)
	assert_array_equal(tracks, tracks0)


def test_smooth_and_split_real(attributions):
	pos, neg, smoothed = _smooth_and_split(attributions, 20)

	assert smoothed.shape == (300, 281)
	assert len(pos) == 49556
	assert len(neg) == 34744
	assert_array_almost_equal([pos[-1], neg[-1]], [14.3691, -2.3010], 4)


##


@pytest.mark.parametrize("target_fdr", [0.01, 0.05, 0.2, 0.5])
def test_isotonic_thresholds_increasing(target_fdr):
	rng = numpy.random.RandomState(0)
	values = numpy.sort(numpy.concatenate([rng.exponential(1, 5000),
		rng.exponential(1, 500) + 6]))
	null_values = rng.exponential(1, 5000)

	threshold = _isotonic_thresholds(values, null_values, increasing=True,
		target_fdr=target_fdr)

	assert threshold in values
	assert threshold > 2


@pytest.mark.parametrize("target_fdr", [0.01, 0.05, 0.2, 0.5])
def test_isotonic_thresholds_decreasing(target_fdr):
	rng = numpy.random.RandomState(0)
	values = numpy.sort(-numpy.concatenate([rng.exponential(1, 5000),
		rng.exponential(1, 500) + 6]))[::-1]
	null_values = -rng.exponential(1, 5000)

	threshold = _isotonic_thresholds(values, null_values, increasing=False,
		target_fdr=target_fdr)

	assert threshold in values
	assert threshold < -2


def test_isotonic_thresholds_fdr_monotonic():
	rng = numpy.random.RandomState(0)
	values = numpy.sort(numpy.concatenate([rng.exponential(1, 5000),
		rng.exponential(2, 2000) + 2]))
	null_values = rng.exponential(1, 5000)

	thresholds = [_isotonic_thresholds(values, null_values, True, fdr)
		for fdr in [0.4, 0.2, 0.1, 0.05, 0.01]]
	assert numpy.all(numpy.diff(thresholds) >= 0)


@pytest.mark.parametrize("min_frac_neg", [0.5, 0.95, 1.0])
def test_isotonic_thresholds_min_frac_neg(min_frac_neg):
	rng = numpy.random.RandomState(0)
	values = numpy.sort(numpy.concatenate([rng.exponential(1, 5000),
		rng.exponential(1, 500) + 6]))
	null_values = rng.exponential(1, 5000)

	threshold = _isotonic_thresholds(values, null_values, True, 0.05,
		min_frac_neg=min_frac_neg)
	assert threshold in values


def test_isotonic_thresholds_min_frac_neg_monotonic():
	# Assuming more of the values are null can only raise the threshold.
	rng = numpy.random.RandomState(1)
	values = numpy.sort(numpy.concatenate([rng.exponential(1, 3000),
		rng.exponential(1, 3000) + 3]))
	null_values = rng.exponential(1, 3000)

	thresholds = [_isotonic_thresholds(values, null_values, True, 0.1,
		min_frac_neg=frac) for frac in [0.0, 0.5, 0.9, 1.0]]
	assert numpy.all(numpy.diff(thresholds) >= 0)


def test_isotonic_thresholds_indistinguishable():
	# With no signal only the most extreme value is forced to pass.
	rng = numpy.random.RandomState(0)
	values = numpy.sort(rng.exponential(1, 2000))
	null_values = rng.exponential(1, 2000)

	threshold = _isotonic_thresholds(values, null_values, True, 0.01)
	assert threshold >= numpy.percentile(values, 99)


def test_isotonic_thresholds_real(attributions):
	pos, neg, smoothed = _smooth_and_split(attributions, 20)
	pos_null, neg_null = _laplacian_null(smoothed, 20, 10000)

	pos_threshold = _isotonic_thresholds(pos, pos_null, True, 0.05)
	neg_threshold = _isotonic_thresholds(neg, neg_null, False, 0.05)
	assert_array_almost_equal([pos_threshold, neg_threshold], [2.1326, -2.3010],
		4)


##


def test_refine_thresholds_unchanged():
	vals = numpy.linspace(-1, 1, 101)
	pos, neg = _refine_thresholds(vals, 0.9, -0.9, 0.05, 0.5)
	assert (pos, neg) == (0.9, -0.9)


@pytest.mark.parametrize("min_frac", [0.1, 0.25, 0.5])
def test_refine_thresholds_too_few(min_frac):
	vals = numpy.random.RandomState(0).randn(10000)
	pos, neg = _refine_thresholds(vals, 100.0, -100.0, min_frac, 0.9)

	assert pos == numpy.percentile(numpy.abs(vals), 100*(1-min_frac))
	assert neg == -pos
	frac = (numpy.sum(vals >= pos) + numpy.sum(vals <= neg)) / len(vals)
	assert abs(frac - min_frac) < 0.01


@pytest.mark.parametrize("max_frac", [0.1, 0.25, 0.5])
def test_refine_thresholds_too_many(max_frac):
	vals = numpy.random.RandomState(0).randn(10000)
	pos, neg = _refine_thresholds(vals, 0.0, 0.0, 0.01, max_frac)

	assert pos == numpy.percentile(numpy.abs(vals), 100*(1-max_frac))
	assert neg == -pos
	frac = (numpy.sum(vals >= pos) + numpy.sum(vals <= neg)) / len(vals)
	assert abs(frac - max_frac) < 0.01


def test_refine_thresholds_asymmetric():
	# Refinement replaces asymmetric thresholds with symmetric ones.
	vals = numpy.random.RandomState(0).randn(10000)
	pos, neg = _refine_thresholds(vals, 0.1, -50.0, 0.01, 0.2)
	assert neg == -pos
	assert pos > 0.1


def test_refine_thresholds_boundaries():
	vals = numpy.array([-2.0, -1.0, 0.0, 1.0, 2.0])
	assert _refine_thresholds(vals, 2.0, -2.0, 0.4, 0.4) == (2.0, -2.0)


##


def test_extract_seqlets_real(attributions):
	seqlets, threshold = extract_seqlets(attributions, window_size=20,
		flank=5, suppress=15, **KWARGS)

	assert len(seqlets) == 551
	assert abs(threshold - 0.72265625) < 1e-6
	assert _coords(seqlets[:4]) == [(0, 117, 147, False), (1, 130, 160, False),
		(2, 144, 174, False), (3, 163, 193, False)]


@pytest.mark.parametrize("window_size", [6, 10, 15, 20, 21])
@pytest.mark.parametrize("flank", [1, 5, 10])
def test_extract_seqlets(attributions, window_size, flank):
	suppress = int(0.5*window_size) + flank
	seqlets, threshold = extract_seqlets(attributions, window_size, flank,
		suppress, **KWARGS)

	assert len(seqlets) > 100
	assert threshold > 0
	for seqlet in seqlets:
		assert len(seqlet) == window_size + 2*flank
		assert seqlet.start >= 0
		assert seqlet.end <= 300
		assert seqlet.is_revcomp is False
		assert seqlet.sequence is None

	idxs = [s.example_idx for s in seqlets]
	assert idxs == sorted(idxs)


@pytest.mark.parametrize("suppress", [5, 15, 30])
def test_extract_seqlets_suppress(attributions, suppress):
	seqlets, _ = extract_seqlets(attributions, 10, 5, suppress, **KWARGS)

	for i in range(300):
		starts = sorted(s.start for s in seqlets if s.example_idx == i)
		assert numpy.all(numpy.diff(starts) > suppress)


def test_extract_seqlets_suppress_count(attributions):
	counts = [len(extract_seqlets(attributions, 10, 5, suppress, **KWARGS)[0])
		for suppress in [5, 10, 20, 40]]
	assert numpy.all(numpy.diff(counts) < 0)


@pytest.mark.parametrize("target_fdr,expected", [(0.01, 496), (0.05, 551),
	(0.2, 698)])
def test_extract_seqlets_target_fdr(attributions, target_fdr, expected):
	# The returned threshold is capped by weak_threshold_for_counting_sign,
	# so only the number of seqlets depends on the target FDR here.
	kwargs = dict(KWARGS, target_fdr=target_fdr)
	seqlets, threshold = extract_seqlets(attributions, 20, 5, 15, **kwargs)

	assert len(seqlets) == expected
	assert abs(threshold - 0.72265625) < 1e-6


@pytest.mark.parametrize("weak", [0.2, 0.5, 0.8, 0.95])
def test_extract_seqlets_weak_threshold(attributions, weak):
	kwargs = dict(KWARGS, weak_threshold_for_counting_sign=weak)
	seqlets, threshold = extract_seqlets(attributions, 20, 5, 15, **kwargs)

	distribution = numpy.sort(numpy.abs(_smooth_and_split(attributions,
		20)[2].ravel()))
	assert threshold <= distribution[int(weak * len(distribution))]
	assert len(seqlets) == 551


def test_extract_seqlets_weak_threshold_monotonic(attributions):
	thresholds = [extract_seqlets(attributions, 20, 5, 15, **dict(KWARGS,
		weak_threshold_for_counting_sign=weak))[1] for weak in [0.1, 0.4, 0.7]]
	assert numpy.all(numpy.diff(thresholds) > 0)


@pytest.mark.parametrize("min_frac", [0.05, 0.1, 0.2])
def test_extract_seqlets_min_passing(attributions, min_frac):
	kwargs = dict(KWARGS, min_passing_windows_frac=min_frac,
		max_passing_windows_frac=0.5)
	seqlets0, _ = extract_seqlets(attributions, 20, 5, 15, **KWARGS)
	seqlets1, _ = extract_seqlets(attributions, 20, 5, 15, **kwargs)
	assert len(seqlets1) >= len(seqlets0)


@pytest.mark.parametrize("max_frac", [0.005, 0.01, 0.02])
def test_extract_seqlets_max_passing(attributions, max_frac):
	kwargs = dict(KWARGS, min_passing_windows_frac=0.001,
		max_passing_windows_frac=max_frac)
	seqlets0, _ = extract_seqlets(attributions, 20, 5, 15, **KWARGS)
	seqlets1, _ = extract_seqlets(attributions, 20, 5, 15, **kwargs)
	assert len(seqlets1) <= len(seqlets0)


def test_extract_seqlets_negative(attributions):
	# The absolute value of passing windows is used, so negating every
	# attribution calls seqlets at the same positions.
	seqlets0, _ = extract_seqlets(attributions, 20, 5, 15, **KWARGS)
	seqlets1, _ = extract_seqlets(-attributions, 20, 5, 15, **KWARGS)

	overlap = set(_coords(seqlets0)) & set(_coords(seqlets1))
	assert len(overlap) > 0.9 * len(seqlets0)


def test_extract_seqlets_planted():
	rng = numpy.random.RandomState(0)
	attributions = rng.laplace(0, 0.05, size=(200, 150))
	attributions[:, 70:80] += rng.uniform(0.5, 1.5, size=(200, 1))

	seqlets, threshold = extract_seqlets(attributions, 10, 3, 8, **KWARGS)
	first = {}
	for s in seqlets:
		first.setdefault(s.example_idx, s)

	assert len(first) == 200
	assert all(abs(s.start + 3 - 70) <= 1 for s in first.values())


def test_extract_seqlets_does_not_modify(attributions):
	attributions0 = attributions.copy()
	extract_seqlets(attributions, 20, 5, 15, **KWARGS)
	assert_array_equal(attributions, attributions0)


def test_extract_seqlets_float64(attributions):
	seqlets0, threshold0 = extract_seqlets(attributions, 20, 5, 15, **KWARGS)
	seqlets1, threshold1 = extract_seqlets(attributions.astype('float64'), 20,
		5, 15, **KWARGS)

	assert _coords(seqlets0) == _coords(seqlets1)
	assert abs(threshold0 - threshold1) < 1e-4


def test_extract_seqlets_subset(attributions):
	seqlets, threshold = extract_seqlets(attributions[:50], 20, 5, 15,
		**KWARGS)
	assert all(s.example_idx < 50 for s in seqlets)
	assert len(seqlets) > 50


@pytest.mark.xfail(strict=True, reason="bug: extract_seqlets with flank=0 sets "
	"smoothed_tracks[:, -0:], which is every column, to -inf, so no seqlets "
	"are ever returned")
def test_extract_seqlets_no_flank(attributions):
	seqlets, threshold = extract_seqlets(attributions, 20, 0, 10, **KWARGS)
	assert len(seqlets) > 0
