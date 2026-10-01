# test_util.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import pytest

from modiscolite.util import MemeDataType
from modiscolite.util import cpu_sliding_window_sum
from modiscolite.util import binary_search_perplexity
from modiscolite.util import compute_per_position_ic
from modiscolite.util import rolling_window
from modiscolite.util import magnitude
from modiscolite.util import l1
from modiscolite.util import get_2d_data_from_patterns
from modiscolite.util import calculate_window_offsets
from modiscolite.util import filter_bed_rows_by_chrom

from modiscolite.core import SeqletSet

from .synthetic import random_track_set
from .synthetic import make_seqlets

from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


def _entropy(beta, distances):
	ps = numpy.exp(-distances * beta)
	sum_ps = numpy.sum(ps) + 1
	ps = ps / sum_ps
	return numpy.log(sum_ps) + beta * numpy.sum(distances * ps)


def _ic(ppm, background, pseudocount):
	ic = numpy.zeros(len(ppm))
	for i in range(len(ppm)):
		for j in range(len(background)):
			p = (ppm[i, j] + pseudocount) / (1 + pseudocount * len(background))
			ic[i] += numpy.log2(p) * ppm[i, j]
			ic[i] -= numpy.log2(background[j]) * background[j]

	return ic


@pytest.fixture
def X():
	return numpy.random.RandomState(0).randn(20, 4)


@pytest.fixture
def random_seqlets():
	track_set = random_track_set(n=6, length=40)
	coords = [(0, 0, 10, False), (1, 5, 15, True), (2, 20, 30, False),
		(5, 30, 40, True)]
	return make_seqlets(track_set, coords)


##


@pytest.mark.parametrize("name,value", [("PFM", "PFM"), ("CWM", "CWM"),
	("hCWM", "hCWM"), ("CWM_PFM", "CWM-PFM"), ("hCWM_PFM", "hCWM-PFM")])
def test_meme_data_type_value(name, value):
	member = getattr(MemeDataType, name)
	assert member.value == value
	assert str(member) == value
	assert MemeDataType(value) is member


def test_meme_data_type_members():
	assert [m.value for m in MemeDataType] == ["PFM", "CWM", "hCWM",
		"CWM-PFM", "hCWM-PFM"]


@pytest.mark.parametrize("value", ["pfm", "cwm", "CWM_PFM", "hcwm", "", "PWM"])
def test_meme_data_type_raises(value):
	assert_raises(ValueError, MemeDataType, value)


##


@pytest.mark.parametrize("length,window_size", [(1, 1), (2, 1), (2, 2),
	(7, 1), (7, 3), (7, 7), (50, 2), (50, 5), (50, 20), (301, 1), (301, 20),
	(301, 30), (301, 301)])
def test_cpu_sliding_window_sum(length, window_size):
	arr = numpy.random.RandomState(length).randn(length)
	y = cpu_sliding_window_sum(arr, window_size)

	assert y.shape == (length - window_size + 1,)
	assert y.dtype == numpy.float64
	assert_array_almost_equal(y, numpy.convolve(arr, numpy.ones(window_size),
		'valid'))


def test_cpu_sliding_window_sum_values():
	arr = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
	assert_array_almost_equal(cpu_sliding_window_sum(arr, 2), [3, 5, 7, 9])
	assert_array_almost_equal(cpu_sliding_window_sum(arr, 3), [6, 9, 12])
	assert_array_almost_equal(cpu_sliding_window_sum(arr, 5), [15])


@pytest.mark.parametrize("dtype", ['int32', 'int64', 'float32', 'float64'])
def test_cpu_sliding_window_sum_dtypes(dtype):
	arr = numpy.arange(10).astype(dtype)
	y = cpu_sliding_window_sum(arr, 4)

	assert y.dtype == numpy.float64
	assert_array_almost_equal(y, [6, 10, 14, 18, 22, 26, 30])


def test_cpu_sliding_window_sum_negative():
	arr = -numpy.ones(6)
	assert_array_almost_equal(cpu_sliding_window_sum(arr, 3), [-3, -3, -3, -3])


def test_cpu_sliding_window_sum_list():
	assert_array_almost_equal(cpu_sliding_window_sum([1, 0, 2, 0], 2),
		[1, 2, 2])


def test_cpu_sliding_window_sum_argmax():
	arr = numpy.zeros(40)
	arr[22:27] = 1
	assert numpy.argmax(cpu_sliding_window_sum(arr, 5)) == 22


def test_cpu_sliding_window_sum_raises():
	assert_raises(ValueError, cpu_sliding_window_sum, numpy.ones(3), 5)


##


@pytest.mark.parametrize("k", [5, 10, 50, 200])
@pytest.mark.parametrize("perplexity", [1.5, 2.0, 3.0, 4.5])
def test_binary_search_perplexity(k, perplexity):
	distances = numpy.random.RandomState(k).uniform(0.1, 5, size=k)
	beta = binary_search_perplexity(perplexity, distances)

	assert isinstance(beta, float)
	assert beta > 0
	assert abs(_entropy(beta, distances) - numpy.log(perplexity)) < 1e-4


@pytest.mark.parametrize("k", [3, 20])
def test_binary_search_perplexity_monotonic(k):
	distances = numpy.random.RandomState(0).uniform(0.1, 5, size=k)
	betas = [binary_search_perplexity(p, distances) for p in [1.2, 1.5, 2, 3]]
	assert all(numpy.diff(betas) < 0)


def test_binary_search_perplexity_permutation():
	distances = numpy.random.RandomState(0).uniform(0.1, 5, size=30)
	beta0 = binary_search_perplexity(5.0, distances)
	beta1 = binary_search_perplexity(5.0, distances[::-1].copy())
	assert abs(beta0 - beta1) < 1e-8


def test_binary_search_perplexity_first_step():
	# beta starts at 1.0 so a matching entropy returns immediately.
	distances = numpy.array([0.5, 1.0, 2.0])
	perplexity = numpy.exp(_entropy(1.0, distances))
	assert binary_search_perplexity(perplexity, distances) == 1.0


def test_binary_search_perplexity_unreachable():
	# Perplexity above k + 1 cannot be reached, so beta shrinks towards zero
	# and the entropy approaches its maximum of log(k + 1).
	distances = numpy.random.RandomState(0).uniform(0.1, 5, size=5)
	beta = binary_search_perplexity(20.0, distances)

	assert beta < 1e-20
	assert abs(_entropy(beta, distances) - numpy.log(6)) < 1e-6


def test_binary_search_perplexity_raises():
	distances = numpy.ones(5, dtype='float32')
	assert_raises(TypeError, binary_search_perplexity, 2.0, distances)


##


@pytest.mark.parametrize("length", [1, 5, 30, 100])
@pytest.mark.parametrize("pseudocount", [0.001, 0.01, 0.1])
def test_compute_per_position_ic(length, pseudocount):
	ppm = numpy.random.RandomState(length).dirichlet(numpy.ones(4), size=length)
	background = numpy.array([0.25, 0.25, 0.25, 0.25])

	ic = compute_per_position_ic(ppm, background, pseudocount)
	assert ic.shape == (length,)
	assert_array_almost_equal(ic, _ic(ppm, background, pseudocount))


@pytest.mark.parametrize("background", [[0.25, 0.25, 0.25, 0.25],
	[0.3, 0.2, 0.2, 0.3], [0.1, 0.4, 0.4, 0.1], [0.7, 0.1, 0.1, 0.1]])
def test_compute_per_position_ic_background(background):
	ppm = numpy.random.RandomState(0).dirichlet(numpy.ones(4), size=12)
	background = numpy.array(background)

	ic = compute_per_position_ic(ppm, background, 0.001)
	assert_array_almost_equal(ic, _ic(ppm, background, 0.001))


@pytest.mark.parametrize("pseudocount", [0.0, 0.001, 0.5])
def test_compute_per_position_ic_uniform(pseudocount):
	ppm = numpy.full((10, 4), 0.25)
	ic = compute_per_position_ic(ppm, numpy.full(4, 0.25), pseudocount)
	assert_array_almost_equal(ic, numpy.zeros(10))


def test_compute_per_position_ic_one_hot():
	ppm = numpy.eye(4)[[0, 1, 2, 3, 3, 2]]
	ic = compute_per_position_ic(ppm, numpy.full(4, 0.25), 0.001)
	assert_array_almost_equal(ic, numpy.full(6, 2 + numpy.log2(1.001 / 1.004)))


def test_compute_per_position_ic_ordering():
	ppm = numpy.array([[0.25, 0.25, 0.25, 0.25], [0.4, 0.2, 0.2, 0.2],
		[0.7, 0.1, 0.1, 0.1], [0.97, 0.01, 0.01, 0.01]])

	ic = compute_per_position_ic(ppm, numpy.full(4, 0.25), 0.001)
	assert all(numpy.diff(ic) > 0)


@pytest.mark.parametrize("scale", [0.5, 2, 10, 137])
def test_compute_per_position_ic_unnormalized(scale):
	ppm = numpy.random.RandomState(0).dirichlet(numpy.ones(4), size=8)
	background = numpy.full(4, 0.25)

	ic0 = compute_per_position_ic(ppm, background, 0.001)
	ic1 = compute_per_position_ic(ppm * scale, background, 0.001)
	assert_array_almost_equal(ic0, ic1)


def test_compute_per_position_ic_mixed_rows():
	ppm = numpy.array([[0.25, 0.25, 0.25, 0.25], [2.0, 2.0, 0.0, 0.0]])
	ic = compute_per_position_ic(ppm, numpy.full(4, 0.25), 0.001)
	assert_array_almost_equal(ic, _ic(ppm / ppm.sum(axis=1, keepdims=True),
		numpy.full(4, 0.25), 0.001))


def test_compute_per_position_ic_patterns(pos_patterns):
	for pattern in pos_patterns:
		ic = compute_per_position_ic(pattern.sequence, numpy.full(4, 0.25),
			0.001)

		assert ic.shape == (len(pattern),)
		assert numpy.all(ic > -0.01)
		assert numpy.all(ic < 2.0)
		assert ic.max() > 1.0


##


@pytest.mark.parametrize("shape", [(10,), (1, 10), (3, 10), (2, 3, 10),
	(2, 1, 3, 25)])
@pytest.mark.parametrize("window", [1, 3, 10])
def test_rolling_window(shape, window):
	a = numpy.random.RandomState(0).randn(*shape)
	y = rolling_window(a, window)

	assert y.shape == shape[:-1] + (shape[-1] - window + 1, window)
	for i in range(shape[-1] - window + 1):
		assert_array_equal(y[..., i, :], a[..., i:i+window])


def test_rolling_window_view():
	a = numpy.arange(12.0)
	y = rolling_window(a, 4)

	assert numpy.shares_memory(a, y)
	assert_array_equal(y.sum(axis=-1), numpy.convolve(a, numpy.ones(4),
		'valid'))


@pytest.mark.parametrize("dtype", ['int8', 'int64', 'float32', 'float64'])
def test_rolling_window_dtypes(dtype):
	a = numpy.arange(6).astype(dtype)
	y = rolling_window(a, 2)

	assert y.dtype == a.dtype
	assert_array_equal(y, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])


##


@pytest.mark.parametrize("shape", [(4,), (20, 4), (30, 8), (3, 20, 4)])
def test_magnitude(shape):
	X = numpy.random.RandomState(0).randn(*shape) + 3
	y = magnitude(X)

	assert y.shape == shape
	assert abs(y.mean()) < 1e-7
	assert abs(numpy.linalg.norm(y.ravel()) - 1) < 1e-5


def test_magnitude_values():
	y = magnitude(numpy.array([1.0, 3.0]))
	assert_array_almost_equal(y, [-0.7071068, 0.7071068])


@pytest.mark.parametrize("scale", [0.01, 1, 17])
def test_magnitude_scale(X, scale):
	assert_array_almost_equal(magnitude(X), magnitude(X * scale))


@pytest.mark.parametrize("shift", [-4, 0.5, 100])
def test_magnitude_shift(X, shift):
	assert_array_almost_equal(magnitude(X), magnitude(X + shift))


@pytest.mark.parametrize("value", [0.0, 1.0, -2.5])
def test_magnitude_constant(value):
	y = magnitude(numpy.full((10, 4), value))
	assert_array_almost_equal(y, numpy.zeros((10, 4)))


def test_magnitude_does_not_modify(X):
	X0 = X.copy()
	magnitude(X)
	assert_array_equal(X, X0)


##


@pytest.mark.parametrize("shape", [(4,), (20, 4), (30, 8), (3, 20, 4)])
def test_l1(shape):
	X = numpy.random.RandomState(0).randn(*shape)
	y = l1(X)

	assert y.shape == shape
	assert abs(numpy.abs(y).sum() - 1) < 1e-7
	assert_array_equal(numpy.sign(y), numpy.sign(X))


def test_l1_values():
	assert_array_almost_equal(l1(numpy.array([1.0, -3.0])), [0.25, -0.75])


@pytest.mark.parametrize("scale", [0.01, 1, 17])
def test_l1_scale(X, scale):
	assert_array_almost_equal(l1(X), l1(X * scale))


def test_l1_negate(X):
	assert_array_almost_equal(l1(-X), -l1(X))


def test_l1_zeros():
	X = numpy.zeros((10, 4))
	y = l1(X)

	assert y is X
	assert_array_equal(y, numpy.zeros((10, 4)))


@pytest.mark.parametrize("dtype", ['float32', 'float64'])
def test_l1_dtype(dtype):
	assert l1(numpy.ones((3, 4), dtype=dtype)).dtype == dtype


##


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize("length", [4, 20, 31])
@pytest.mark.parametrize("transformer", ['l1', 'magnitude'])
@pytest.mark.parametrize("include_hypothetical", [True, False])
def test_get_2d_data_from_patterns_shape(n, length, transformer,
	include_hypothetical):
	track_set = random_track_set(n=n, length=40)
	seqlets = make_seqlets(track_set, [(i, 3, 3+length, False)
		for i in range(n)])

	fwd, rev = get_2d_data_from_patterns(seqlets, transformer=transformer,
		include_hypothetical=include_hypothetical)

	d = 8 if include_hypothetical else 4
	assert fwd.shape == (n, length, d)
	assert rev.shape == (n, length, d)


@pytest.mark.parametrize("transformer,func", [('l1', l1),
	('magnitude', magnitude), ('other', magnitude)])
def test_get_2d_data_from_patterns_hypothetical(random_seqlets, transformer, func):
	fwd, rev = get_2d_data_from_patterns(random_seqlets, transformer=transformer)

	for i, seqlet in enumerate(random_seqlets):
		hyp, contrib = seqlet.hypothetical_contribs, seqlet.contrib_scores

		assert_array_almost_equal(fwd[i, :, :4], func(hyp))
		assert_array_almost_equal(fwd[i, :, 4:], func(contrib))
		assert_array_almost_equal(rev[i, :, :4], func(hyp[::-1, ::-1]))
		assert_array_almost_equal(rev[i, :, 4:], func(contrib[::-1, ::-1]))


@pytest.mark.parametrize("transformer,func", [('l1', l1),
	('magnitude', magnitude)])
def test_get_2d_data_from_patterns_contrib_only(random_seqlets, transformer, func):
	fwd, rev = get_2d_data_from_patterns(random_seqlets, transformer=transformer,
		include_hypothetical=False)

	for i, seqlet in enumerate(random_seqlets):
		contrib = seqlet.contrib_scores
		assert_array_almost_equal(fwd[i], func(contrib))
		assert_array_almost_equal(rev[i], func(contrib[::-1, ::-1]))


@pytest.mark.parametrize("include_hypothetical", [True, False])
def test_get_2d_data_from_patterns_revcomp(random_seqlets, include_hypothetical):
	fwd, rev = get_2d_data_from_patterns(random_seqlets,
		include_hypothetical=include_hypothetical)

	d = fwd.shape[-1]
	for i in range(0, d, 4):
		assert_array_almost_equal(rev[:, :, i:i+4], fwd[:, ::-1, i:i+4][:, :, ::-1])


def test_get_2d_data_from_patterns_l1_norm(random_seqlets):
	fwd, rev = get_2d_data_from_patterns(random_seqlets, transformer='l1')

	assert_array_almost_equal(numpy.abs(fwd[:, :, :4]).sum(axis=(1, 2)),
		numpy.ones(4))
	assert_array_almost_equal(numpy.abs(rev[:, :, 4:]).sum(axis=(1, 2)),
		numpy.ones(4))


def test_get_2d_data_from_patterns_seqlet_sets(random_seqlets):
	patterns = [SeqletSet(random_seqlets[:2]), SeqletSet(random_seqlets[2:])]
	fwd, rev = get_2d_data_from_patterns(patterns)

	assert fwd.shape == (2, 10, 8)
	assert_array_almost_equal(fwd[0, :, 4:], l1(patterns[0].contrib_scores))
	assert_array_almost_equal(fwd[1, :, :4],
		l1(patterns[1].hypothetical_contribs))


def test_get_2d_data_from_patterns_real(seqlets):
	fwd, rev = get_2d_data_from_patterns(seqlets, transformer='magnitude')

	assert fwd.shape == (551, 30, 8)
	assert rev.shape == (551, 30, 8)
	assert fwd.dtype == numpy.float32

	assert_array_almost_equal(fwd[0, 10:13], [
		[-0.0146,  0.0622, -0.0045, -0.0432, -0.0232, -0.0232, -0.0287, -0.0232],
		[ 0.0144, -0.013 , -0.0022,  0.0007, -0.0057, -0.0232, -0.0232, -0.0232],
		[-0.0373, -0.0119,  0.0668, -0.0177, -0.0232, -0.0232,  0.0579, -0.0232]
	], 4)

	assert_array_almost_equal(rev[-1, 14:16], [
		[ 0.0381, -0.0083, -0.0169, -0.0129,  0.0189, -0.0281, -0.0281, -0.0281],
		[-0.0977, -0.1159,  0.3375, -0.1239, -0.0281, -0.0281,  0.3877, -0.0281]
	], 4)


def test_get_2d_data_from_patterns_empty():
	fwd, rev = get_2d_data_from_patterns([])
	assert fwd.shape == (0,)
	assert rev.shape == (0,)


##


@pytest.mark.parametrize("center,window_size,expected", [(200, 400, (0, 400)),
	(200, 200, (100, 300)), (1057, 2, (1056, 1058)), (5, 10, (0, 10)),
	(0, 0, (0, 0)), (50, 30, (35, 65)), (1000, 1000, (500, 1500))])
def test_calculate_window_offsets(center, window_size, expected):
	assert calculate_window_offsets(center, window_size) == expected


@pytest.mark.parametrize("window_size", [2, 10, 50, 400, 1000])
def test_calculate_window_offsets_width(window_size):
	start, end = calculate_window_offsets(1057, window_size)
	assert end - start == window_size


@pytest.mark.xfail(strict=True, reason="bug: calculate_window_offsets returns a window of "
	"width window_size - 1 when window_size is odd, e.g. (8, 12) for "
	"center=10, window_size=5")
@pytest.mark.parametrize("window_size", [1, 5, 21, 401])
def test_calculate_window_offsets_width_odd(window_size):
	start, end = calculate_window_offsets(1057, window_size)
	assert end - start == window_size


@pytest.mark.parametrize("center", [-5, 0, 3])
def test_calculate_window_offsets_negative(center):
	assert calculate_window_offsets(center, 20) == (center - 10, center + 10)


##


@pytest.fixture
def peak_rows():
	return ['chr1\t1\t2\tpeak1', 'chr2\t1\t2\tpeak2', 'chrX\t5\t9\tpeak3',
		'chr1\t7\t8\tpeak4', 'chr10\t3\t4\tpeak5']


@pytest.mark.parametrize("valid_chroms,expected", [
	(['chr1'], [0, 3]),
	(['chr2'], [1]),
	(['chr1', 'chrX'], [0, 2, 3]),
	(['chrX', 'chr1'], [0, 2, 3]),
	(['chr1', 'chr2', 'chrX', 'chr10'], [0, 1, 2, 3, 4]),
	(['chr10'], [4]),
	(['1'], []),
	(['chr'], []),
	([], [])
])
def test_filter_bed_rows_by_chrom(peak_rows, valid_chroms, expected):
	rows = filter_bed_rows_by_chrom(peak_rows, valid_chroms)
	assert rows == [peak_rows[i] for i in expected]


def test_filter_bed_rows_by_chrom_no_prefix():
	rows = ['1\t0\t10', '2\t0\t10', 'X\t0\t10', '1\t20\t30']
	assert filter_bed_rows_by_chrom(rows, ['1']) == ['1\t0\t10', '1\t20\t30']
	assert filter_bed_rows_by_chrom(rows, ['X', '2']) == ['2\t0\t10', 'X\t0\t10']


def test_filter_bed_rows_by_chrom_empty_rows():
	assert filter_bed_rows_by_chrom([], ['chr1']) == []
	assert filter_bed_rows_by_chrom([''], ['chr1']) == []
	assert filter_bed_rows_by_chrom([''], ['']) == ['']


def test_filter_bed_rows_by_chrom_spaces():
	# Only tabs delimit columns.
	rows = ['chr1 0 10', 'chr1\t0\t10']
	assert filter_bed_rows_by_chrom(rows, ['chr1']) == ['chr1\t0\t10']
