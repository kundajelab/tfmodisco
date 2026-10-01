# test_descriptive_report.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import base64
import os
import shutil

import h5py
import numpy
import pytest

import modiscolite.descriptive_report

from modiscolite.descriptive_report import plot_to_base64
from modiscolite.descriptive_report import plot_histogram_to_base64
from modiscolite.descriptive_report import extract_seqlet_data
from modiscolite.descriptive_report import compute_global_region_size_and_distances
from modiscolite.descriptive_report import create_logos
from modiscolite.descriptive_report import create_distribution_plots
from modiscolite.descriptive_report import create_seqlet_example_logos
from modiscolite.descriptive_report import create_tomtom_match_logos
from modiscolite.descriptive_report import create_descriptive_names
from modiscolite.descriptive_report import generate_descriptive_report

from modiscolite.io import convert
from modiscolite.io import convert_new_to_old
from modiscolite.io import save_hdf5
from modiscolite.report import compute_per_position_ic

from .synthetic import random_track_set
from .synthetic import make_pattern

from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


requires_tomtom = pytest.mark.skipif(shutil.which("tomtom") is None,
	reason="the MEME suite tomtom executable is not installed")

TAGS = ["pos_patterns.pattern_0", "pos_patterns.pattern_1",
	"neg_patterns.pattern_0", "neg_patterns.pattern_1"]

GROUPS = ['pos_patterns', 'neg_patterns']

STUB = "data:image/png;base64,STUB"


def _is_png(data):
	prefix = "data:image/png;base64,"
	assert data.startswith(prefix)
	return base64.b64decode(data[len(prefix):])[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def plots(monkeypatch):
	"""Replace the plotting functions with fast recorders.

	Rendering a 50 bp logo takes about 0.25 s, so tests of the logic around
	the plots record what would be drawn instead. The recorders write a small
	file wherever a file would be written.
	"""

	calls = {'weights': [], 'base64': [], 'histogram': []}

	def _plot_weights(array, path, figsize=(10, 3), clamp=True):
		calls['weights'].append((numpy.array(array), path, figsize, clamp))
		with open(path, "wb") as f:
			f.write(b"png")

	def _plot_to_base64(array, figsize=(10, 3), clamp=True):
		calls['base64'].append((numpy.array(array), figsize, clamp))
		return STUB

	def _histogram(data, **kwargs):
		calls['histogram'].append((numpy.array(data), kwargs))
		return STUB

	module = modiscolite.descriptive_report
	monkeypatch.setattr(module, "_plot_weights", _plot_weights)
	monkeypatch.setattr(module, "plot_to_base64", _plot_to_base64)
	monkeypatch.setattr(module, "plot_histogram_to_base64", _histogram)
	return calls


@pytest.fixture
def small_h5(tmp_path):
	"""A results file with one 6 bp positive pattern of 5 seqlets, which is
	quick to render."""

	track_set = random_track_set(n=8, length=40)
	pattern = make_pattern(track_set, [(i, 3*i, 3*i+6, bool(i % 2))
		for i in range(5)])

	filename = tmp_path / "small.h5"
	save_hdf5(filename, [pattern], None, window_size=40)
	return filename


@pytest.fixture
def patterns_data(modisco_h5):
	return compute_global_region_size_and_distances(
		extract_seqlet_data(modisco_h5, GROUPS))


@pytest.fixture
def small_data(small_h5):
	return compute_global_region_size_and_distances(
		extract_seqlet_data(small_h5, GROUPS))


##


@pytest.mark.parametrize("length", [1, 5, 12])
@pytest.mark.parametrize("clamp", [True, False])
def test_plot_to_base64(length, clamp):
	array = numpy.abs(numpy.random.RandomState(length).randn(length, 4))
	assert _is_png(plot_to_base64(array, clamp=clamp))


@pytest.mark.parametrize("figsize", [(4, 1), (8, 1.2), (12, 3)])
def test_plot_to_base64_figsize(figsize):
	array = numpy.random.RandomState(0).randn(6, 4)
	assert _is_png(plot_to_base64(array, figsize=figsize))


def test_plot_to_base64_deterministic():
	array = numpy.random.RandomState(0).randn(6, 4)
	assert plot_to_base64(array) == plot_to_base64(array)


def test_plot_to_base64_closes_figure():
	import matplotlib.pyplot as plt
	n = len(plt.get_fignums())
	plot_to_base64(numpy.ones((5, 4)))
	assert len(plt.get_fignums()) == n


def test_plot_to_base64_raises():
	assert_raises(ValueError, plot_to_base64, numpy.ones((5, 3)))


##


@pytest.mark.parametrize("bins", [5, 30])
@pytest.mark.parametrize("xlim", [None, (-5, 5)])
def test_plot_histogram_to_base64(bins, xlim):
	data = numpy.random.RandomState(0).randn(200)
	image = plot_histogram_to_base64(data, bins=bins, xlim=xlim)
	assert _is_png(image)


def test_plot_histogram_to_base64_labels():
	data = numpy.random.RandomState(0).randn(50)
	image0 = plot_histogram_to_base64(data)
	image1 = plot_histogram_to_base64(data, xlabel="x", ylabel="y",
		title="title", color="lightcoral", figsize=(6, 3))

	assert _is_png(image1)
	assert image0 != image1


def test_plot_histogram_to_base64_closes_figure():
	import matplotlib.pyplot as plt
	n = len(plt.get_fignums())
	plot_histogram_to_base64(numpy.arange(10))
	assert len(plt.get_fignums()) == n


##


def test_extract_seqlet_data(modisco_h5):
	data = extract_seqlet_data(modisco_h5, GROUPS)
	assert list(data.keys()) == TAGS

	with h5py.File(modisco_h5, "r") as f:
		for tag, pattern_data in data.items():
			grp = f[tag.replace(".", "/")]
			seqlets = grp["seqlets"]
			contribs = seqlets["contrib_scores"][:]
			importance = numpy.abs(contribs).sum(axis=(1, 2))

			assert_array_equal(pattern_data['ppm'], grp["sequence"][:])
			assert_array_equal(pattern_data['cwm'], grp["contrib_scores"][:])
			assert_array_equal(pattern_data['hcwm'],
				grp["hypothetical_contribs"][:])
			assert pattern_data['n_seqlets'] == seqlets["n_seqlets"][0]
			assert_array_equal(pattern_data['seqlet_starts'], seqlets["start"][:])
			assert_array_equal(pattern_data['seqlet_ends'], seqlets["end"][:])
			assert_array_equal(pattern_data['seqlet_example_idx'],
				seqlets["example_idx"][:])
			assert_array_equal(pattern_data['seqlet_contribs'], contribs)
			assert_array_almost_equal(pattern_data['seqlet_importance'],
				importance)

			cwm = grp["contrib_scores"][:]
			assert abs(pattern_data['avg_importance'] -
				numpy.abs(cwm).sum(axis=1).mean()) < 1e-12
			assert abs(pattern_data['std_importance'] - importance.std()) < 1e-6
			assert numpy.isnan(pattern_data['median_abs_distance_from_center'])
			assert numpy.isnan(pattern_data['std_distance_from_center'])


def test_extract_seqlet_data_values(modisco_h5):
	data = extract_seqlet_data(modisco_h5, GROUPS)

	assert [data[tag]['n_seqlets'] for tag in TAGS] == [74, 26, 77, 27]
	assert_array_almost_equal([data[tag]['avg_importance'] for tag in TAGS],
		[numpy.abs(data[tag]["cwm"]).sum(axis=1).mean() for tag in TAGS])


@pytest.mark.skip(reason="bug: extract_seqlet_data computes gc_content as "
	"the mean over the C and G columns of the PPM, which is half of the GC "
	"fraction")
def test_extract_seqlet_data_gc_content(modisco_h5):
	data = extract_seqlet_data(modisco_h5, GROUPS)
	for pattern_data in data.values():
		ppm = pattern_data['ppm']
		assert abs(pattern_data['gc_content'] - ppm[:, 1:3].sum(axis=1).mean()
			) < 1e-12


@pytest.mark.parametrize("groups,expected", [(['pos_patterns'], TAGS[:2]),
	(['neg_patterns'], TAGS[2:]), (['neg_patterns', 'pos_patterns'],
	TAGS[2:] + TAGS[:2]), (['missing'], []), ([], [])])
def test_extract_seqlet_data_groups(modisco_h5, groups, expected):
	assert list(extract_seqlet_data(modisco_h5, groups).keys()) == expected


def test_extract_seqlet_data_order(tmp_path):
	track_set = random_track_set(n=6, length=40)
	patterns = [make_pattern(track_set, [(0, i, i+6, False)])
		for i in range(12)]

	filename = tmp_path / "results.h5"
	save_hdf5(filename, patterns, None, 40)

	data = extract_seqlet_data(filename, GROUPS)
	assert list(data.keys()) == ["pos_patterns.pattern_{}".format(i)
		for i in range(12)]
	assert [d['seqlet_starts'][0] for d in data.values()] == list(range(12))


def test_extract_seqlet_data_no_seqlet_contribs(modisco_h5, tmp_path):
	# Files converted from the original format have no per-seqlet scores.
	old, new = tmp_path / "old.h5", tmp_path / "new.h5"
	convert_new_to_old(modisco_h5, old)
	convert(old, new)

	data = extract_seqlet_data(new, GROUPS)
	assert list(data.keys()) == TAGS
	for pattern_data in data.values():
		assert len(pattern_data['seqlet_importance']) == 0
		assert len(pattern_data['seqlet_contribs']) == 0
		assert numpy.isnan(pattern_data['std_importance'])
		assert len(pattern_data['seqlet_starts']) == pattern_data['n_seqlets']


##


def _data(starts, ends):
	return {'seqlet_starts': numpy.array(starts),
		'seqlet_ends': numpy.array(ends)}


def test_compute_global_region_size_and_distances():
	data = {'a': _data([10, 20], [30, 40]), 'b': _data([0], [20])}
	updated = compute_global_region_size_and_distances(data)

	assert list(updated.keys()) == ['a', 'b']
	for d in updated.values():
		assert d['global_region_size'] == 40
		assert d['global_center'] == 20

	# Centers are 20 and 30 for a, and 10 for b.
	assert updated['a']['median_abs_distance_from_center'] == 5
	assert updated['a']['std_distance_from_center'] == 5
	assert updated['b']['median_abs_distance_from_center'] == 10
	assert updated['b']['std_distance_from_center'] == 0


def test_compute_global_region_size_and_distances_empty():
	data = {'a': _data([], []), 'b': _data([], [])}
	updated = compute_global_region_size_and_distances(data)

	for d in updated.values():
		assert d['global_region_size'] == 400
		assert d['global_center'] == 200
		assert numpy.isnan(d['median_abs_distance_from_center'])
		assert numpy.isnan(d['std_distance_from_center'])


def test_compute_global_region_size_and_distances_mixed():
	data = {'a': _data([], []), 'b': _data([100, 150], [120, 190])}
	updated = compute_global_region_size_and_distances(data)

	# Centers of 110 and 170 lie 35 and 25 bp from the global center of 145.
	assert updated['a']['global_region_size'] == 90
	assert updated['a']['global_center'] == 145
	assert numpy.isnan(updated['a']['median_abs_distance_from_center'])
	assert updated['b']['median_abs_distance_from_center'] == 30
	assert updated['b']['std_distance_from_center'] == 5


def test_compute_global_region_size_and_distances_no_patterns():
	assert compute_global_region_size_and_distances({}) == {}


@pytest.mark.parametrize("offset", [0, 17, 1000])
def test_compute_global_region_size_and_distances_translation(offset):
	data0 = {'a': _data([10, 20, 5], [30, 40, 25])}
	data1 = {'a': _data([10+offset, 20+offset, 5+offset],
		[30+offset, 40+offset, 25+offset])}

	d0 = compute_global_region_size_and_distances(data0)['a']
	d1 = compute_global_region_size_and_distances(data1)['a']

	assert d0['global_region_size'] == d1['global_region_size']
	assert d1['global_center'] == d0['global_center'] + offset
	assert d0['median_abs_distance_from_center'] == \
		d1['median_abs_distance_from_center']


def test_compute_global_region_size_and_distances_real(patterns_data):
	for data in patterns_data.values():
		assert data['global_region_size'] == 275
		assert data['global_center'] == 149.5

	assert_array_almost_equal([patterns_data[t][
		'median_abs_distance_from_center'] for t in TAGS],
		[9.5, 31.5, 16.5, 47.5], 4)
	assert_array_almost_equal([patterns_data[t][
		'std_distance_from_center'] for t in TAGS],
		[11.7567, 22.1886, 20.0074, 25.3212], 4)


##


LOGO_KEYS = ['cwm', 'cwm_path', 'hcwm', 'hcwm_path', 'ic_ppm', 'ic_ppm_path',
	'pwm', 'pwm_path', 'trimmed_cwm_fwd', 'trimmed_cwm_fwd_path',
	'trimmed_cwm_rev', 'trimmed_cwm_rev_path', 'trimmed_cwm',
	'trimmed_cwm_path']


def test_create_logos(patterns_data, tmp_path, plots):
	logos = create_logos(patterns_data, str(tmp_path), 0.3)

	assert list(logos.keys()) == TAGS
	for tag in TAGS:
		assert sorted(logos[tag].keys()) == sorted(LOGO_KEYS)
		assert logos[tag]['cwm'] == STUB
		assert logos[tag]['pwm'] == logos[tag]['ic_ppm']
		assert logos[tag]['trimmed_cwm'] == logos[tag]['trimmed_cwm_fwd']
		assert logos[tag]['cwm_path'] == os.path.join(str(tmp_path), "logos",
			tag, "cwm_logo.png")

		assert sorted(os.listdir(tmp_path / "logos" / tag)) == [
			"cwm_logo.png", "hcwm_logo.png", "ic_ppm_logo.png",
			"trimmed_cwm_fwd_logo.png", "trimmed_cwm_rev_logo.png"]

	assert len(plots['weights']) == 5 * len(TAGS)
	assert len(plots['base64']) == 5 * len(TAGS)


def test_create_logos_arrays(patterns_data, tmp_path, plots):
	create_logos({TAGS[0]: patterns_data[TAGS[0]]}, str(tmp_path), 0.3)
	data = patterns_data[TAGS[0]]
	(cwm, _, _, clamp0), (hcwm, _, _, clamp1), (ic_ppm, _, _, _) = \
		plots['weights'][:3]

	assert_array_equal(cwm, data['cwm'])
	assert_array_equal(hcwm, data['hcwm'])
	assert (clamp0, clamp1) == (True, False)

	ic = compute_per_position_ic(data['ppm'], numpy.full(4, 0.25), 0.001)
	assert_array_almost_equal(ic_ppm, data['ppm'] * ic[:, None])

	for (a0, _, _, c0), (a1, _, c1) in zip(plots['weights'], plots['base64']):
		assert_array_equal(a0, a1)
		assert c0 == c1


@pytest.mark.parametrize("trim_threshold", [0.0, 0.3, 0.6, 1.0])
def test_create_logos_trim(tmp_path, plots, trim_threshold):
	cwm = numpy.zeros((20, 4))
	cwm[:, 0] = [0.1, 0, 0, 0, 0, 0.2, 0.5, 1, 0.5, 0.2, 0, 0, 0.9, 0, 0, 0, 0,
		0, 0, 0.05]
	data = {'p': {'cwm': cwm, 'hcwm': cwm, 'ppm': numpy.full((20, 4), 0.25)}}
	create_logos(data, str(tmp_path), trim_threshold)

	fwd, rev = plots['weights'][3][0], plots['weights'][4][0]
	score = numpy.abs(cwm).sum(axis=1)
	passing = numpy.where(score >= score.max() * trim_threshold)[0]
	start, end = max(passing.min() - 2, 0), min(passing.max() + 3, 20)
	assert_array_equal(fwd, cwm[start:end])

	score = score[::-1]
	passing = numpy.where(score >= score.max() * trim_threshold)[0]
	start, end = max(passing.min() - 2, 0), min(passing.max() + 3, 20)
	assert_array_equal(rev, cwm[::-1, ::-1][start:end])


def test_create_logos_trim_values(tmp_path, plots):
	cwm = numpy.zeros((20, 4))
	cwm[8:11, 2] = 1.0
	data = {'p': {'cwm': cwm, 'hcwm': cwm, 'ppm': numpy.full((20, 4), 0.25)}}
	create_logos(data, str(tmp_path), 0.3)

	# Two positions of padding on the left and two on the right.
	assert_array_equal(plots['weights'][3][0], cwm[6:13])
	assert_array_equal(plots['weights'][4][0], cwm[::-1, ::-1][7:14])


def test_create_logos_figsize(patterns_data, tmp_path, plots):
	create_logos({TAGS[1]: patterns_data[TAGS[1]]}, str(tmp_path), 0.3)
	sizes = [figsize for _, _, figsize, _ in plots['weights']]
	assert sizes == [(12, 3), (12, 3), (12, 3), (10, 3), (10, 3)]


def test_create_logos_render(small_data, tmp_path):
	logos = create_logos(small_data, str(tmp_path), 0.3)

	tag = "pos_patterns.pattern_0"
	assert list(logos.keys()) == [tag]
	for key in ['cwm', 'hcwm', 'ic_ppm', 'trimmed_cwm_fwd', 'trimmed_cwm_rev']:
		assert _is_png(logos[tag][key])
		with open(logos[tag][key + '_path'], "rb") as f:
			assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_create_logos_existing_dir(small_data, tmp_path, plots):
	(tmp_path / "logos" / "pos_patterns.pattern_0").mkdir(parents=True)
	logos = create_logos(small_data, str(tmp_path), 0.3)
	assert list(logos.keys()) == ["pos_patterns.pattern_0"]


def test_create_logos_empty(tmp_path, plots):
	assert create_logos({}, str(tmp_path), 0.3) == {}
	assert os.listdir(tmp_path / "logos") == []


##


def test_create_distribution_plots(patterns_data, tmp_path, plots):
	distributions = create_distribution_plots(patterns_data, str(tmp_path))

	assert list(distributions.keys()) == TAGS
	for tag in TAGS:
		assert distributions[tag] == {'importance': STUB, 'spatial': STUB}

	assert len(plots['histogram']) == 2 * len(TAGS)


def test_create_distribution_plots_data(patterns_data, tmp_path, plots):
	data = patterns_data[TAGS[0]]
	create_distribution_plots({TAGS[0]: data}, str(tmp_path))
	(importance, kw0), (centers, kw1) = plots['histogram']

	assert_array_equal(importance, data['seqlet_importance'])
	assert kw0['xlabel'] == 'Seqlet Total Contribution Score'
	assert kw0['title'] == ('Seqlet Contribution Score Distribution - '
		'pos_patterns.pattern_0')

	expected = (data['seqlet_starts'] + data['seqlet_ends']) / 2 - 149.5
	assert_array_equal(centers, expected)
	assert kw1['xlim'] == (-137.5, 137.5)


def test_create_distribution_plots_no_global(tmp_path, plots):
	# Without a global region the centers are not shifted.
	data = {'p': {'seqlet_importance': numpy.array([]),
		'seqlet_starts': numpy.array([10, 20]),
		'seqlet_ends': numpy.array([30, 50])}}
	distributions = create_distribution_plots(data, str(tmp_path))

	assert distributions == {'p': {'spatial': STUB}}
	assert_array_equal(plots['histogram'][0][0], [20, 35])
	assert plots['histogram'][0][1]['xlim'] is None


def test_create_distribution_plots_no_seqlets(tmp_path, plots):
	data = {'p': {'seqlet_importance': numpy.array([]),
		'seqlet_starts': numpy.array([]), 'seqlet_ends': numpy.array([])}}

	assert create_distribution_plots(data, str(tmp_path)) == {'p': {}}
	assert plots['histogram'] == []


def test_create_distribution_plots_render(small_data, tmp_path):
	distributions = create_distribution_plots(small_data, str(tmp_path))
	plots = distributions["pos_patterns.pattern_0"]

	assert _is_png(plots['importance'])
	assert _is_png(plots['spatial'])


##


@pytest.mark.parametrize("n_examples,expected", [(1, 1), (3, 3), (10, 10),
	(100, 26)])
def test_create_seqlet_example_logos(patterns_data, tmp_path, plots,
	n_examples, expected):
	data = {TAGS[1]: patterns_data[TAGS[1]]}
	examples = create_seqlet_example_logos(data, str(tmp_path), n_examples)

	logos = examples[TAGS[1]]
	assert len(logos) == expected
	assert [logo['rank'] for logo in logos] == list(range(expected, 0, -1))
	assert all(logo['base64'] == STUB for logo in logos)

	quantiles = numpy.linspace(10, 100, expected)[::-1].astype(int)
	assert [logo['quantile'] for logo in logos] == list(quantiles)
	assert all(os.path.exists(logo['path']) for logo in logos)


def test_create_seqlet_example_logos_selection(tmp_path, plots):
	importance = numpy.array([5.0, 1.0, 3.0, 2.0, 4.0])
	contribs = numpy.arange(5)[:, None, None] * numpy.ones((5, 6, 4))
	data = {'p': {'seqlet_importance': importance, 'seqlet_contribs': contribs}}

	examples = create_seqlet_example_logos(data, str(tmp_path), 5)['p']

	# Quantiles 10, 32.5, 55, 77.5 and 100 are closest to 1.4, 2.3, 3.2, 4.1
	# and 5, which are seqlets 1, 3, 2, 4 and 0.
	assert [e['importance'] for e in examples] == [5.0, 4.0, 3.0, 2.0, 1.0]
	assert [e['quantile'] for e in examples] == [100, 77, 55, 32, 10]
	assert [int(a[0, 0]) for a, _, _, _ in plots['weights']] == [1, 3, 2,
		4, 0]


def test_create_seqlet_example_logos_unique(tmp_path, plots):
	# Seqlets are not reused even when several quantiles share a value.
	importance = numpy.array([1.0, 1.0, 1.0, 9.0])
	contribs = numpy.arange(4)[:, None, None] * numpy.ones((4, 5, 4))
	data = {'p': {'seqlet_importance': importance, 'seqlet_contribs': contribs}}

	create_seqlet_example_logos(data, str(tmp_path), 4)
	used = [int(a[0, 0]) for a, _, _, _ in plots['weights']]
	assert sorted(used) == [0, 1, 2, 3]


def test_create_seqlet_example_logos_skips(patterns_data, tmp_path, plots):
	data = {'empty': {'seqlet_importance': numpy.array([]),
		'seqlet_contribs': []}, TAGS[0]: patterns_data[TAGS[0]]}
	examples = create_seqlet_example_logos(data, str(tmp_path), 2)

	assert list(examples.keys()) == [TAGS[0]]
	assert not os.path.exists(tmp_path / "seqlet_examples" / "empty")


def test_create_seqlet_example_logos_paths(patterns_data, tmp_path, plots):
	data = {TAGS[0]: patterns_data[TAGS[0]]}
	examples = create_seqlet_example_logos(data, str(tmp_path), 2)

	assert [e['path'] for e in examples[TAGS[0]]] == [
		os.path.join(str(tmp_path), "seqlet_examples", TAGS[0],
		"quantile_{}.png".format(q)) for q in [100, 10]]
	assert all(figsize == (8, 1.2) for _, _, figsize, _ in plots['weights'])


def test_create_seqlet_example_logos_render(small_data, tmp_path):
	examples = create_seqlet_example_logos(small_data, str(tmp_path), 2)
	logos = examples["pos_patterns.pattern_0"]

	assert len(logos) == 2
	assert all(_is_png(logo['base64']) for logo in logos)
	assert all(isinstance(logo['importance'], float) for logo in logos)


##


def test_create_tomtom_match_logos(meme_db, tmp_path, plots):
	tomtom_data = {'a': {'match_0': 'db_motif_2', 'pval_0': 0.01,
		'match_1': 'db_motif_3', 'pval_1': 0.02},
		'b': {'match_0': ' db_motif_4 ', 'pval_0': 0.5}}

	logos = create_tomtom_match_logos(tomtom_data, str(tmp_path), meme_db, 2)

	assert sorted(logos.keys()) == ['a', 'b']
	assert logos['a'] == {
		'match_0_logo': os.path.join(str(tmp_path), "tomtom_logos",
			"a_match_0.png"), 'match_0_base64': STUB,
		'match_1_logo': os.path.join(str(tmp_path), "tomtom_logos",
			"a_match_1.png"), 'match_1_base64': STUB}
	assert sorted(logos['b'].keys()) == ['match_0_base64', 'match_0_logo']


def test_create_tomtom_match_logos_ic(meme_db, tmp_path, plots):
	from memelite.io import read_meme
	ppm = read_meme(str(meme_db))["db_motif_3 NAME3"].T

	create_tomtom_match_logos({'a': {'match_0': 'db_motif_3'}}, str(tmp_path),
		meme_db, 1)

	ic = compute_per_position_ic(ppm, numpy.full(4, 0.25), 0.001)
	assert_array_almost_equal(plots['weights'][0][0], ppm * ic[:, None])
	assert plots['weights'][0][2] == (8, 2)


@pytest.mark.parametrize("top_n_matches,expected", [(1, 1), (2, 2), (3, 2)])
def test_create_tomtom_match_logos_top_n(meme_db, tmp_path, plots,
	top_n_matches, expected):
	tomtom_data = {'a': {'match_0': 'db_motif_0', 'match_1': 'db_motif_1'}}
	logos = create_tomtom_match_logos(tomtom_data, str(tmp_path), meme_db,
		top_n_matches)
	assert len(logos['a']) == 2 * expected


@pytest.mark.parametrize("matches", [{}, {'match_0': None}, {'match_0': ''},
	{'match_0': 'not_in_db'}, {'match_1': 'db_motif_0'}])
def test_create_tomtom_match_logos_skips(meme_db, tmp_path, plots, matches):
	logos = create_tomtom_match_logos({'a': matches}, str(tmp_path), meme_db, 1)
	assert logos == {'a': {}}
	assert plots['weights'] == []


def test_create_tomtom_match_logos_render(meme_db, tmp_path):
	logos = create_tomtom_match_logos({'a': {'match_0': 'db_motif_2'}},
		str(tmp_path), meme_db, 1)

	assert _is_png(logos['a']['match_0_base64'])
	with open(logos['a']['match_0_logo'], "rb") as f:
		assert f.read(8) == b"\x89PNG\r\n\x1a\n"


@pytest.mark.skip(reason="bug: create_tomtom_match_logos keys the database by "
	"the first word of each MOTIF line, but tomtom-lite reports the whole "
	"line, so --lite reports never draw match logos for databases with "
	"alternate names such as JASPAR")
def test_create_tomtom_match_logos_alt_name(meme_db, tmp_path, plots):
	logos = create_tomtom_match_logos({'a': {'match_0': 'db_motif_2 NAME2'}},
		str(tmp_path), meme_db, 1)
	assert 'match_0_logo' in logos['a']


##


@pytest.mark.parametrize("matches,expected", [
	({'match_0': 'CTCF', 'match_1': 'CTCFL', 'match_2': 'ZNF143'},
		"CTCF;CTCFL;ZNF143"),
	({'match_0': 'MA0139.1 CTCF', 'match_1': None}, "MA0139.1 C"),
	({'match_0': ' JUND ', 'match_1': ' FOSL2'}, "JUND;FOSL2"),
	({'match_0': 'A'*25}, "A"*10),
	({'match_1': 'SP1'}, "SP1"),
	({'match_0': '', 'match_1': 'KLF4'}, "KLF4"),
	({'match_0': None, 'match_1': None, 'match_2': None}, "tag"),
	({}, "tag"),
])
def test_create_descriptive_names(matches, expected):
	assert create_descriptive_names({'tag': matches}) == {'tag': expected}


@pytest.mark.parametrize("top_n_matches,expected", [(1, "A"), (2, "A;B"),
	(3, "A;B;C"), (5, "A;B;C")])
def test_create_descriptive_names_top_n(top_n_matches, expected):
	matches = {'match_{}'.format(i): name for i, name in enumerate("ABCDE")}
	names = create_descriptive_names({'tag': matches}, top_n_matches)
	assert names == {'tag': expected}


def test_create_descriptive_names_multiple():
	tomtom_data = {'p0': {'match_0': 'X'}, 'p1': {}, 'p2': {'match_0': 'Y',
		'match_1': 'Z'}}
	assert create_descriptive_names(tomtom_data) == {'p0': 'X', 'p1': 'p1',
		'p2': 'Y;Z'}


def test_create_descriptive_names_empty():
	assert create_descriptive_names({}) == {}


##


def test_generate_descriptive_report(modisco_h5, tmp_path, plots, capsys):
	output_dir = tmp_path / "report"
	path = generate_descriptive_report(str(modisco_h5), str(output_dir),
		n_examples=2)

	assert path == os.path.join(str(output_dir), "report.html")
	assert capsys.readouterr().out == "Report generated: {}\n".format(path)
	assert sorted(os.listdir(output_dir)) == ["logos", "report.html",
		"seqlet_examples"]

	html = open(path).read()
	for tag in TAGS:
		assert 'id="pattern-{}"'.format(tag.replace(".", "-")) in html
	assert "tomtom_logos" not in os.listdir(output_dir)


def test_generate_descriptive_report_nested_dir(small_h5, tmp_path, plots):
	output_dir = tmp_path / "a" / "b" / "c"
	generate_descriptive_report(str(small_h5), str(output_dir), n_examples=1)
	assert (output_dir / "report.html").exists()


@pytest.mark.parametrize("n_examples", [1, 3])
def test_generate_descriptive_report_n_examples(small_h5, tmp_path, plots,
	n_examples):
	generate_descriptive_report(str(small_h5), str(tmp_path),
		n_examples=n_examples)
	files = os.listdir(tmp_path / "seqlet_examples" / "pos_patterns.pattern_0")
	assert len(files) == n_examples


def test_generate_descriptive_report_render(small_h5, tmp_path, capsys):
	path = generate_descriptive_report(str(small_h5), str(tmp_path),
		n_examples=1)

	html = open(path).read()
	assert 'id="pattern-pos_patterns-pattern_0"' in html
	assert html.count("data:image/png;base64,") >= 6


def test_generate_descriptive_report_ttl(modisco_h5, tmp_path, plots,
	meme_db):
	path = generate_descriptive_report(str(modisco_h5), str(tmp_path),
		meme_motif_db=str(meme_db), top_n_matches=2, ttl=True, n_examples=1)

	html = open(path).read()
	assert "db_motif_0 NAME0" in html
	assert "db_motif_1 NAME1" in html
	assert "P-value" in html
	assert os.path.isdir(tmp_path / "tomtom_logos")


@requires_tomtom
def test_generate_descriptive_report_tomtom(small_h5, tmp_path, plots,
	meme_db):
	path = generate_descriptive_report(str(small_h5), str(tmp_path),
		meme_motif_db=str(meme_db), top_n_matches=1, ttl=False, n_examples=1)

	html = open(path).read()
	assert "Q-value" in html


@pytest.mark.parametrize("suffix", ["./", "report/"])
def test_generate_descriptive_report_suffix(small_h5, tmp_path, plots, suffix):
	path = generate_descriptive_report(str(small_h5), str(tmp_path),
		img_path_suffix=suffix, n_examples=1)
	assert os.path.exists(path)
