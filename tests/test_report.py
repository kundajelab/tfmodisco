# test_report.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import shutil
import struct

from html.parser import HTMLParser

import numpy
import pandas
import pytest

from modiscolite.report import compute_per_position_ic
from modiscolite.report import write_meme_file
from modiscolite.report import fetch_tomtom_matches
from modiscolite.report import generate_tomtom_dataframe
from modiscolite.report import tomtomlite_dataframe
from modiscolite.report import path_to_image_html
from modiscolite.report import _plot_weights
from modiscolite.report import make_logo
from modiscolite.report import create_modisco_logos
from modiscolite.report import report_motifs

from modiscolite.io import save_hdf5

from memelite.io import read_meme

from .synthetic import random_track_set
from .synthetic import synthetic_patterns

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


requires_tomtom = pytest.mark.skipif(shutil.which("tomtom") is None,
	reason="the MEME suite tomtom executable is not installed")


def _png_size(filename):
	with open(filename, "rb") as f:
		header = f.read(24)

	assert header[:8] == b"\x89PNG\r\n\x1a\n"
	return struct.unpack(">II", header[16:24])


def _ic(ppm, background, pseudocount):
	p = (ppm + pseudocount) / (1 + pseudocount * len(background))
	return (numpy.log2(p) * ppm).sum(axis=1) - (numpy.log2(background) *
		background).sum()


class _TableParser(HTMLParser):
	"""Collect the cells of an HTML table, using the src of any image."""

	def __init__(self):
		super().__init__()
		self.rows, self._cell = [], None

	def handle_starttag(self, tag, attrs):
		if tag == 'tr':
			self.rows.append([])
		elif tag in ('td', 'th'):
			self._cell = ''
		elif tag == 'img' and self._cell is not None:
			self._cell += dict(attrs)['src']

	def handle_endtag(self, tag):
		if tag in ('td', 'th'):
			self.rows[-1].append(self._cell.strip())
			self._cell = None

	def handle_data(self, data):
		if self._cell is not None:
			self._cell += data


def _read_table(filename):
	"""Return the header and the rows of the table in an HTML file."""

	parser = _TableParser()
	parser.feed(open(filename).read())
	return parser.rows[0], parser.rows[1:]


def _column(filename, name):
	header, rows = _read_table(filename)
	return [row[header.index(name)] for row in rows]


@pytest.fixture
def small_h5(pos_patterns, neg_patterns, tmp_path):
	"""One positive and one negative pattern from the real results."""

	filename = tmp_path / "small.h5"
	save_hdf5(filename, pos_patterns[:1], neg_patterns[:1], window_size=300)
	return filename


@pytest.fixture
def ppm():
	return numpy.random.RandomState(0).dirichlet(numpy.ones(4), size=10)


##


@pytest.mark.parametrize("length", [1, 8, 30])
@pytest.mark.parametrize("pseudocount", [0.001, 0.1])
def test_compute_per_position_ic(length, pseudocount):
	ppm = numpy.random.RandomState(length).dirichlet(numpy.ones(4), size=length)
	background = numpy.full(4, 0.25)

	ic = compute_per_position_ic(ppm, background, pseudocount)
	assert ic.shape == (length,)
	assert_array_almost_equal(ic, _ic(ppm, background, pseudocount))


def test_compute_per_position_ic_background():
	ppm = numpy.random.RandomState(0).dirichlet(numpy.ones(4), size=5)
	background = numpy.array([0.3, 0.2, 0.2, 0.3])
	assert_array_almost_equal(compute_per_position_ic(ppm, background, 0.001),
		_ic(ppm, background, 0.001))


def test_compute_per_position_ic_uniform():
	ic = compute_per_position_ic(numpy.full((4, 4), 0.25), numpy.full(4, 0.25),
		0.001)
	assert_array_almost_equal(ic, numpy.zeros(4))


def test_compute_per_position_ic_unnormalized():
	# Unlike util.compute_per_position_ic, rows are not renormalized.
	ppm = numpy.random.RandomState(0).dirichlet(numpy.ones(4), size=5)
	ic0 = compute_per_position_ic(ppm, numpy.full(4, 0.25), 0.001)
	ic1 = compute_per_position_ic(ppm * 2, numpy.full(4, 0.25), 0.001)

	assert not numpy.allclose(ic0, ic1)
	assert_array_almost_equal(ic1, _ic(ppm * 2, numpy.full(4, 0.25), 0.001))


##


@pytest.mark.parametrize("width", [1, 6, 25])
def test_write_meme_file(tmp_path, width):
	ppm = numpy.random.RandomState(width).dirichlet(numpy.ones(4), size=width)
	filename = tmp_path / "motif.meme"
	write_meme_file(ppm, [0.25, 0.25, 0.25, 0.25], filename)

	lines = open(filename).read().split("\n")
	assert lines[:8] == ["MEME version 4", "", "ALPHABET= ACGT", "",
		"strands: + -", "",
		"Background letter frequencies (from unknown source):",
		"A 0.250 C 0.250 G 0.250 T 0.250"]
	assert lines[9] == "MOTIF 1 TEMP"
	assert lines[10] == ("letter-probability matrix: alength= 4 w= {} "
		"nsites= 1 E= 0e+0".format(width))
	assert lines[11:11+width] == ["%.5f %.5f %.5f %.5f" % tuple(row)
		for row in ppm]
	assert lines[11+width:] == ["URL", "", ""]


def test_write_meme_file_background(tmp_path, ppm):
	filename = tmp_path / "motif.meme"
	write_meme_file(ppm, numpy.array([0.3, 0.2, 0.2, 0.3]), filename)
	assert "A 0.300 C 0.200 G 0.200 T 0.300\n" in open(filename).read()


def test_write_meme_file_read_meme(tmp_path, ppm):
	filename = tmp_path / "motif.meme"
	write_meme_file(ppm, [0.25, 0.25, 0.25, 0.25], filename)

	motifs = read_meme(str(filename))
	assert list(motifs.keys()) == ["1 TEMP"]
	assert_array_almost_equal(motifs["1 TEMP"].T, ppm, 5)


##


def test_fetch_tomtom_matches_raises(tmp_path, ppm, meme_db):
	assert_raises(ValueError, fetch_tomtom_matches, ppm, ppm, False, tmp_path,
		"pattern_0", meme_db, tomtom_exec_path="not-a-real-tomtom-binary")


@requires_tomtom
def test_fetch_tomtom_matches(pos_patterns, tmp_path, meme_db):
	pattern = pos_patterns[0]
	results = fetch_tomtom_matches(pattern.sequence, pattern.contrib_scores,
		False, tmp_path, "pos_patterns.pattern_0", meme_db)

	assert isinstance(results, pandas.DataFrame)
	assert list(results.columns) == ["Target_ID", "q-value"]
	assert results["Target_ID"].iloc[0] == "db_motif_0"
	assert numpy.all(numpy.diff(results["q-value"]) >= 0) or len(results) < 3
	assert not os.path.exists(tmp_path / "tomtom")


@requires_tomtom
def test_fetch_tomtom_matches_write(pos_patterns, tmp_path, meme_db):
	pattern = pos_patterns[1]
	results = fetch_tomtom_matches(pattern.sequence, pattern.contrib_scores,
		True, tmp_path, "pos_patterns.pattern_1", meme_db)

	filename = tmp_path / "tomtom" / "pos_patterns.pattern_1.tomtom.tsv"
	assert filename.exists()
	assert results["Target_ID"].iloc[0] == "db_motif_1"
	assert "db_motif_1" in open(filename).read()


##


@requires_tomtom
@pytest.mark.parametrize("top_n_matches", [1, 3])
def test_generate_tomtom_dataframe(modisco_h5, tmp_path, meme_db,
	top_n_matches):
	df = generate_tomtom_dataframe(modisco_h5, tmp_path, meme_db, False,
		['pos_patterns', 'neg_patterns'], top_n_matches=top_n_matches)

	columns = []
	for i in range(top_n_matches):
		columns += ["match{}".format(i), "qval{}".format(i)]

	assert list(df.columns) == columns
	assert len(df) == 4
	# The negative patterns carry the same two motifs as the positive ones.
	assert list(df["match0"]) == ["db_motif_0", "db_motif_1", "db_motif_0",
		"db_motif_1"]


@requires_tomtom
def test_generate_tomtom_dataframe_groups(modisco_h5, tmp_path, meme_db):
	df = generate_tomtom_dataframe(modisco_h5, tmp_path, meme_db, False,
		['neg_patterns', 'missing_patterns'], top_n_matches=1)
	assert list(df["match0"]) == ["db_motif_0", "db_motif_1"]


##


@pytest.mark.parametrize("top_n_matches", [1, 2, 3, 5])
def test_tomtomlite_dataframe(modisco_h5, meme_db, top_n_matches):
	df = tomtomlite_dataframe(modisco_h5, meme_db,
		['pos_patterns', 'neg_patterns'], top_n_matches=top_n_matches)

	columns = []
	for i in range(top_n_matches):
		columns += ["match{}".format(i), "pval{}".format(i)]

	assert list(df.columns) == columns
	assert len(df) == 4


def test_tomtomlite_dataframe_matches(modisco_h5, meme_db):
	# Each positive pattern best matches the database entry made from its own
	# PPM, and each negative pattern the entry for the same motif.
	df = tomtomlite_dataframe(modisco_h5, meme_db,
		['pos_patterns', 'neg_patterns'], top_n_matches=3)

	assert list(df["match0"]) == ["db_motif_0 NAME0", "db_motif_1 NAME1",
		"db_motif_0 NAME0", "db_motif_1 NAME1"]
	assert numpy.all(df["pval0"] <= df["pval1"])
	assert numpy.all(df["pval1"] <= df["pval2"])
	assert numpy.all((df["pval0"] >= 0) & (df["pval0"] <= 1))


@pytest.mark.parametrize("groups,expected", [(['pos_patterns'], 2),
	(['neg_patterns'], 2), (['neg_patterns', 'pos_patterns'], 4),
	(['pos_patterns', 'missing'], 2)])
def test_tomtomlite_dataframe_groups(modisco_h5, meme_db, groups, expected):
	df = tomtomlite_dataframe(modisco_h5, meme_db, groups, top_n_matches=1)
	assert len(df) == expected


def test_tomtomlite_dataframe_group_order(modisco_h5, meme_db):
	df0 = tomtomlite_dataframe(modisco_h5, meme_db,
		['pos_patterns', 'neg_patterns'], top_n_matches=2)
	df1 = tomtomlite_dataframe(modisco_h5, meme_db,
		['neg_patterns', 'pos_patterns'], top_n_matches=2)

	assert df0["pval0"].nunique() == 4
	pandas.testing.assert_frame_equal(df0,
		df1.iloc[[2, 3, 0, 1]].reset_index(drop=True))


@pytest.mark.parametrize("trim_threshold", [0.0, 0.3, 0.6, 1.0])
def test_tomtomlite_dataframe_trim_threshold(modisco_h5, meme_db,
	trim_threshold):
	df = tomtomlite_dataframe(modisco_h5, meme_db, ['pos_patterns'],
		top_n_matches=2, trim_threshold=trim_threshold)
	assert len(df) == 2
	assert df["match0"].str.startswith("db_motif_").all()


##


@pytest.mark.parametrize("path", ["a.png", "./trimmed_logos/x.cwm.fwd.png",
	"/abs/path with space.png"])
def test_path_to_image_html(path):
	assert path_to_image_html(path) == '<img src="{}" width="240" >'.format(
		path)


##


# A single position with a negative sum gives identical y-limits.
@pytest.mark.filterwarnings("ignore:Attempting to set identical low and high")
@pytest.mark.parametrize("length", [1, 10, 50])
def test_plot_weights(tmp_path, length):
	array = numpy.random.RandomState(length).randn(length, 4)
	filename = tmp_path / "logo.png"
	_plot_weights(array, filename)

	assert _png_size(filename) == (1000, 300)


@pytest.mark.parametrize("figsize", [(4, 2), (8, 1.2), (12, 3)])
def test_plot_weights_figsize(tmp_path, figsize):
	filename = tmp_path / "logo.png"
	_plot_weights(numpy.random.RandomState(0).randn(10, 4), filename,
		figsize=figsize)

	assert _png_size(filename) == (int(figsize[0]*100), int(figsize[1]*100))


@pytest.mark.parametrize("clamp", [True, False])
def test_plot_weights_clamp(tmp_path, clamp):
	filename = tmp_path / "logo.png"
	_plot_weights(numpy.abs(numpy.random.RandomState(0).randn(10, 4)),
		filename, clamp=clamp)
	assert filename.stat().st_size > 0


def test_plot_weights_closes_figure(tmp_path):
	import matplotlib.pyplot as plt
	n = len(plt.get_fignums())
	_plot_weights(numpy.ones((5, 4)), tmp_path / "logo.png")
	assert len(plt.get_fignums()) == n


def test_plot_weights_raises(tmp_path):
	assert_raises(ValueError, _plot_weights, numpy.ones((5, 3)),
		tmp_path / "logo.png")


##


def test_make_logo(tmp_path, ppm):
	make_logo("motif_a", tmp_path, {"motif_a": ppm, "motif_b": ppm})

	assert [p.name for p in tmp_path.iterdir()] == ["motif_a.png"]
	_png_size(tmp_path / "motif_a.png")


def test_make_logo_na(tmp_path, ppm):
	assert make_logo("NA", tmp_path, {"motif_a": ppm}) is None
	assert list(tmp_path.iterdir()) == []


def test_make_logo_raises(tmp_path, ppm):
	assert_raises(KeyError, make_logo, "motif_c", tmp_path, {"motif_a": ppm})


##


def test_create_modisco_logos(small_h5, tmp_path):
	logo_dir = tmp_path / "logos"
	logo_dir.mkdir()
	tags = create_modisco_logos(small_h5, logo_dir, 0.3,
		['pos_patterns', 'neg_patterns'])

	assert tags == ["pos_patterns.pattern_0", "neg_patterns.pattern_0"]
	assert sorted(p.name for p in logo_dir.iterdir()) == [
		"neg_patterns.pattern_0.cwm.fwd.png",
		"neg_patterns.pattern_0.cwm.rev.png",
		"pos_patterns.pattern_0.cwm.fwd.png",
		"pos_patterns.pattern_0.cwm.rev.png"]


@pytest.mark.parametrize("groups,expected", [(['pos_patterns'],
	["pos_patterns.pattern_0"]), (['missing'], []),
	(['neg_patterns', 'missing'], ["neg_patterns.pattern_0"])])
def test_create_modisco_logos_groups(small_h5, tmp_path, groups, expected):
	logo_dir = tmp_path / "logos"
	logo_dir.mkdir()

	tags = create_modisco_logos(small_h5, logo_dir, 0.3, groups)
	assert tags == expected
	assert len(list(logo_dir.iterdir())) == 2 * len(expected)


def test_create_modisco_logos_order(tmp_path):
	# Patterns are ordered by their index rather than by name.
	track_set = random_track_set(n=6, length=40)
	h5 = tmp_path / "results.h5"
	save_hdf5(h5, synthetic_patterns(track_set, 11, 2, 8), None, 40)

	logo_dir = tmp_path / "logos"
	logo_dir.mkdir()

	import modiscolite.report
	calls = []
	original = modiscolite.report._plot_weights
	modiscolite.report._plot_weights = lambda array, path, **kw: calls.append(
		path)
	try:
		tags = create_modisco_logos(h5, logo_dir, 0.3, ['pos_patterns'])
	finally:
		modiscolite.report._plot_weights = original

	assert tags == ["pos_patterns.pattern_{}".format(i) for i in range(11)]
	assert len(calls) == 22


def test_create_modisco_logos_raises(small_h5, tmp_path):
	assert_raises(FileNotFoundError, create_modisco_logos, small_h5,
		tmp_path / "missing", 0.3, ['pos_patterns'])


##


def test_report_motifs(small_h5, tmp_path):
	output_dir = tmp_path / "report"
	report_motifs(small_h5, output_dir, "./", None, False)

	assert sorted(p.name for p in output_dir.iterdir()) == ["motifs.html",
		"trimmed_logos"]
	assert len(list((output_dir / "trimmed_logos").iterdir())) == 4

	html = output_dir / "motifs.html"
	header, rows = _read_table(html)
	assert header == ["pattern", "num_seqlets", "modisco_cwm_fwd",
		"modisco_cwm_rev"]
	assert rows == [
		["pos_patterns.pattern_0", "74",
			"./trimmed_logos/pos_patterns.pattern_0.cwm.fwd.png",
			"./trimmed_logos/pos_patterns.pattern_0.cwm.rev.png"],
		["neg_patterns.pattern_0", "77",
			"./trimmed_logos/neg_patterns.pattern_0.cwm.fwd.png",
			"./trimmed_logos/neg_patterns.pattern_0.cwm.rev.png"]]


@pytest.mark.parametrize("suffix", ["./", "report/", "/abs/"])
def test_report_motifs_suffix(small_h5, tmp_path, suffix):
	output_dir = tmp_path / "report"
	report_motifs(small_h5, output_dir, suffix, None, False)

	html = open(output_dir / "motifs.html").read()
	assert '<img src="{}" width="240" >'.format(os.path.join(suffix,
		"trimmed_logos", "pos_patterns.pattern_0.cwm.fwd.png")) in html


def test_report_motifs_existing_dir(small_h5, tmp_path):
	output_dir = tmp_path / "report"
	(output_dir / "trimmed_logos").mkdir(parents=True)

	report_motifs(small_h5, output_dir, "./", None, False)
	assert (output_dir / "motifs.html").exists()


def test_report_motifs_ttl(small_h5, tmp_path, meme_db):
	output_dir = tmp_path / "report"
	report_motifs(small_h5, output_dir, "./", meme_db, False, top_n_matches=1,
		ttl=True)

	html = output_dir / "motifs.html"
	header, rows = _read_table(html)
	assert header == ["pattern", "num_seqlets", "modisco_cwm_fwd",
		"modisco_cwm_rev", "match0", "pval0", "match0_logo"]
	assert _column(html, "match0") == ["db_motif_0 NAME0", "db_motif_0 NAME0"]
	assert _column(html, "match0_logo") == ["./db_motif_0 NAME0.png"] * 2
	assert all(0 <= float(p) <= 1 for p in _column(html, "pval0"))
	assert (output_dir / "db_motif_0 NAME0.png").exists()


@requires_tomtom
def test_report_motifs_tomtom(small_h5, tmp_path, meme_db):
	output_dir = tmp_path / "report"
	report_motifs(small_h5, output_dir, "./", meme_db, True, top_n_matches=1)

	html = output_dir / "motifs.html"
	header, rows = _read_table(html)
	assert header == ["pattern", "num_seqlets", "modisco_cwm_fwd",
		"modisco_cwm_rev", "match0", "qval0", "match0_logo"]
	assert _column(html, "match0") == ["db_motif_0", "db_motif_0"]
	assert (output_dir / "db_motif_0.png").exists()
	assert (output_dir / "tomtom" / "pos_patterns.pattern_0.tomtom.tsv"
		).exists()


def test_report_motifs_raises(small_h5, tmp_path):
	assert_raises(FileNotFoundError, report_motifs, small_h5,
		tmp_path / "a" / "b", "./", None, False)
