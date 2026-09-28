# test_bed_writer.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

from collections import OrderedDict

import pytest

from modiscolite.bed_writer import BEDTrackLine
from modiscolite.bed_writer import BEDRow
from modiscolite.bed_writer import BEDTrack
from modiscolite.bed_writer import BEDWriter

from numpy.testing import assert_raises


OPTIONAL = [("name", "row1"), ("score", 100), ("strand", "+"),
	("thick_start", 110), ("thick_end", 190), ("item_rgb", "0,0,255"),
	("block_count", 2), ("block_sizes", "10,20"), ("block_starts", "0,80")]


@pytest.fixture
def track_line():
	return BEDTrackLine(OrderedDict([("name", "pattern_0"),
		("description", "TF-MoDISco pattern")]))


@pytest.fixture
def rows():
	return [BEDRow("chr1", 100, 200, name="a", score=5, strand="+"),
		BEDRow("chr2", 0, 10, name="b", score=7, strand="-")]


##


@pytest.mark.parametrize("arguments,expected", [
	(OrderedDict(), 'track'),
	(OrderedDict([("name", "x")]), 'track name="x"'),
	(OrderedDict([("name", "x"), ("description", "a b c")]),
		'track name="x" description="a b c"'),
	(OrderedDict([("visibility", 2), ("color", "0,0,255")]),
		'track visibility="2" color="0,0,255"'),
])
def test_bed_track_line_str(arguments, expected):
	assert str(BEDTrackLine(arguments)) == expected


def test_bed_track_line_order():
	arguments = OrderedDict([("b", 1), ("a", 2), ("c", 3)])
	assert str(BEDTrackLine(arguments)) == 'track b="1" a="2" c="3"'


def test_bed_track_line_init(track_line):
	assert list(track_line.arguments.keys()) == ["name", "description"]


def test_bed_track_line_repr():
	line = BEDTrackLine(OrderedDict([("name", "x")]))
	assert repr(line) == "BEDTrackLine(arguments={})".format(
		OrderedDict([("name", "x")]))


def test_bed_track_line_dict():
	assert str(BEDTrackLine({"name": "x", "useScore": 1})) == \
		'track name="x" useScore="1"'


##


@pytest.mark.parametrize("chrom,start,end", [("chr1", 0, 10),
	("chrX", 12345, 12400), ("1", 5, 6), ("chr2_random", 100, 100)])
def test_bed_row_required(chrom, start, end):
	row = BEDRow(chrom, start, end)
	assert str(row) == "{}\t{}\t{}".format(chrom, start, end)


def test_bed_row_init():
	row = BEDRow("chr1", 100, 200)

	assert (row.chrom, row.chrom_start, row.chrom_end) == ("chr1", 100, 200)
	for field, _ in OPTIONAL:
		assert getattr(row, field) is None


@pytest.mark.parametrize("n", range(1, len(OPTIONAL) + 1))
def test_bed_row_optional(n):
	kwargs = dict(OPTIONAL[:n])
	row = BEDRow("chr1", 100, 200, **kwargs)

	expected = ["chr1", "100", "200"] + [str(v) for _, v in OPTIONAL[:n]]
	assert str(row) == "\t".join(expected)


@pytest.mark.parametrize("field,value", OPTIONAL)
def test_bed_row_single_optional(field, value):
	row = BEDRow("chr1", 100, 200, **{field: value})

	assert getattr(row, field) == value
	assert str(row) == "chr1\t100\t200\t{}".format(value)


def test_bed_row_dots():
	row = BEDRow("chr1", 100, 200, name=".", score=".", strand=".")
	assert str(row) == "chr1\t100\t200\t.\t.\t."


def test_bed_row_all():
	row = BEDRow("chr1", 100, 200, **dict(OPTIONAL))
	assert str(row) == "chr1\t100\t200\trow1\t100\t+\t110\t190\t0,0,255\t2" \
		"\t10,20\t0,80"
	assert len(str(row).split("\t")) == 12


def test_bed_row_numpy_types():
	import numpy
	row = BEDRow("chr1", numpy.int64(5), numpy.int64(15), score="3.5")
	assert str(row) == "chr1\t5\t15\t3.5"


##


def test_bed_track_init():
	track = BEDTrack()
	assert track.track_line is None
	assert track.rows == []
	assert str(track) == ""


def test_bed_track_independent_rows():
	track0 = BEDTrack()
	track0.add_row(BEDRow("chr1", 0, 1))
	assert BEDTrack().rows == []


def test_bed_track_rows(rows):
	track = BEDTrack(rows=rows)
	assert track.rows is rows
	assert str(track) == "chr1\t100\t200\ta\t5\t+\nchr2\t0\t10\tb\t7\t-\n"


def test_bed_track_track_line(track_line, rows):
	track = BEDTrack(track_line=track_line, rows=rows)
	assert str(track) == ('track name="pattern_0" description="TF-MoDISco '
		'pattern"\nchr1\t100\t200\ta\t5\t+\nchr2\t0\t10\tb\t7\t-\n')


def test_bed_track_track_line_only(track_line):
	track = BEDTrack(track_line=track_line)
	assert str(track) == str(track_line) + "\n"


@pytest.mark.parametrize("n", [1, 2, 10])
def test_bed_track_add_row(n):
	track = BEDTrack()
	for i in range(n):
		track.add_row(BEDRow("chr1", i, i+1))

	assert len(track.rows) == n
	assert str(track).count("\n") == n
	assert str(track).split("\n")[n-1] == "chr1\t{}\t{}".format(n-1, n)


def test_bed_track_repr(rows):
	track = BEDTrack(rows=rows[:1])
	assert repr(track).startswith("BEDTrack(track_line=None, rows=[")


##


def test_bed_writer_init():
	writer = BEDWriter()
	assert writer.tracks == []
	assert writer.get_output() == ""


def test_bed_writer_independent():
	writer0 = BEDWriter()
	writer0.add_track(BEDTrack())
	assert BEDWriter().tracks == []


def test_bed_writer_get_output(track_line, rows):
	writer = BEDWriter()
	writer.add_track(BEDTrack(track_line=track_line, rows=rows[:1]))
	writer.add_track(BEDTrack(rows=rows[1:]))

	assert writer.get_output() == ('track name="pattern_0" description='
		'"TF-MoDISco pattern"\nchr1\t100\t200\ta\t5\t+\n\n'
		'chr2\t0\t10\tb\t7\t-\n')


@pytest.mark.parametrize("n", [1, 2, 5])
def test_bed_writer_add_track(rows, n):
	writer = BEDWriter()
	for _ in range(n):
		writer.add_track(BEDTrack(rows=rows))

	assert len(writer.tracks) == n
	assert writer.get_output().count("chr1\t100\t200") == n


def test_bed_writer_write(track_line, rows, tmp_path):
	writer = BEDWriter()
	writer.add_track(BEDTrack(track_line=track_line, rows=rows))

	filename = tmp_path / "seqlets.bed"
	writer.write(filename)
	assert open(filename).read() == writer.get_output()


def test_bed_writer_write_str_path(rows, tmp_path):
	writer = BEDWriter()
	writer.add_track(BEDTrack(rows=rows))

	filename = str(tmp_path / "seqlets.bed")
	writer.write(filename)
	assert open(filename).read() == writer.get_output()


def test_bed_writer_write_raises(tmp_path):
	filename = tmp_path / "missing" / "seqlets.bed"
	assert_raises(IOError, BEDWriter().write, filename)
