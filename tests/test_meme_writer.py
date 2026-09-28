# test_meme_writer.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import pytest

from modiscolite.meme_writer import MEMEWriterMotif
from modiscolite.meme_writer import MEMEWriter
from modiscolite.meme_writer import array_to_string

from memelite.io import read_meme

from numpy.testing import assert_raises
from numpy.testing import assert_array_almost_equal


@pytest.fixture
def ppm():
	return numpy.random.RandomState(0).dirichlet(numpy.ones(4), size=6)


@pytest.fixture
def motif(ppm):
	return MEMEWriterMotif(name="motif_a", probability_matrix=ppm,
		source_sites=12, alphabet="ACGT")


def _writer(**kwargs):
	# Always pass motifs explicitly; the default list is shared between
	# instances.
	return MEMEWriter(memesuite_version='5', motifs=[], **kwargs)


##


@pytest.mark.parametrize("alphabet,expected", [("ACGT", 4), ("ACGU", 4),
	("AC", 2), ("ACDEFGHIKLMNPQRSTVWY", 20)])
def test_meme_writer_motif_alphabet_length(ppm, alphabet, expected):
	motif = MEMEWriterMotif("m", ppm, 1, alphabet)
	assert motif.alphabet_length == expected


@pytest.mark.parametrize("alphabet_length", [1, 4, 7])
def test_meme_writer_motif_alphabet_length_explicit(ppm, alphabet_length):
	motif = MEMEWriterMotif("m", ppm, 1, "ACGT", alphabet_length=alphabet_length)
	assert motif.alphabet_length == alphabet_length


def test_meme_writer_motif_init(motif, ppm):
	assert motif.name == "motif_a"
	assert motif.probability_matrix is ppm
	assert motif.source_sites == 12
	assert motif.e_value is None
	assert motif.url is None


@pytest.mark.parametrize("name", ["motif_a", "pos_patterns.pattern_0",
	"MA0139.1 CTCF", ""])
def test_meme_writer_motif_repr(ppm, name):
	motif = MEMEWriterMotif(name, ppm, 1, "ACGT")
	assert repr(motif) == "MEMEWriterMotif(name={})".format(name)


def test_meme_writer_motif_str(motif, ppm):
	lines = str(motif).split("\n")

	assert lines[0] == "MOTIF motif_a"
	assert lines[1] == "letter-probability matrix: alength= 4 w= 6 nsites= 12"
	assert len(lines) == 8
	for line, row in zip(lines[2:], ppm):
		assert line == " ".join("{:.6f}".format(x) for x in row)


@pytest.mark.parametrize("width", [1, 5, 30, 100])
def test_meme_writer_motif_str_width(width):
	ppm = numpy.random.RandomState(width).dirichlet(numpy.ones(4), size=width)
	lines = str(MEMEWriterMotif("m", ppm, 3, "ACGT")).split("\n")

	assert "w= {}".format(width) in lines[1]
	assert len(lines) == width + 2


@pytest.mark.parametrize("source_sites", [0, 1, 17, 20000])
def test_meme_writer_motif_str_source_sites(ppm, source_sites):
	motif = MEMEWriterMotif("m", ppm, source_sites, "ACGT")
	assert str(motif).split("\n")[1].endswith("nsites= {}".format(source_sites))


def test_meme_writer_motif_str_values():
	ppm = numpy.array([[0.1, 0.2, 0.3, 0.4], [1.0, 0.0, 0.0, 0.0]])
	motif = MEMEWriterMotif("x", ppm, 2, "ACGT")

	assert str(motif) == ("MOTIF x\n"
		"letter-probability matrix: alength= 4 w= 2 nsites= 2\n"
		"0.100000 0.200000 0.300000 0.400000\n"
		"1.000000 0.000000 0.000000 0.000000")


def test_meme_writer_motif_str_alphabet_length(ppm):
	motif = MEMEWriterMotif("m", ppm, 1, "ACGT", alphabet_length=5)
	assert "alength= 5 " in str(motif)


def test_meme_writer_motif_str_e_value(ppm):
	motif = MEMEWriterMotif("m", ppm, 1, "ACGT", e_value="1.2e-05")
	assert "E= 1.2e-05" in str(motif).split("\n")[1]


def test_meme_writer_motif_str_url(ppm):
	motif = MEMEWriterMotif("m", ppm, 1, "ACGT", url="http://x.org/m")
	assert "URL http://x.org/m" in str(motif)


@pytest.mark.skip(reason="bug: MEMEWriterMotif writes 'nsites= 12E= 0.01' "
	"with no space before E=, and appends 'URL ...' to the last matrix row "
	"without a newline")
def test_meme_writer_motif_str_e_value_url_format(ppm):
	motif = MEMEWriterMotif("m", ppm, 12, "ACGT", e_value="0.01",
		url="http://x.org/m")
	lines = str(motif).split("\n")

	assert lines[1].endswith("nsites= 12 E= 0.01")
	assert lines[-1] == "URL http://x.org/m"


##


def test_meme_writer_init():
	writer = MEMEWriter(memesuite_version='5', motifs=[])

	assert writer.memesuite_version == '5'
	assert writer.motifs == []
	assert writer.alphabet is None
	assert writer.background_frequencies is None
	assert writer.background_frequencies_source is None
	assert writer.strands is None


def test_meme_writer_init_motifs(motif):
	motifs = [motif]
	writer = MEMEWriter('5', motifs=motifs)
	assert writer.motifs is motifs


def test_meme_writer_init_none():
	writer = MEMEWriter('5', motifs=None)
	assert writer.motifs == []


@pytest.mark.skip(reason="bug: MEMEWriter's default motifs=[] is one list "
	"shared by every instance, so motifs added to one writer appear in all "
	"later writers, including every write_meme_from_h5 call in a process")
def test_meme_writer_default_motifs_independent(motif):
	writer0 = MEMEWriter('5')
	writer0.add_motif(motif)

	writer1 = MEMEWriter('5')
	assert writer1.motifs == []


@pytest.mark.parametrize("n", [1, 2, 5])
def test_meme_writer_add_motif(ppm, n):
	writer = _writer()
	motifs = [MEMEWriterMotif("m{}".format(i), ppm, 1, "ACGT")
		for i in range(n)]

	for motif in motifs:
		writer.add_motif(motif)

	assert writer.motifs == motifs


def test_meme_writer_get_output_minimal():
	assert _writer().get_output() == "MEME version 5\n\n"


@pytest.mark.parametrize("version", ['4', '5', '5.5.0'])
def test_meme_writer_get_output_version(version):
	writer = MEMEWriter(memesuite_version=version, motifs=[])
	assert writer.get_output() == "MEME version {}\n\n".format(version)


def test_meme_writer_get_output_alphabet():
	output = _writer(alphabet="ACGT").get_output()
	assert output == "MEME version 5\n\nALPHABET= ACGT\n\n"


def test_meme_writer_get_output_strands():
	output = _writer(strands="+ -").get_output()
	assert output == "MEME version 5\n\nstrands: + -\n\n"


def test_meme_writer_get_output_background():
	output = _writer(background_frequencies="A 0.25 C 0.25 G 0.25 T 0.25"
		).get_output()
	assert output == ("MEME version 5\n\nBackground letter frequencies\n"
		"A 0.25 C 0.25 G 0.25 T 0.25\n\n")


def test_meme_writer_get_output_background_source():
	output = _writer(background_frequencies="A 0.3 C 0.2 G 0.2 T 0.3",
		background_frequencies_source="genome").get_output()
	assert ("Background letter frequencies (from genome):\n"
		"A 0.3 C 0.2 G 0.2 T 0.3\n\n") in output


def test_meme_writer_get_output_background_source_only():
	# A source without frequencies writes nothing.
	output = _writer(background_frequencies_source="genome").get_output()
	assert output == "MEME version 5\n\n"


def test_meme_writer_get_output_order(motif):
	writer = _writer(alphabet="ACGT", strands="+ -",
		background_frequencies="A 0.25 C 0.25 G 0.25 T 0.25")
	writer.add_motif(motif)
	output = writer.get_output()

	positions = [output.index(s) for s in ["MEME version", "ALPHABET",
		"strands", "Background", "MOTIF"]]
	assert positions == sorted(positions)
	assert output.endswith(str(motif) + "\n\n")


def test_meme_writer_get_output_motifs(ppm):
	writer = _writer()
	for i in range(3):
		writer.add_motif(MEMEWriterMotif("m{}".format(i), ppm, 1, "ACGT"))

	output = writer.get_output()
	assert output.count("MOTIF ") == 3
	assert output.count("letter-probability matrix") == 3
	assert output.index("MOTIF m0") < output.index("MOTIF m1") < output.index(
		"MOTIF m2")


def test_meme_writer_write(motif, tmp_path):
	writer = _writer(alphabet="ACGT")
	writer.add_motif(motif)

	filename = tmp_path / "motifs.meme"
	writer.write(filename)
	assert open(filename).read() == writer.get_output()


def test_meme_writer_write_str_path(motif, tmp_path):
	writer = _writer()
	writer.add_motif(motif)

	filename = str(tmp_path / "motifs.meme")
	writer.write(filename)
	assert open(filename).read() == writer.get_output()


def test_meme_writer_write_overwrites(motif, tmp_path):
	filename = tmp_path / "motifs.meme"
	filename.write_text("old contents")

	_writer().write(filename)
	assert open(filename).read() == "MEME version 5\n\n"


def test_meme_writer_write_raises(tmp_path):
	filename = tmp_path / "missing" / "motifs.meme"
	assert_raises(IOError, _writer().write, filename)


@pytest.mark.parametrize("widths", [[5], [6, 12], [3, 8, 20, 1]])
def test_meme_writer_read_meme(tmp_path, widths):
	rng = numpy.random.RandomState(0)
	ppms = [rng.dirichlet(numpy.ones(4), size=w) for w in widths]

	writer = _writer(alphabet="ACGT",
		background_frequencies="A 0.25 C 0.25 G 0.25 T 0.25")
	for i, ppm in enumerate(ppms):
		writer.add_motif(MEMEWriterMotif("m{}".format(i), ppm, 1, "ACGT"))

	filename = tmp_path / "motifs.meme"
	writer.write(filename)
	motifs = read_meme(str(filename))

	assert list(motifs.keys()) == ["m{}".format(i) for i in range(len(ppms))]
	for pwm, ppm in zip(motifs.values(), ppms):
		assert_array_almost_equal(pwm.T, ppm, 6)


def test_meme_writer_repr(motif):
	writer = _writer(alphabet="ACGT")
	writer.add_motif(motif)

	assert repr(writer) == ("MEMEWriter(memesuite_version=5, "
		"motifs=[MEMEWriterMotif(name=motif_a)], alphabet=ACGT, "
		"background_frequencies=None, strands=None)")


##


@pytest.mark.parametrize("precision", [0, 1, 3, 6, 10])
def test_array_to_string_precision(precision):
	array = numpy.array([[0.123456789, 1.5], [-2.25, 3.0]])
	string = array_to_string(array, precision)
	fmt = "{:." + str(precision) + "f}"

	assert string == "\n".join(" ".join(fmt.format(x) for x in row)
		for row in array)


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (5, 4), (3, 20)])
def test_array_to_string_shape(shape):
	array = numpy.random.RandomState(0).randn(*shape)
	lines = array_to_string(array, 4).split("\n")

	assert len(lines) == shape[0]
	assert all(len(line.split(" ")) == shape[1] for line in lines)


def test_array_to_string_values():
	array = numpy.array([[0.5, -0.25], [1, 0]])
	assert array_to_string(array, 2) == "0.50 -0.25\n1.00 0.00"


def test_array_to_string_rounding():
	assert array_to_string(numpy.array([[0.125, 0.9999]]), 2) == "0.12 1.00"


def test_array_to_string_empty():
	assert array_to_string(numpy.zeros((0, 4)), 6) == ""


def test_array_to_string_list():
	assert array_to_string([[1, 2], [3, 4]], 1) == "1.0 2.0\n3.0 4.0"
