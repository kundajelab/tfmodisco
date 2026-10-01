# test_fasta_writer.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import pytest

from modiscolite.fasta_writer import FASTAEntry
from modiscolite.fasta_writer import FASTAWriter

from numpy.testing import assert_raises


@pytest.fixture
def entries():
	return [FASTAEntry("chr1:10-20 dir=+ pattern_0.0", "ACGTACGTAC"),
		FASTAEntry("chr2:5-9 dir=- pattern_0.1", "TTGA"),
		FASTAEntry("seq3", "N")]


##


@pytest.mark.parametrize("header,sequence", [("seq", "ACGT"),
	("chr1:10-20 dir=+ pattern_0.0", "ACGTACGTAC"), ("", ""), ("x y z", "N"*50)])
def test_fasta_entry(header, sequence):
	entry = FASTAEntry(header, sequence)

	assert entry.header == header
	assert entry.sequence == sequence
	assert str(entry) == ">{}\n{}".format(header, sequence)
	assert repr(entry) == "FASTAEntry(header={}, sequence={})".format(header,
		sequence)


def test_fasta_entry_keywords():
	entry = FASTAEntry(sequence="ACGT", header="seq")
	assert str(entry) == ">seq\nACGT"


##


def test_fasta_writer_init():
	writer = FASTAWriter()
	assert writer.entries == []
	assert writer.get_output() == ""
	assert str(writer) == ""


def test_fasta_writer_independent():
	writer0 = FASTAWriter()
	writer0.add_pair(FASTAEntry("a", "A"))
	assert FASTAWriter().entries == []


@pytest.mark.xfail(strict=True, reason="bug: FASTAWriter.__init__ ignores its entries "
	"argument and always starts empty")
def test_fasta_writer_init_entries(entries):
	writer = FASTAWriter(entries=entries)
	assert writer.entries == entries


@pytest.mark.parametrize("n", [1, 2, 3])
def test_fasta_writer_add_pair(entries, n):
	writer = FASTAWriter()
	for entry in entries[:n]:
		writer.add_pair(entry)

	assert writer.entries == entries[:n]


def test_fasta_writer_get_output(entries):
	writer = FASTAWriter()
	for entry in entries:
		writer.add_pair(entry)

	assert writer.get_output() == (">chr1:10-20 dir=+ pattern_0.0\nACGTACGTAC\n"
		">chr2:5-9 dir=- pattern_0.1\nTTGA\n>seq3\nN")
	assert str(writer) == writer.get_output()


def test_fasta_writer_get_output_single():
	writer = FASTAWriter()
	writer.add_pair(FASTAEntry("a", "ACGT"))
	assert writer.get_output() == ">a\nACGT"


def test_fasta_writer_repr(entries):
	writer = FASTAWriter()
	writer.add_pair(entries[2])
	assert repr(writer) == "FASTAWriter(entries=[FASTAEntry(header=seq3, " \
		"sequence=N)])"


def test_fasta_writer_write(entries, tmp_path):
	writer = FASTAWriter()
	for entry in entries:
		writer.add_pair(entry)

	filename = tmp_path / "seqlets.fa"
	writer.write(filename)
	assert open(filename).read() == writer.get_output()


def test_fasta_writer_write_str_path(entries, tmp_path):
	writer = FASTAWriter()
	writer.add_pair(entries[0])

	filename = str(tmp_path / "seqlets.fa")
	writer.write(filename)
	assert open(filename).read() == ">chr1:10-20 dir=+ pattern_0.0\nACGTACGTAC"


def test_fasta_writer_write_empty(tmp_path):
	filename = tmp_path / "seqlets.fa"
	FASTAWriter().write(filename)
	assert open(filename).read() == ""


def test_fasta_writer_write_overwrites(entries, tmp_path):
	filename = tmp_path / "seqlets.fa"
	filename.write_text("old contents")

	writer = FASTAWriter()
	writer.add_pair(entries[1])
	writer.write(filename)
	assert open(filename).read() == ">chr2:5-9 dir=- pattern_0.1\nTTGA"


def test_fasta_writer_write_raises(tmp_path):
	filename = tmp_path / "missing" / "seqlets.fa"
	assert_raises(IOError, FASTAWriter().write, filename)
