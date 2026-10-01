# test_io.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import h5py
import numpy
import scipy.special
import pytest

from modiscolite.core import TrackSet
from modiscolite.io import save_pattern
from modiscolite.io import save_hdf5
from modiscolite.io import write_meme_from_h5
from modiscolite.io import write_bed_from_h5
from modiscolite.io import write_fasta_from_h5
from modiscolite.io import convert
from modiscolite.io import convert_new_to_old

from modiscolite.util import MemeDataType

from memelite.io import read_meme

from .conftest import DATA_DIR
from .synthetic import random_track_set
from .synthetic import make_pattern
from .synthetic import synthetic_patterns
from .synthetic import write_peaks

from numpy.testing import assert_raises
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


pytestmark = pytest.mark.usefixtures("meme_writer_default")


@pytest.fixture
def small_ts():
	return random_track_set(n=6, length=40)


@pytest.fixture
def small_h5(small_ts, tmp_path):
	"""A results file with one positive and one negative pattern of known
	seqlets and a window size of 40."""

	pos = make_pattern(small_ts, [(0, 2, 12, False), (3, 20, 30, True)])
	neg = make_pattern(small_ts, [(5, 30, 40, False)])

	filename = tmp_path / "small.h5"
	save_hdf5(filename, [pos], [neg], window_size=40)
	return filename


@pytest.fixture
def peaks(tmp_path):
	filename = tmp_path / "peaks.bed"
	rows = write_peaks(filename, 300)
	return filename, rows


def _meme_matrices(filename):
	return {name: pwm.T for name, pwm in read_meme(str(filename)).items()}


def _parse_bed(output):
	tracks = {}
	for line in output.strip().split("\n"):
		if line.startswith("track"):
			name = line.split('"')[1]
			tracks.setdefault(name, [])
		elif line:
			tracks[name].append(line.split("\t"))

	return tracks


def _parse_fasta(output):
	lines = output.split("\n")
	return list(zip(lines[::2], lines[1::2]))


##


def test_save_pattern(small_ts, tmp_path):
	pattern = make_pattern(small_ts, [(0, 2, 12, False), (3, 20, 30, True),
		(4, 0, 10, False)])

	with h5py.File(tmp_path / "p.h5", "w") as f:
		save_pattern(pattern, f.create_group("pattern_0"))

	with h5py.File(tmp_path / "p.h5", "r") as f:
		grp = f["pattern_0"]
		assert_array_almost_equal(grp["sequence"][:], pattern.sequence)
		assert_array_almost_equal(grp["contrib_scores"][:],
			pattern.contrib_scores)
		assert_array_almost_equal(grp["hypothetical_contribs"][:],
			pattern.hypothetical_contribs)

		seqlets = grp["seqlets"]
		assert_array_equal(seqlets["n_seqlets"][:], [3])
		assert_array_equal(seqlets["start"][:], [2, 20, 0])
		assert_array_equal(seqlets["end"][:], [12, 30, 10])
		assert_array_equal(seqlets["example_idx"][:], [0, 3, 4])
		assert_array_equal(seqlets["is_revcomp"][:], [False, True, False])
		assert seqlets["sequence"].shape == (3, 10, 4)
		assert_array_almost_equal(seqlets["contrib_scores"][1],
			pattern.seqlets[1].contrib_scores)
		assert_array_almost_equal(seqlets["hypothetical_contribs"][2],
			pattern.seqlets[2].hypothetical_contribs)


def test_save_pattern_no_subpatterns(small_ts, tmp_path):
	pattern = make_pattern(small_ts, [(0, 2, 12, False)])

	with h5py.File(tmp_path / "p.h5", "w") as f:
		save_pattern(pattern, f.create_group("pattern_0"))
		assert sorted(f["pattern_0"].keys()) == ["contrib_scores",
			"hypothetical_contribs", "seqlets", "sequence"]


def test_save_pattern_subpatterns(pos_patterns, tmp_path):
	pattern = pos_patterns[0]

	with h5py.File(tmp_path / "p.h5", "w") as f:
		save_pattern(pattern, f.create_group("pattern_0"))

	with h5py.File(tmp_path / "p.h5", "r") as f:
		grp = f["pattern_0"]
		names = sorted(k for k in grp.keys() if k.startswith("subpattern_"))
		assert names == ["subpattern_{}".format(k) for k in
			sorted(pattern.subcluster_to_subpattern.keys())]

		for k, subpattern in pattern.subcluster_to_subpattern.items():
			sub = grp["subpattern_{}".format(k)]
			assert sub["seqlets"]["n_seqlets"][0] == len(subpattern.seqlets)
			assert_array_almost_equal(sub["sequence"][:], subpattern.sequence)
			assert_array_equal(sub["seqlets"]["start"][:],
				[s.start for s in subpattern.seqlets])


##


@pytest.mark.parametrize("n_pos,n_neg", [(1, 1), (3, 0), (0, 2), (4, 5)])
def test_save_hdf5(small_ts, tmp_path, n_pos, n_neg):
	pos = synthetic_patterns(small_ts, n_pos, 3, 10, random_state=0)
	neg = synthetic_patterns(small_ts, n_neg, 2, 8, random_state=1)

	filename = tmp_path / "results.h5"
	save_hdf5(filename, pos, neg, window_size=40)

	with h5py.File(filename, "r") as f:
		assert f.attrs["window_size"] == 40
		assert sorted(f["pos_patterns"].keys()) == sorted("pattern_{}".format(i)
			for i in range(n_pos))
		assert sorted(f["neg_patterns"].keys()) == sorted("pattern_{}".format(i)
			for i in range(n_neg))

		for i, pattern in enumerate(pos):
			assert_array_almost_equal(f["pos_patterns/pattern_{}/sequence"
				.format(i)][:], pattern.sequence)
		for i, pattern in enumerate(neg):
			assert_array_almost_equal(f["neg_patterns/pattern_{}/contrib_scores"
				.format(i)][:], pattern.contrib_scores)


@pytest.mark.parametrize("pos,neg,groups", [(None, None, []),
	("pos", None, ["pos_patterns"]), (None, "neg", ["neg_patterns"])])
def test_save_hdf5_none(small_ts, tmp_path, pos, neg, groups):
	patterns = synthetic_patterns(small_ts, 1, 2, 10)
	filename = tmp_path / "results.h5"
	save_hdf5(filename, patterns if pos else None, patterns if neg else None,
		window_size=40)

	with h5py.File(filename, "r") as f:
		assert sorted(f.keys()) == groups


@pytest.mark.parametrize("window_size", [20, 400, 1000])
def test_save_hdf5_window_size(small_ts, tmp_path, window_size):
	filename = tmp_path / "results.h5"
	save_hdf5(filename, None, None, window_size=window_size)

	with h5py.File(filename, "r") as f:
		assert f.attrs["window_size"] == window_size


def test_save_hdf5_real(modisco_h5, pos_patterns, neg_patterns):
	with h5py.File(modisco_h5, "r") as f:
		assert f.attrs["window_size"] == 300
		assert sorted(f["pos_patterns"].keys()) == ["pattern_0", "pattern_1"]
		assert sorted(f["neg_patterns"].keys()) == ["pattern_0", "pattern_1"]

		for name, patterns in [("pos_patterns", pos_patterns),
			("neg_patterns", neg_patterns)]:
			for i, pattern in enumerate(patterns):
				grp = f[name]["pattern_{}".format(i)]
				assert grp["seqlets/n_seqlets"][0] == len(pattern.seqlets)
				assert grp["seqlets/sequence"].shape == (len(pattern.seqlets),
					50, 4)
				assert_array_almost_equal(grp["contrib_scores"][:],
					pattern.contrib_scores)


def test_save_hdf5_overwrites(small_ts, tmp_path):
	filename = tmp_path / "results.h5"
	save_hdf5(filename, synthetic_patterns(small_ts, 3, 2, 10), None, 40)
	save_hdf5(filename, synthetic_patterns(small_ts, 1, 2, 10), None, 20)

	with h5py.File(filename, "r") as f:
		assert list(f["pos_patterns"].keys()) == ["pattern_0"]
		assert f.attrs["window_size"] == 20


##


@pytest.mark.parametrize("datatype", list(MemeDataType))
def test_write_meme_from_h5(modisco_h5, tmp_path, datatype):
	filename = tmp_path / "motifs.meme"
	write_meme_from_h5(modisco_h5, datatype, filename, is_quiet=True)

	matrices = _meme_matrices(filename)
	assert list(matrices.keys()) == ["pos_patterns.pattern_0",
		"pos_patterns.pattern_1", "neg_patterns.pattern_0",
		"neg_patterns.pattern_1"]

	with h5py.File(modisco_h5, "r") as f:
		for name, matrix in matrices.items():
			grp = f[name.replace(".", "/")]
			sequence = grp["sequence"][:]
			contrib = grp["contrib_scores"][:]
			hyp = grp["hypothetical_contribs"][:]

			expected = {
				MemeDataType.PFM: sequence / sequence.sum(axis=1, keepdims=True),
				MemeDataType.CWM: contrib,
				MemeDataType.hCWM: hyp,
				MemeDataType.CWM_PFM: scipy.special.softmax(contrib, axis=1),
				MemeDataType.hCWM_PFM: scipy.special.softmax(hyp, axis=1)
			}[datatype]

			assert matrix.shape == (50, 4)
			assert_array_almost_equal(matrix, expected, 6)


@pytest.mark.parametrize("datatype", [MemeDataType.PFM, MemeDataType.CWM_PFM,
	MemeDataType.hCWM_PFM])
def test_write_meme_from_h5_probabilities(modisco_h5, tmp_path, datatype):
	filename = tmp_path / "motifs.meme"
	write_meme_from_h5(modisco_h5, datatype, filename, is_quiet=True)

	for matrix in _meme_matrices(filename).values():
		assert_array_almost_equal(matrix.sum(axis=1), numpy.ones(50), 5)
		assert numpy.all(matrix >= 0)


def test_write_meme_from_h5_header(modisco_h5, tmp_path):
	filename = tmp_path / "motifs.meme"
	write_meme_from_h5(modisco_h5, MemeDataType.PFM, filename, is_quiet=True)

	lines = open(filename).read().split("\n")
	assert lines[:7] == ["MEME version 5", "", "ALPHABET= ACGT", "",
		"Background letter frequencies", "A 0.25 C 0.25 G 0.25 T 0.25", ""]
	assert lines[7] == "MOTIF pos_patterns.pattern_0"
	assert lines[8] == "letter-probability matrix: alength= 4 w= 50 nsites= 1"


def test_write_meme_from_h5_stdout(modisco_h5, tmp_path, capsys):
	filename = tmp_path / "motifs.meme"
	write_meme_from_h5(modisco_h5, MemeDataType.CWM, filename, is_quiet=False)

	assert capsys.readouterr().out == open(filename).read() + "\n"


def test_write_meme_from_h5_stdout_only(modisco_h5, tmp_path, capsys):
	write_meme_from_h5(modisco_h5, MemeDataType.PFM, None, is_quiet=False)

	output = capsys.readouterr().out
	assert output.startswith("MEME version 5\n")
	assert output.count("MOTIF ") == 4
	assert list(tmp_path.iterdir()) == []


def test_write_meme_from_h5_quiet(modisco_h5, tmp_path, capsys):
	write_meme_from_h5(modisco_h5, MemeDataType.PFM, tmp_path / "m.meme",
		is_quiet=True)
	assert capsys.readouterr().out == ""


def test_write_meme_from_h5_neither(modisco_h5, tmp_path, capsys):
	write_meme_from_h5(modisco_h5, MemeDataType.PFM, None, is_quiet=True)
	assert capsys.readouterr().out == ""
	assert list(tmp_path.iterdir()) == []


def test_write_meme_from_h5_small(small_h5, tmp_path):
	filename = tmp_path / "motifs.meme"
	write_meme_from_h5(small_h5, MemeDataType.hCWM, filename, is_quiet=True)

	matrices = _meme_matrices(filename)
	assert list(matrices.keys()) == ["pos_patterns.pattern_0",
		"neg_patterns.pattern_0"]
	assert all(m.shape == (10, 4) for m in matrices.values())


def test_write_meme_from_h5_missing_group(small_ts, tmp_path):
	filename = tmp_path / "results.h5"
	save_hdf5(filename, synthetic_patterns(small_ts, 2, 3, 10), None, 40)

	write_meme_from_h5(filename, MemeDataType.PFM, tmp_path / "m.meme", True)
	assert list(_meme_matrices(tmp_path / "m.meme").keys()) == [
		"pos_patterns.pattern_0", "pos_patterns.pattern_1"]


def test_write_meme_from_h5_skips_datasets(small_h5, tmp_path):
	with h5py.File(small_h5, "a") as f:
		f["pos_patterns"].create_dataset("not_a_pattern", data=numpy.ones(3))

	write_meme_from_h5(small_h5, MemeDataType.PFM, tmp_path / "m.meme", True)
	assert list(_meme_matrices(tmp_path / "m.meme").keys()) == [
		"pos_patterns.pattern_0", "neg_patterns.pattern_0"]


def test_write_meme_from_h5_order(small_ts, tmp_path):
	# h5py lists groups in name order, so pattern_10 precedes pattern_2.
	filename = tmp_path / "results.h5"
	save_hdf5(filename, synthetic_patterns(small_ts, 12, 2, 10), None, 40)

	write_meme_from_h5(filename, MemeDataType.PFM, tmp_path / "m.meme", True)
	names = list(_meme_matrices(tmp_path / "m.meme").keys())
	assert names[:4] == ["pos_patterns.pattern_0", "pos_patterns.pattern_1",
		"pos_patterns.pattern_10", "pos_patterns.pattern_11"]
	assert len(names) == 12


def test_write_meme_from_h5_repeated(modisco_h5, tmp_path, meme_writer_default):
	write_meme_from_h5(modisco_h5, MemeDataType.PFM, tmp_path / "a.meme", True)
	meme_writer_default.clear()
	write_meme_from_h5(modisco_h5, MemeDataType.PFM, tmp_path / "b.meme", True)

	assert open(tmp_path / "a.meme").read() == open(tmp_path / "b.meme").read()


@pytest.mark.parametrize("datatype", ["PFM", "CWM", None, 3])
def test_write_meme_from_h5_raises(modisco_h5, datatype):
	assert_raises(ValueError, write_meme_from_h5, modisco_h5, datatype, None,
		True)


##


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S01: BED start must be zero-based')
def test_write_bed_from_h5_small(small_h5, tmp_path, capsys):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	filename = tmp_path / "seqlets.bed"

	write_bed_from_h5(small_h5, peaks, filename, '*', None, is_quiet=True)

	# Peak i lies on chroms[i % 3], spans 1000*i + 17 to 1000*i + 517 with its
	# center at 1000*i + 267, and the window of 40 starts 20 bp before the
	# center.
	assert open(filename).read() == (
		'track name="pattern_0" description="TF-MoDISco pattern \'pattern_0\' '
		'on the positive strand."\n'
		'chr1\t249\t259\tpattern_0.0\t100\t+\n'
		'chr1\t3267\t3277\tpattern_0.1\t103\t-\n'
		'\n'
		'track name="pattern_0" description="TF-MoDISco pattern \'pattern_0\' '
		'on the positive strand."\n'
		'chrX\t5277\t5287\tpattern_0.0\t105\t+\n')
	assert capsys.readouterr().out == ""


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S01: BED start must be zero-based')
def test_write_bed_from_h5(modisco_h5, peaks, tmp_path):
	peaks_file, rows = peaks
	filename = tmp_path / "seqlets.bed"
	write_bed_from_h5(modisco_h5, peaks_file, filename, '*', None, True)

	tracks = _parse_bed(open(filename).read())
	assert list(tracks.keys()) == ["pattern_0", "pattern_1"]

	with h5py.File(modisco_h5, "r") as f:
		n = sum(f[g][p]["seqlets/n_seqlets"][0] for g in f.keys()
			for p in f[g].keys())
		lines = [l for l in open(filename).read().split("\n")
			if l and not l.startswith("track")]
		assert len(lines) == n

		seqlets = f["pos_patterns/pattern_0/seqlets"]
		for i, line in enumerate(lines[:len(seqlets["start"])]):
			chrom, start, end, name, score, strand = line.split("\t")
			idx = seqlets["example_idx"][i]
			peak = rows[idx].split("\t")
			center = (int(peak[1]) + int(peak[2])) // 2

			assert chrom == peak[0]
			assert score == peak[4]
			assert name == "pattern_0.{}".format(i)
			assert int(start) == center - 150 + seqlets["start"][i]
			assert int(end) == center - 150 + seqlets["end"][i]
			assert strand == ("-" if seqlets["is_revcomp"][i] else "+")


@pytest.mark.parametrize("window_size", [40, 100, 400])
@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S01: BED start must be zero-based')
def test_write_bed_from_h5_window_size(small_h5, tmp_path, window_size):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	filename = tmp_path / "seqlets.bed"

	write_bed_from_h5(small_h5, peaks, filename, '*', window_size, True)
	first = open(filename).read().split("\n")[1].split("\t")

	assert int(first[1]) == 267 - window_size // 2 + 2
	assert int(first[2]) == 267 - window_size // 2 + 12


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S01: BED start must be zero-based')
def test_write_bed_from_h5_chroms(small_ts, tmp_path):
	# Seqlet example indices refer to the peaks on the chosen chromosomes.
	peaks = tmp_path / "peaks.bed"
	rows = write_peaks(peaks, 18, chroms=('chr1', 'chr2', 'chr3'))
	pattern = make_pattern(small_ts, [(i, 2, 12, False) for i in range(6)])

	h5 = tmp_path / "results.h5"
	save_hdf5(h5, [pattern], None, window_size=40)

	filename = tmp_path / "seqlets.bed"
	write_bed_from_h5(h5, peaks, filename, ['chr2'], None, True)
	lines = open(filename).read().strip().split("\n")[1:]

	chr2 = [r for r in rows if r.startswith("chr2\t")]
	assert len(lines) == 6
	for line, peak in zip(lines, chr2):
		fields, peak = line.split("\t"), peak.split("\t")
		center = (int(peak[1]) + int(peak[2])) // 2

		assert fields[0] == "chr2"
		assert int(fields[1]) == center - 20 + 2
		assert fields[4] == peak[4]


def test_write_bed_from_h5_chroms_multiple(small_ts, tmp_path):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 12, chroms=('chr1', 'chr2', 'chr3', 'chr4'))
	pattern = make_pattern(small_ts, [(i, 2, 12, False) for i in range(6)])

	h5 = tmp_path / "results.h5"
	save_hdf5(h5, [pattern], None, window_size=40)

	filename = tmp_path / "seqlets.bed"
	write_bed_from_h5(h5, peaks, filename, ['chr4', 'chr1'], None, True)
	chroms = [l.split("\t")[0] for l in open(filename).read().strip().split(
		"\n")[1:]]
	assert chroms == ["chr1", "chr4", "chr1", "chr4", "chr1", "chr4"]


def test_write_bed_from_h5_stdout(small_h5, tmp_path, capsys):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	filename = tmp_path / "seqlets.bed"

	write_bed_from_h5(small_h5, peaks, filename, '*', None, is_quiet=False)
	assert capsys.readouterr().out == open(filename).read() + "\n"


def test_write_bed_from_h5_stdout_only(small_h5, tmp_path, capsys):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)

	write_bed_from_h5(small_h5, peaks, None, '*', None, is_quiet=False)
	assert capsys.readouterr().out.count("pattern_0.") == 3
	assert sorted(p.name for p in tmp_path.iterdir()) == ["peaks.bed",
		"small.h5"]


def test_write_bed_from_h5_no_window(small_ts, tmp_path, capsys):
	h5 = tmp_path / "results.h5"
	with h5py.File(h5, "w") as f:
		save_pattern(make_pattern(small_ts, [(0, 2, 12, False)]),
			f.create_group("pos_patterns").create_group("pattern_0"))

	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)

	assert_raises(SystemExit, write_bed_from_h5, h5, peaks, None, '*', None,
		True)
	assert "window_size must be specified" in capsys.readouterr().out


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S01: BED start must be zero-based')
def test_write_bed_from_h5_no_window_explicit(small_ts, tmp_path):
	h5 = tmp_path / "results.h5"
	with h5py.File(h5, "w") as f:
		save_pattern(make_pattern(small_ts, [(0, 2, 12, False)]),
			f.create_group("pos_patterns").create_group("pattern_0"))

	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)

	filename = tmp_path / "seqlets.bed"
	write_bed_from_h5(h5, peaks, filename, '*', 40, True)
	assert "chr1\t249\t259\tpattern_0.0\t100\t+" in open(filename).read()


def test_write_bed_from_h5_raises_index(small_h5, tmp_path):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 3)
	assert_raises(IndexError, write_bed_from_h5, small_h5, peaks, None, '*',
		None, True)


##


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='B11: FASTA must include the complete source span')
def test_write_fasta_from_h5_small(small_ts, tmp_path):
	h5 = tmp_path / "results.h5"
	pattern = make_pattern(small_ts, [(0, 2, 12, False), (3, 20, 30, True)])
	save_hdf5(h5, [pattern], None, window_size=40)

	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	sequences = tmp_path / "sequences.npz"
	numpy.savez(sequences, small_ts.one_hot.transpose(0, 2, 1))

	filename = tmp_path / "seqlets.fa"
	write_fasta_from_h5(h5, peaks, sequences, filename, '*', None, True)

	bases = numpy.array(list("ACGT"))
	seq0 = "".join(bases[small_ts.one_hot[0, 2:12].argmax(axis=1)])
	seq1 = "".join(bases[small_ts.one_hot[3, 20:30].argmax(axis=1)])

	assert open(filename).read() == (">chr1:250-259 dir=+ pattern_0.0\n{}\n"
		">chr1:3268-3277 dir=- pattern_0.1\n{}".format(seq0, seq1))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='B11: FASTA must include the complete source span')
def test_write_fasta_from_h5(modisco_h5, peaks, tmp_path):
	peaks_file, rows = peaks
	filename = tmp_path / "seqlets.fa"
	write_fasta_from_h5(modisco_h5, peaks_file, DATA_DIR / "sequences.npz",
		filename, '*', None, True)

	entries = _parse_fasta(open(filename).read())
	sequences = numpy.load(DATA_DIR / "sequences.npz")["arr_0"]
	bases = numpy.array(list("ACGT"))

	with h5py.File(modisco_h5, "r") as f:
		n = sum(f[g][p]["seqlets/n_seqlets"][0] for g in f.keys()
			for p in f[g].keys())
		assert len(entries) == n

		seqlets = f["pos_patterns/pattern_0/seqlets"]
		for i, (header, sequence) in enumerate(entries[:len(seqlets["start"])]):
			idx, start, end = (seqlets["example_idx"][i], seqlets["start"][i],
				seqlets["end"][i])
			peak = rows[idx].split("\t")
			center = (int(peak[1]) + int(peak[2])) // 2
			strand = "-" if seqlets["is_revcomp"][i] else "+"

			assert header == ">{}:{}-{} dir={} pattern_0.{}".format(peak[0],
				center - 150 + start + 1, center - 150 + end, strand, i)
			assert sequence == "".join(bases[sequences[idx, :, start:end]
				.argmax(axis=0)])


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='B11: FASTA must include the complete source span')
def test_write_fasta_from_h5_forward_strand(modisco_h5, peaks, tmp_path):
	# Sequences are always read from the forward strand.
	peaks_file, _ = peaks
	filename = tmp_path / "seqlets.fa"
	write_fasta_from_h5(modisco_h5, peaks_file, DATA_DIR / "sequences.npz",
		filename, '*', None, True)

	entries = _parse_fasta(open(filename).read())
	with h5py.File(modisco_h5, "r") as f:
		seqlets = f["pos_patterns/pattern_0/seqlets"]
		rc = numpy.where(seqlets["is_revcomp"][:])[0]
		assert len(rc) > 0

		comp = str.maketrans("ACGT", "TGCA")
		for i in rc[:5]:
			sequence = entries[i][1]
			stored = seqlets["sequence"][i].argmax(axis=1)
			stored = "".join(numpy.array(list("ACGT"))[stored])
			assert sequence == stored[::-1].translate(comp)


def test_write_fasta_from_h5_chroms(small_ts, tmp_path):
	h5 = tmp_path / "results.h5"
	pattern = make_pattern(small_ts, [(i, 2, 12, False) for i in range(6)])
	save_hdf5(h5, [pattern], None, window_size=40)

	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 18, chroms=('chr1', 'chr2', 'chr3'))
	sequences = tmp_path / "sequences.npz"
	numpy.savez(sequences, small_ts.one_hot.transpose(0, 2, 1))

	filename = tmp_path / "seqlets.fa"
	write_fasta_from_h5(h5, peaks, sequences, filename, ['chr3'], None, True)
	headers = [h for h, _ in _parse_fasta(open(filename).read())]

	assert headers[0] == ">chr3:2250-2259 dir=+ pattern_0.0"
	assert all(h.startswith(">chr3:") for h in headers)


@pytest.mark.parametrize("window_size", [40, 60])
def test_write_fasta_from_h5_window_size(small_h5, small_ts, tmp_path,
	window_size):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	sequences = tmp_path / "sequences.npz"
	one_hot = numpy.zeros((6, 4, window_size), dtype=small_ts.one_hot.dtype)
	one_hot[:, 0, :] = 1
	one_hot[:, :, :40] = small_ts.one_hot.transpose(0, 2, 1)
	numpy.savez(sequences, one_hot)

	filename = tmp_path / "seqlets.fa"
	write_fasta_from_h5(small_h5, peaks, sequences, filename, '*', window_size,
		True)
	header = open(filename).read().split("\n")[0]
	assert header == ">chr1:{}-{} dir=+ pattern_0.0".format(
		267 - window_size // 2 + 3, 267 - window_size // 2 + 12)


def test_write_fasta_from_h5_stdout(small_h5, small_ts, tmp_path, capsys):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	sequences = tmp_path / "sequences.npz"
	numpy.savez(sequences, small_ts.one_hot.transpose(0, 2, 1))

	filename = tmp_path / "seqlets.fa"
	write_fasta_from_h5(small_h5, peaks, sequences, filename, '*', None, False)
	assert capsys.readouterr().out == open(filename).read() + "\n"


def test_write_fasta_from_h5_quiet(small_h5, small_ts, tmp_path, capsys):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	sequences = tmp_path / "sequences.npz"
	numpy.savez(sequences, small_ts.one_hot.transpose(0, 2, 1))

	write_fasta_from_h5(small_h5, peaks, sequences, None, '*', None, True)
	assert capsys.readouterr().out == ""


def test_write_fasta_from_h5_row_mismatch(small_h5, small_ts, tmp_path,
	capsys):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 7)
	sequences = tmp_path / "sequences.npz"
	numpy.savez(sequences, small_ts.one_hot.transpose(0, 2, 1))

	assert_raises(SystemExit, write_fasta_from_h5, small_h5, peaks, sequences,
		None, '*', None, True)
	assert "does not match the number of peaks" in capsys.readouterr().out


def test_write_fasta_from_h5_missing_key(small_h5, small_ts, tmp_path,
	capsys):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	sequences = tmp_path / "sequences.npz"
	numpy.savez(sequences, seqs=small_ts.one_hot.transpose(0, 2, 1))

	assert_raises(SystemExit, write_fasta_from_h5, small_h5, peaks, sequences,
		None, '*', None, True)
	assert "does not\ncontain an 'arr_0' key" in capsys.readouterr().out


def test_write_fasta_from_h5_no_window(small_ts, tmp_path):
	h5 = tmp_path / "results.h5"
	with h5py.File(h5, "w") as f:
		save_pattern(make_pattern(small_ts, [(0, 2, 12, False)]),
			f.create_group("pos_patterns").create_group("pattern_0"))

	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	sequences = tmp_path / "sequences.npz"
	numpy.savez(sequences, small_ts.one_hot.transpose(0, 2, 1))

	assert_raises(ValueError, write_fasta_from_h5, h5, peaks, sequences, None,
		'*', None, True)


@pytest.mark.xfail(strict=True, reason="bug: write_fasta_from_h5 reads positions "
	"start+1 through end-1, one base fewer than the start+1 to end span its "
	"header reports")
def test_write_fasta_from_h5_length(small_h5, small_ts, tmp_path):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	sequences = tmp_path / "sequences.npz"
	numpy.savez(sequences, small_ts.one_hot.transpose(0, 2, 1))

	filename = tmp_path / "seqlets.fa"
	write_fasta_from_h5(small_h5, peaks, sequences, filename, '*', None, True)
	for header, sequence in _parse_fasta(open(filename).read()):
		start, end = header.split(":")[1].split(" ")[0].split("-")
		assert len(sequence) == int(end) - int(start) + 1


@pytest.mark.xfail(strict=True, reason="bug: write_fasta_from_h5 indexes the sequences "
	"with window-relative seqlet positions, so sequences longer than the "
	"window, such as the full-length input given to `modisco motifs`, "
	"return bases from the wrong offset")
def test_write_fasta_from_h5_long_sequences(small_ts, tmp_path):
	# HDF5 coordinates refer to the 20-base central window, not the full input.
	window_ts = TrackSet(small_ts.one_hot[:, 10:30],
		small_ts.contrib_scores[:, 10:30], small_ts.hypothetical_contribs[:, 10:30])
	pattern = make_pattern(window_ts, [(0, 2, 12, False)])
	h5 = tmp_path / "results.h5"
	save_hdf5(h5, [pattern], None, window_size=20)

	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	sequences = tmp_path / "sequences.npz"
	numpy.savez(sequences, small_ts.one_hot.transpose(0, 2, 1))

	filename = tmp_path / "seqlets.fa"
	write_fasta_from_h5(h5, peaks, sequences, filename, '*', None, True)

	# The 20 bp window is the center of each 40 bp sequence.
	bases = numpy.array(list("ACGT"))
	expected = "".join(bases[small_ts.one_hot[0, 12:22].argmax(axis=1)])
	observed = open(filename).read().split("\n")[1]
	# The suffix isolates the window offset from the separate missing-first-base bug.
	assert observed[-4:] == expected[-4:]


##


def _read_new(filename):
	"""Return {group/pattern: (sequence, contrib, hyp, start, end, idx, rc)}."""

	data = {}
	with h5py.File(filename, "r") as f:
		for group in f.keys():
			for name, pattern in f[group].items():
				seqlets = pattern["seqlets"]
				data["{}/{}".format(group, name)] = (pattern["sequence"][:],
					pattern["contrib_scores"][:],
					pattern["hypothetical_contribs"][:], seqlets["start"][:],
					seqlets["end"][:], seqlets["example_idx"][:],
					seqlets["is_revcomp"][:])

	return data


def test_convert_new_to_old(modisco_h5, tmp_path):
	old = tmp_path / "old.h5"
	convert_new_to_old(modisco_h5, old)

	with h5py.File(old, "r") as f:
		assert [x.decode() for x in f["task_names"][:]] == ["task0"]
		grp = f["metacluster_idx_to_submetacluster_results"]
		assert sorted(grp.keys()) == ["metacluster_0", "metacluster_1"]

		result = grp["metacluster_0/seqlets_to_patterns_result"]
		assert result.attrs["success"]
		assert result.attrs["total_time_taken"] == 1.0

		patterns = result["patterns"]
		assert [x.decode() for x in patterns["all_pattern_names"][:]] == [
			"pattern_0", "pattern_1"]

		with h5py.File(modisco_h5, "r") as g:
			new = g["pos_patterns/pattern_0"]
			assert_array_equal(patterns["pattern_0/sequence/fwd"][:],
				new["sequence"][:])
			assert_array_equal(patterns["pattern_0/task0_contrib_scores/fwd"][:],
				new["contrib_scores"][:])
			assert_array_equal(
				patterns["pattern_0/task0_hypothetical_contribs/fwd"][:],
				new["hypothetical_contribs"][:])

			seqlets = [x.decode() for x in
				patterns["pattern_0/seqlets_and_alnmts/seqlets"][:]]
			n = new["seqlets/n_seqlets"][0]
			assert len(seqlets) == n
			assert seqlets[0] == "example:{},start:{},end:{},rc:{}".format(
				new["seqlets/example_idx"][0], new["seqlets/start"][0],
				new["seqlets/end"][0], new["seqlets/is_revcomp"][0])

			alnmts = patterns["pattern_0/seqlets_and_alnmts/alnmts"]
			assert alnmts.dtype == numpy.int32
			assert_array_equal(alnmts[:], numpy.zeros(n))


def test_convert_new_to_old_metacluster_seqlets(modisco_h5, tmp_path):
	old = tmp_path / "old.h5"
	convert_new_to_old(modisco_h5, old)

	with h5py.File(old, "r") as f, h5py.File(modisco_h5, "r") as g:
		for new_name, old_name in [("pos_patterns", "metacluster_0"),
			("neg_patterns", "metacluster_1")]:
			seqlets = f["metacluster_idx_to_submetacluster_results"][old_name][
				"seqlets"]
			n = sum(g[new_name][p]["seqlets/n_seqlets"][0]
				for p in g[new_name].keys())
			assert len(seqlets) == n


def test_convert_new_to_old_numeric_order(small_ts, tmp_path):
	new = tmp_path / "new.h5"
	save_hdf5(new, synthetic_patterns(small_ts, 12, 2, 10), None, 40)

	old = tmp_path / "old.h5"
	convert_new_to_old(new, old)

	with h5py.File(old, "r") as f:
		names = f["metacluster_idx_to_submetacluster_results/metacluster_0/"
			"seqlets_to_patterns_result/patterns/all_pattern_names"][:]
		assert [x.decode() for x in names] == ["pattern_{}".format(i)
			for i in range(12)]


def test_convert_new_to_old_pos_only(small_ts, tmp_path):
	new = tmp_path / "new.h5"
	save_hdf5(new, synthetic_patterns(small_ts, 2, 2, 10), None, 40)

	old = tmp_path / "old.h5"
	convert_new_to_old(new, old)

	with h5py.File(old, "r") as f:
		assert list(f["metacluster_idx_to_submetacluster_results"].keys()) == [
			"metacluster_0"]


def test_convert_new_to_old_empty_group(small_ts, tmp_path):
	new = tmp_path / "new.h5"
	save_hdf5(new, [], None, 40)

	old = tmp_path / "old.h5"
	convert_new_to_old(new, old)

	with h5py.File(old, "r") as f:
		mc = f["metacluster_idx_to_submetacluster_results/metacluster_0"]
		assert "patterns" not in mc["seqlets_to_patterns_result"]
		assert len(mc["seqlets"]) == 0


@pytest.mark.xfail(strict=True, reason="bug: convert_new_to_old looks for subpatterns "
	"named subcluster_*, but save_pattern names them subpattern_*, so "
	"subpatterns are never converted")
def test_convert_new_to_old_subpatterns(modisco_h5, tmp_path):
	old = tmp_path / "old.h5"
	convert_new_to_old(modisco_h5, old)

	with h5py.File(old, "r") as f:
		pattern = f["metacluster_idx_to_submetacluster_results/metacluster_0/"
			"seqlets_to_patterns_result/patterns/pattern_0"]
		assert "subcluster_to_subpattern" in pattern


def test_convert_round_trip(modisco_h5, tmp_path):
	old = tmp_path / "old.h5"
	new = tmp_path / "new.h5"
	convert_new_to_old(modisco_h5, old)
	convert(old, new)

	expected = _read_new(modisco_h5)
	observed = _read_new(new)

	assert observed.keys() == expected.keys()
	for key in expected:
		for x, y in zip(observed[key], expected[key]):
			assert_array_equal(x, y)


@pytest.mark.parametrize("n_pos,n_neg", [(1, 0), (0, 1), (3, 2), (11, 1)])
def test_convert_round_trip_synthetic(small_ts, tmp_path, n_pos, n_neg):
	original = tmp_path / "original.h5"
	save_hdf5(original, synthetic_patterns(small_ts, n_pos, 3, 10) or None,
		synthetic_patterns(small_ts, n_neg, 2, 12, random_state=5) or None, 40)

	old = tmp_path / "old.h5"
	new = tmp_path / "new.h5"
	convert_new_to_old(original, old)
	convert(old, new)

	expected = _read_new(original)
	observed = _read_new(new)

	assert observed.keys() == expected.keys()
	for key in expected:
		for x, y in zip(observed[key], expected[key]):
			assert_array_equal(x, y)


def test_convert_n_seqlets(modisco_h5, tmp_path):
	old = tmp_path / "old.h5"
	new = tmp_path / "new.h5"
	convert_new_to_old(modisco_h5, old)
	convert(old, new)

	with h5py.File(new, "r") as f, h5py.File(modisco_h5, "r") as g:
		for group in g.keys():
			for name in g[group].keys():
				assert f[group][name]["seqlets/n_seqlets"][0] == g[group][name][
					"seqlets/n_seqlets"][0]


def _write_old(filename, subpatterns=True, metaclusters=(0, 1)):
	"""Write a file in the original TF-MoDISco format by hand."""

	rng = numpy.random.RandomState(0)
	with h5py.File(filename, "w") as f:
		root = f.create_group("metacluster_idx_to_submetacluster_results")
		for mc in metaclusters:
			result = root.create_group("metacluster_{}".format(mc)).create_group(
				"seqlets_to_patterns_result")
			patterns = result.create_group("patterns")
			patterns.create_dataset("all_pattern_names", data=["pattern_0"])

			pattern = patterns.create_group("pattern_0")
			for name in ["sequence", "task0_contrib_scores",
				"task0_hypothetical_contribs"]:
				pattern.create_dataset(name + "/fwd", data=rng.randn(8, 4))

			pattern.create_dataset("seqlets_and_alnmts/seqlets", data=[
				"example:3,start:10,end:18,rc:False",
				"example:7,start:2,end:10,rc:True"])

			if subpatterns:
				sub = pattern.create_group("subcluster_to_subpattern")
				sub.create_dataset("subcluster_names", data=["subcluster_0"])
				s0 = sub.create_group("subcluster_0")
				for name in ["sequence", "task0_contrib_scores",
					"task0_hypothetical_contribs"]:
					s0.create_dataset(name + "/fwd", data=rng.randn(8, 4))
				s0.create_dataset("seqlets_and_alnmts/seqlets", data=[
					"example:3,start:10,end:18,rc:False"])


def test_convert(tmp_path):
	old = tmp_path / "old.h5"
	new = tmp_path / "new.h5"
	_write_old(old)
	convert(old, new)

	with h5py.File(new, "r") as f, h5py.File(old, "r") as g:
		assert sorted(f.keys()) == ["neg_patterns", "pos_patterns"]
		pattern = f["pos_patterns/pattern_0"]
		old_pattern = g["metacluster_idx_to_submetacluster_results/"
			"metacluster_0/seqlets_to_patterns_result/patterns/pattern_0"]

		assert_array_equal(pattern["sequence"][:],
			old_pattern["sequence/fwd"][:])
		assert_array_equal(pattern["seqlets/start"][:], [10, 2])
		assert_array_equal(pattern["seqlets/end"][:], [18, 10])
		assert_array_equal(pattern["seqlets/example_idx"][:], [3, 7])
		assert_array_equal(pattern["seqlets/is_revcomp"][:], [False, True])
		assert_array_equal(pattern["seqlets/n_seqlets"][:], [2])


def test_convert_subpatterns(tmp_path):
	old = tmp_path / "old.h5"
	new = tmp_path / "new.h5"
	_write_old(old)
	convert(old, new)

	with h5py.File(new, "r") as f:
		for group in ["pos_patterns", "neg_patterns"]:
			sub = f[group]["pattern_0/subcluster_0"]
			assert sub["sequence"].shape == (8, 4)
			assert_array_equal(sub["seqlets/n_seqlets"][:], [1])


def test_convert_no_subpatterns(tmp_path):
	old = tmp_path / "old.h5"
	new = tmp_path / "new.h5"
	_write_old(old, subpatterns=False)
	convert(old, new)

	with h5py.File(new, "r") as f:
		assert sorted(f["pos_patterns/pattern_0"].keys()) == ["contrib_scores",
			"hypothetical_contribs", "seqlets", "sequence"]


@pytest.mark.parametrize("metaclusters,groups", [((0,), ["pos_patterns"]),
	((1,), ["neg_patterns"]), ((0, 1), ["neg_patterns", "pos_patterns"])])
def test_convert_metaclusters(tmp_path, metaclusters, groups):
	old = tmp_path / "old.h5"
	new = tmp_path / "new.h5"
	_write_old(old, metaclusters=metaclusters)
	convert(old, new)

	with h5py.File(new, "r") as f:
		assert sorted(f.keys()) == groups
