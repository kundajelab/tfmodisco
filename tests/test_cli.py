# test_cli.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os

import h5py
import numpy
import pytest

from click.testing import CliRunner

from modiscolite.cli import cli
from modiscolite.cli import _split_chroms

from modiscolite.io import save_hdf5
from modiscolite.util import MemeDataType

from memelite.io import read_meme

from .conftest import DATA_DIR
from .synthetic import random_track_set
from .synthetic import make_pattern
from .synthetic import write_peaks

from numpy.testing import assert_array_equal


pytestmark = pytest.mark.usefixtures("meme_writer_default")

SEQUENCES = str(DATA_DIR / "sequences.npz")
ATTRIBUTIONS = str(DATA_DIR / "attributions.npz")

COMMANDS = ['convert', 'convert-backward', 'meme', 'motifs', 'report',
	'report-simple', 'seqlet-bed', 'seqlet-fasta']


def _invoke(args):
	result = CliRunner().invoke(cli, [str(arg) for arg in args])
	return result


def _motifs(tmp_path, *args, n=60):
	output = tmp_path / "modisco_results.h5"
	result = _invoke(["motifs", "-s", SEQUENCES, "-a", ATTRIBUTIONS, "-n", n,
		"-o", output] + list(args))
	return result, output


def _patterns(filename):
	with h5py.File(filename, "r") as f:
		return {"{}/{}".format(g, p): f[g][p]["sequence"][:]
			for g in f.keys() for p in f[g].keys()}


@pytest.fixture
def report_h5(tmp_path):
	"""A single 6 bp pattern, which keeps the rendered reports fast."""

	track_set = random_track_set(n=6, length=40)
	pattern = make_pattern(track_set, [(0, 2, 8, False), (3, 20, 26, True)])

	filename = tmp_path / "report.h5"
	save_hdf5(filename, [pattern], None, window_size=40)
	return filename


@pytest.fixture
def small_h5(tmp_path):
	track_set = random_track_set(n=6, length=40)
	pos = make_pattern(track_set, [(0, 2, 8, False), (3, 20, 26, True)])
	neg = make_pattern(track_set, [(5, 30, 36, False)])

	filename = tmp_path / "small.h5"
	save_hdf5(filename, [pos], [neg], window_size=40)
	return filename


##


@pytest.mark.parametrize("chroms,expected", [("*", "*"), ("chr1", ["chr1"]),
	("chr1,chr2,chrX", ["chr1", "chr2", "chrX"]), ("1,2,X", ["1", "2", "X"]),
	("", [""]), ("chr1,", ["chr1", ""])])
def test_split_chroms(chroms, expected):
	assert _split_chroms(chroms) == expected


##


@pytest.mark.cmd
def test_cli_help():
	result = _invoke(["--help"])

	assert result.exit_code == 0
	assert "TF-MoDISco is a motif detection algorithm" in result.output
	for command in COMMANDS:
		assert command in result.output


@pytest.mark.cmd
def test_cli_commands():
	assert sorted(cli.commands.keys()) == COMMANDS


@pytest.mark.cmd
@pytest.mark.parametrize("command", COMMANDS)
def test_cli_command_help(command):
	result = _invoke([command, "--help"])
	assert result.exit_code == 0
	assert "Usage:" in result.output


@pytest.mark.cmd
def test_cli_unknown_command():
	assert _invoke(["not-a-command"]).exit_code == 2


##


@pytest.mark.cmd
def test_motifs(tmp_path):
	result, output = _motifs(tmp_path, "-w", 300, n=100)

	assert result.exit_code == 0, result.output
	assert result.output == ""
	with h5py.File(output, "r") as f:
		assert f.attrs["window_size"] == 300
		assert list(f.keys()) == ["pos_patterns"]
		assert len(f["pos_patterns"]) > 0
		for pattern in f["pos_patterns"].values():
			assert pattern["sequence"].shape == (50, 4)


@pytest.mark.cmd
@pytest.mark.parametrize("window", [100, 200, 300])
def test_motifs_window(tmp_path, window):
	result, output = _motifs(tmp_path, "-w", window)

	assert result.exit_code == 0, result.output
	with h5py.File(output, "r") as f:
		assert f.attrs["window_size"] == window
		for pattern in f["pos_patterns"].values():
			assert numpy.all(pattern["seqlets/end"][:] <= window)


@pytest.mark.cmd
def test_motifs_default_window(tmp_path):
	# The default window of 400 is longer than the 300 bp fixture.
	result, output = _motifs(tmp_path)

	assert result.exit_code == 1
	assert isinstance(result.exception, ValueError)
	assert "Window (400) cannot be longer than the sequence length" in \
		str(result.exception)
	assert not output.exists()


@pytest.mark.cmd
@pytest.mark.skip(reason="bug: a window longer than the sequences gives a "
	"negative start offset, which wraps around, so the error reports a "
	"sequence length of 50 instead of 300")
def test_motifs_default_window_message(tmp_path):
	result, output = _motifs(tmp_path)
	assert "sequence length (300)" in str(result.exception)


@pytest.mark.cmd
@pytest.mark.skip(reason="bug: calculate_window_offsets returns a window one "
	"bp short when the window is odd, so an odd -w always fails with a "
	"misleading 'Window cannot be longer than the sequence length' error")
def test_motifs_odd_window(tmp_path):
	result, output = _motifs(tmp_path, "-w", 201)
	assert result.exit_code == 0, result.output


@pytest.mark.cmd
@pytest.mark.parametrize("n", [40, 60, 100])
def test_motifs_max_seqlets(tmp_path, n):
	result, output = _motifs(tmp_path, "-w", 300, "-v", n=n)

	assert result.exit_code == 0, result.output
	assert result.output == "Using {} positive seqlets\n".format(n)
	with h5py.File(output, "r") as f:
		total = sum(p["seqlets/n_seqlets"][0] for p in f["pos_patterns"].values())
		assert total <= n


@pytest.mark.cmd
def test_motifs_max_seqlets_required(tmp_path):
	result = _invoke(["motifs", "-s", SEQUENCES, "-a", ATTRIBUTIONS, "-o",
		tmp_path / "out.h5"])
	assert result.exit_code == 2
	assert "Missing option '-n'" in result.output


@pytest.mark.cmd
@pytest.mark.parametrize("args,length", [([], 50), (["-t", 20], 40),
	(["-g", 5], 40), (["-j", 3], 56), (["-t", 16, "-g", 2, "-j", 1], 22)])
def test_motifs_pattern_length(tmp_path, args, length):
	result, output = _motifs(tmp_path, "-w", 300, *args)

	assert result.exit_code == 0, result.output
	with h5py.File(output, "r") as f:
		for pattern in f["pos_patterns"].values():
			assert pattern["sequence"].shape == (length, 4)


@pytest.mark.cmd
@pytest.mark.parametrize("args", [["-z", 15], ["-z", 12, "-f", 3], ["-l", 1],
	["-l", 3]])
def test_motifs_options(tmp_path, args):
	result, output = _motifs(tmp_path, "-w", 300, *args)

	assert result.exit_code == 0, result.output
	assert len(_patterns(output)) > 0


@pytest.mark.cmd
def test_motifs_npy(tmp_path):
	sequences = tmp_path / "sequences.npy"
	attributions = tmp_path / "attributions.npy"
	numpy.save(sequences, numpy.load(SEQUENCES)["arr_0"])
	numpy.save(attributions, numpy.load(ATTRIBUTIONS)["arr_0"])

	output0 = tmp_path / "npz.h5"
	output1 = tmp_path / "npy.h5"
	_invoke(["motifs", "-s", SEQUENCES, "-a", ATTRIBUTIONS, "-n", 60, "-w", 300,
		"-o", output0])
	result = _invoke(["motifs", "-s", sequences, "-a", attributions, "-n", 60,
		"-w", 300, "-o", output1])

	assert result.exit_code == 0, result.output
	patterns0, patterns1 = _patterns(output0), _patterns(output1)
	assert patterns0.keys() == patterns1.keys()
	for key in patterns0:
		assert_array_equal(patterns0[key], patterns1[key])


@pytest.mark.cmd
def test_motifs_h5_hyp_scores(tmp_path):
	sequences = numpy.load(SEQUENCES)["arr_0"].transpose(0, 2, 1)
	attributions = numpy.load(ATTRIBUTIONS)["arr_0"].transpose(0, 2, 1)

	h5 = tmp_path / "scores.h5"
	with h5py.File(h5, "w") as f:
		f.create_dataset("hyp_scores", data=attributions)
		f.create_dataset("input_seqs", data=sequences)

	output0 = tmp_path / "npz.h5"
	output1 = tmp_path / "h5.h5"
	_invoke(["motifs", "-s", SEQUENCES, "-a", ATTRIBUTIONS, "-n", 60, "-w", 200,
		"-o", output0])
	result = _invoke(["motifs", "-i", h5, "-n", 60, "-w", 200, "-o", output1])

	assert result.exit_code == 0, result.output
	patterns0, patterns1 = _patterns(output0), _patterns(output1)
	assert patterns0.keys() == patterns1.keys()
	for key in patterns0:
		assert_array_equal(patterns0[key], patterns1[key])


@pytest.mark.cmd
def test_motifs_h5_shap(tmp_path):
	h5 = tmp_path / "scores.h5"
	with h5py.File(h5, "w") as f:
		f.create_dataset("shap/seq", data=numpy.load(ATTRIBUTIONS)["arr_0"])
		f.create_dataset("raw/seq", data=numpy.load(SEQUENCES)["arr_0"])

	output0 = tmp_path / "npz.h5"
	output1 = tmp_path / "h5.h5"
	_invoke(["motifs", "-s", SEQUENCES, "-a", ATTRIBUTIONS, "-n", 60, "-w", 200,
		"-o", output0])
	result = _invoke(["motifs", "-i", h5, "-n", 60, "-w", 200, "-o", output1])

	assert result.exit_code == 0, result.output
	patterns0, patterns1 = _patterns(output0), _patterns(output1)
	assert patterns0.keys() == patterns1.keys()
	for key in patterns0:
		assert_array_equal(patterns0[key], patterns1[key])


@pytest.mark.cmd
def test_motifs_negative(tmp_path):
	attributions = tmp_path / "attributions.npz"
	numpy.savez(attributions, -numpy.load(ATTRIBUTIONS)["arr_0"])

	output = tmp_path / "out.h5"
	result = _invoke(["motifs", "-s", SEQUENCES, "-a", attributions, "-n", 60,
		"-w", 300, "-o", output, "-v"])

	assert result.exit_code == 0, result.output
	assert result.output == "Extracted 60 negative seqlets\n"
	with h5py.File(output, "r") as f:
		assert list(f.keys()) == ["neg_patterns"]


@pytest.mark.cmd
def test_motifs_default_output(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	result = _invoke(["motifs", "-s", SEQUENCES, "-a", ATTRIBUTIONS, "-n", 60,
		"-w", 300])

	assert result.exit_code == 0, result.output
	assert (tmp_path / "modisco_results.h5").exists()


@pytest.mark.cmd
def test_motifs_missing_file(tmp_path):
	result = _invoke(["motifs", "-s", tmp_path / "missing.npz", "-a",
		ATTRIBUTIONS, "-n", 60])
	assert result.exit_code == 2
	assert "does not exist" in result.output


##


@pytest.mark.cmd
@pytest.mark.parametrize("datatype", [str(d) for d in MemeDataType])
def test_meme(modisco_h5, tmp_path, datatype):
	output = tmp_path / "motifs.meme"
	result = _invoke(["meme", "-i", modisco_h5, "-t", datatype, "-o", output,
		"-q"])

	assert result.exit_code == 0, result.output
	assert result.output == ""
	assert list(read_meme(str(output)).keys()) == ["pos_patterns.pattern_0",
		"pos_patterns.pattern_1", "neg_patterns.pattern_0",
		"neg_patterns.pattern_1"]


@pytest.mark.cmd
def test_meme_stdout(modisco_h5):
	result = _invoke(["meme", "-i", modisco_h5, "-t", "PFM"])

	assert result.exit_code == 0, result.output
	assert result.output.startswith("MEME version 5\n")
	assert result.output.count("MOTIF ") == 4


@pytest.mark.cmd
def test_meme_repeated(modisco_h5, meme_writer_default):
	result0 = _invoke(["meme", "-i", modisco_h5, "-t", "CWM"])
	meme_writer_default.clear()
	result1 = _invoke(["meme", "-i", modisco_h5, "-t", "CWM"])
	assert result0.output == result1.output


@pytest.mark.cmd
@pytest.mark.parametrize("datatype", ["pfm", "PWM", "CWM_PFM", ""])
def test_meme_bad_datatype(modisco_h5, datatype):
	result = _invoke(["meme", "-i", modisco_h5, "-t", datatype])
	assert result.exit_code == 2


@pytest.mark.cmd
def test_meme_datatype_required(modisco_h5):
	result = _invoke(["meme", "-i", modisco_h5])
	assert result.exit_code == 2
	assert "Missing option '-t'" in result.output


##


@pytest.mark.cmd
def test_seqlet_bed(small_h5, tmp_path):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	output = tmp_path / "seqlets.bed"

	result = _invoke(["seqlet-bed", "-i", small_h5, "-p", peaks, "-c", "*",
		"-o", output, "-q"])

	assert result.exit_code == 0, result.output
	assert result.output == ""
	lines = [l for l in open(output).read().split("\n")
		if l and not l.startswith("track")]
	assert lines == ["chr1\t250\t255\tpattern_0.0\t100\t+",
		"chr1\t3268\t3273\tpattern_0.1\t103\t-",
		"chrX\t5278\t5283\tpattern_0.0\t105\t+"]


@pytest.mark.cmd
def test_seqlet_bed_stdout(small_h5, tmp_path):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)

	result = _invoke(["seqlet-bed", "-i", small_h5, "-p", peaks, "-c", "*"])
	assert result.exit_code == 0, result.output
	assert result.output.count("pattern_0.") == 3


@pytest.mark.cmd
def test_seqlet_bed_chroms(small_h5, tmp_path):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 12, chroms=('chr1', 'chr2'))
	output = tmp_path / "seqlets.bed"

	result = _invoke(["seqlet-bed", "-i", small_h5, "-p", peaks, "-c", "chr2",
		"-o", output, "-q"])
	assert result.exit_code == 0, result.output

	chroms = [l.split("\t")[0] for l in open(output).read().split("\n")
		if l and not l.startswith("track")]
	assert chroms == ["chr2", "chr2", "chr2"]


@pytest.mark.cmd
def test_seqlet_bed_windowsize(small_h5, tmp_path):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	output = tmp_path / "seqlets.bed"

	result = _invoke(["seqlet-bed", "-i", small_h5, "-p", peaks, "-c", "*",
		"-o", output, "-q", "-w", 100])
	assert result.exit_code == 0, result.output
	assert "chr1\t220\t225\tpattern_0.0" in open(output).read()


@pytest.mark.cmd
@pytest.mark.parametrize("missing", ["-i", "-p", "-c"])
def test_seqlet_bed_required(small_h5, tmp_path, missing):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)

	args = {"-i": small_h5, "-p": peaks, "-c": "*"}
	del args[missing]
	result = _invoke(["seqlet-bed"] + [x for kv in args.items() for x in kv])
	assert result.exit_code == 2


##


@pytest.mark.cmd
def test_seqlet_fasta(small_h5, tmp_path):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)
	sequences = tmp_path / "sequences.npz"
	numpy.savez(sequences, random_track_set(n=6, length=40).one_hot.transpose(
		0, 2, 1))
	output = tmp_path / "seqlets.fa"

	result = _invoke(["seqlet-fasta", "-i", small_h5, "-p", peaks, "-s",
		sequences, "-c", "*", "-o", output, "-q"])

	assert result.exit_code == 0, result.output
	headers = open(output).read().split("\n")[::2]
	assert headers == [">chr1:250-255 dir=+ pattern_0.0",
		">chr1:3268-3273 dir=- pattern_0.1", ">chrX:5278-5283 dir=+ pattern_0.0"]


@pytest.mark.cmd
def test_seqlet_fasta_real(modisco_h5, tmp_path):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 300)
	output = tmp_path / "seqlets.fa"

	result = _invoke(["seqlet-fasta", "-i", modisco_h5, "-p", peaks, "-s",
		SEQUENCES, "-c", "*", "-o", output, "-q"])

	assert result.exit_code == 0, result.output
	lines = open(output).read().split("\n")
	assert len(lines) == 2 * (74 + 26 + 77 + 27)


@pytest.mark.cmd
def test_seqlet_fasta_mismatch(small_h5, tmp_path):
	peaks = tmp_path / "peaks.bed"
	write_peaks(peaks, 6)

	result = _invoke(["seqlet-fasta", "-i", small_h5, "-p", peaks, "-s",
		SEQUENCES, "-c", "*", "-q"])
	assert result.exit_code == 1
	assert "does not match the number of peaks" in result.output


##


@pytest.mark.cmd
def test_convert_backward(modisco_h5, tmp_path):
	old = tmp_path / "old.h5"
	new = tmp_path / "new.h5"

	result0 = _invoke(["convert-backward", "-i", modisco_h5, "-o", old])
	result1 = _invoke(["convert", "-i", old, "-o", new])

	assert result0.exit_code == 0, result0.output
	assert result1.exit_code == 0, result1.output

	patterns0, patterns1 = _patterns(modisco_h5), _patterns(new)
	assert patterns0.keys() == patterns1.keys()
	for key in patterns0:
		assert_array_equal(patterns0[key], patterns1[key])


@pytest.mark.cmd
@pytest.mark.parametrize("command", ["convert", "convert-backward"])
def test_convert_required(command, tmp_path):
	result = _invoke([command, "-o", tmp_path / "out.h5"])
	assert result.exit_code == 2
	assert "Missing option '-i'" in result.output


##


@pytest.mark.cmd
def test_report(report_h5, tmp_path):
	output = tmp_path / "report"
	result = _invoke(["report", "-i", report_h5, "-o", output, "--n_examples",
		1])

	assert result.exit_code == 0, result.output
	assert result.output == "Report generated: {}\n".format(os.path.join(
		str(output), "report.html"))
	assert (output / "report.html").exists()


@pytest.mark.cmd
def test_report_lite(report_h5, tmp_path, meme_db):
	output = tmp_path / "report"
	result = _invoke(["report", "-i", report_h5, "-o", output, "-m", meme_db,
		"-l", "-n", 1, "--n_examples", 1, "--trim_threshold", 0.2])

	assert result.exit_code == 0, result.output
	assert "P-value" in open(output / "report.html").read()


@pytest.mark.cmd
def test_report_simple(report_h5, tmp_path):
	output = tmp_path / "report"
	result = _invoke(["report-simple", "-i", report_h5, "-o", output, "-s",
		"report/"])

	assert result.exit_code == 0, result.output
	html = open(output / "motifs.html").read()
	assert "report/trimmed_logos/pos_patterns.pattern_0.cwm.fwd.png" in html
	assert len(os.listdir(output / "trimmed_logos")) == 2


@pytest.mark.cmd
def test_report_simple_lite(report_h5, tmp_path, meme_db):
	output = tmp_path / "report"
	result = _invoke(["report-simple", "-i", report_h5, "-o", output, "-m",
		meme_db, "-l", "-n", 1])

	assert result.exit_code == 0, result.output
	assert "match0" in open(output / "motifs.html").read()


@pytest.mark.cmd
@pytest.mark.parametrize("command", ["report", "report-simple"])
def test_report_required(command, tmp_path):
	result = _invoke([command, "-o", tmp_path / "report"])
	assert result.exit_code == 2
