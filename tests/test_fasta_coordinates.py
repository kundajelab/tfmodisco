"""Independent, literal sequence oracles for the public FASTA export path."""
import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from modiscolite.cli import cli
from modiscolite.io import write_fasta_from_h5


def fasta_inputs(tmp_path, sequence, window, start, end, revcomp=False,
                 metadata=True, category="pos_patterns"):
    h5 = tmp_path / "results.h5"
    with h5py.File(h5, "w") as f:
        if metadata:
            f.attrs["window_size"] = window
        seqlets = f.create_group(f"{category}/pattern_0/seqlets")
        for key, value in dict(start=start, end=end, example_idx=0,
                               is_revcomp=revcomp).items():
            seqlets.create_dataset(key, data=[value])
    peaks = tmp_path / "peaks.bed"
    # Center at genomic coordinate 1000; independent of NPZ length.
    peaks.write_text("chr1\t500\t1500\tpeak\t1\t.\n")
    npz = tmp_path / "sequences.npz"
    bases = "ACGT"
    one_hot = np.eye(4, dtype=np.int8)[[bases.index(b) for b in sequence]]
    np.savez(npz, one_hot.T[None])
    return h5, peaks, npz, tmp_path / "seqlets.fa"


def export(inputs, window=None, chroms="*"):
    write_fasta_from_h5(*inputs, chroms, window, True)
    return inputs[3].read_text().splitlines()


@pytest.mark.parametrize("revcomp", [False, True])
def test_fasta_cropped_sequence(tmp_path, revcomp):
    inputs = fasta_inputs(tmp_path, "A" * 400 + "C" * 200 + "G" * 400,
                          200, 10, 30, revcomp)
    header, sequence = export(inputs)
    print(f"FASTA_SEQUENCE={sequence}; LENGTH={len(sequence)}; HEADER={header}")
    assert sequence == "C" * 20
    assert header == f">chr1:911-930 dir={'-' if revcomp else '+'} pattern_0.0"


@pytest.mark.parametrize("revcomp", [False, True])
@pytest.mark.parametrize("start,end,expected,span", [
    (0, 4, "ACGA", "999-1002"),
    (0, 1, "A", "999-999"),
    (3, 4, "A", "1002-1002"),
])
def test_fasta_full_window_and_boundaries(tmp_path, revcomp, start, end,
                                          expected, span):
    inputs = fasta_inputs(tmp_path, "ACGA", 4, start, end, revcomp)
    header, sequence = export(inputs)
    assert sequence == expected  # Forward genomic orientation on both strands.
    assert header == f">chr1:{span} dir={'-' if revcomp else '+'} pattern_0.0"
    assert len(sequence) == end - start


@pytest.mark.parametrize("sequence,window,expected", [
    ("AAACGTACCCC", 4, "CGTA"),  # Original length 11, crop starts at index 3.
    ("AAACGTACCC", 5, "CGTA"),   # CLI's existing odd window extracts four bases.
    ("AAACGTACCCC", 5, "CGTA"),
    ("AAACGTACCC", 4, "CGTA"),
])
def test_fasta_crop_origin_parity(tmp_path, sequence, window, expected):
    inputs = fasta_inputs(tmp_path, sequence, window, 0, 4)
    assert export(inputs) == [">chr1:999-1002 dir=+ pattern_0.0", expected]


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("window", [np.uint8(200), np.int8(100),
                                     np.uint64(200), np.int64(100)])
def test_fasta_numpy_window_types(tmp_path, metadata, window):
    inputs = fasta_inputs(tmp_path, "A" * 400 + "C" * 200 + "G" * 400,
                          window, 10, 30, metadata=metadata)
    assert export(inputs, None if metadata else window)[1] == "C" * 20


def test_fasta_explicit_window_overrides_metadata(tmp_path):
    inputs = fasta_inputs(tmp_path, "AACGTTAA", 8, 0, 4)
    assert export(inputs, 4) == [">chr1:999-1002 dir=+ pattern_0.0", "CGTT"]


def test_fasta_filtered_example_index_and_negative_pattern(tmp_path):
    inputs = fasta_inputs(tmp_path, "AACGTTAA", 4, 0, 4, True,
                          category="neg_patterns")
    inputs[1].write_text("chr2\t0\t100\tother\t1\t.\n"
                         "chr1\t500\t1500\tpeak\t1\t.\n")
    # NPZ rows already correspond to selected chromosomes, as the API requires.
    assert export(inputs, chroms=["chr1"]) == [
        ">chr1:999-1002 dir=- pattern_0.0", "CGTT"]


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("window", [0, -1, 9, True, 4.5, "4"])
def test_fasta_rejects_invalid_window(tmp_path, metadata, window):
    inputs = fasta_inputs(tmp_path, "AACGTTAA", window, 0, 4,
                          metadata=metadata)
    with pytest.raises(ValueError, match="window_size"):
        export(inputs, None if metadata else window)
    assert not inputs[3].exists()


@pytest.mark.parametrize("start,end", [(-1, 2), (0, 5), (3, 2), (2, 2)])
def test_fasta_rejects_invalid_seqlet_bounds(tmp_path, start, end):
    inputs = fasta_inputs(tmp_path, "AACGTTAA", 4, start, end)
    with pytest.raises(ValueError, match="seqlet.*bounds"):
        export(inputs)
    assert not inputs[3].exists()


@pytest.mark.parametrize("shape", [(4, 8), (1, 8, 4), (1, 4, 8, 1)])
def test_fasta_rejects_invalid_input_shape(tmp_path, shape):
    inputs = fasta_inputs(tmp_path, "AACGTTAA", 4, 0, 4)
    np.savez(inputs[2], np.zeros(shape))
    with pytest.raises(ValueError, match="shape"):
        export(inputs)


@pytest.mark.cmd
def test_fasta_cli_cropped_sequence(tmp_path):
    h5, peaks, npz, out = fasta_inputs(tmp_path,
        "A" * 400 + "C" * 200 + "G" * 400, 200, 10, 30, True)
    result = CliRunner().invoke(cli, ["seqlet-fasta", "-i", str(h5),
        "-p", str(peaks), "-s", str(npz), "-o", str(out), "-c", "*", "-q"])
    assert result.exit_code == 0, result.output
    assert out.read_text().splitlines() == [
        ">chr1:911-930 dir=- pattern_0.0", "C" * 20]
