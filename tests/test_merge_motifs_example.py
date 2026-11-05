"""Exercise the example's reconstruction, without its downloads or motif merging."""
import ast
import json
from pathlib import Path

import h5py as h5
import numpy as np
import pytest

from modiscolite.core import Seqlet, TrackSet
from modiscolite.util import calculate_window_offsets


NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "MergeMotifsAcrossRuns.ipynb"


def _reconstruct(merge_list, input_window_sizes=None, pattern_group="pos_patterns"):
    notebook = json.loads(NOTEBOOK.read_text())
    cells = ["".join(cell["source"]) for cell in notebook["cells"]
             if cell["cell_type"] == "code"
             and "union_patterncoords = []" in "".join(cell["source"])]
    assert len(cells) == 1, "Expected one notebook reconstruction block"
    source = "\n".join(line for line in cells[0].splitlines()
                       if not line.startswith(("%", "!")))
    body = ast.parse(source).body

    def assigns(node, name):
        return (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))

    start = next(i for i, node in enumerate(body) if assigns(node, "union_onehot"))
    stop = next(i for i, node in enumerate(body) if assigns(node, "track_set"))
    block = ast.Module(body=body[start:stop + 1], type_ignores=[])
    namespace = dict(np=np, h5=h5, TrackSet=TrackSet, merge_list=merge_list,
                     pattern_group=pattern_group, input_window_sizes=input_window_sizes or {})
    exec(compile(block, str(NOTEBOOK), "exec"), namespace)
    seqlets = [Seqlet(*coord) for pattern in namespace["union_patterncoords"]
               for coord in zip(*pattern)]
    return namespace, namespace["track_set"].create_seqlets(seqlets)


def _write_run(tmp_path, name="run", length=1000, window=200, n_examples=1,
               patterns=None, pattern_group="pos_patterns"):
    # The first example is A*400 + C*200 + G*400 at the default length.
    bases = np.zeros((n_examples, length), dtype=int)
    bases[:, 400:600] = 1
    bases[:, 600:] = 2
    onehot = np.eye(4, dtype=np.float32)[bases]
    scores = np.arange(n_examples * length * 4, dtype=np.float32).reshape(onehot.shape) + 0.5
    results = tmp_path / f"{name}.h5"
    ohe = tmp_path / f"{name}_ohe.npz"
    attrs = tmp_path / f"{name}_scores.npz"
    np.savez(ohe, onehot.transpose(0, 2, 1))
    np.savez(attrs, scores.transpose(0, 2, 1))
    if patterns is None:
        patterns = [[(0, 10, 30, False)]]
    with h5.File(results, "w") as handle:
        if window is not None:
            handle.attrs["window_size"] = window
        for i, coords in enumerate(patterns):
            group = handle.create_group(f"{pattern_group}/pattern_{i}/seqlets")
            for key, values in zip(("example_idx", "start", "end", "is_revcomp"), zip(*coords)):
                group.create_dataset(key, data=values)
    return (str(results), str(ohe), str(attrs)), onehot, scores


def test_cropped_window_reconstructs_sequence_and_scores(tmp_path):
    entry, onehot, scores = _write_run(tmp_path)
    state, (seqlet,) = _reconstruct([entry])
    observed = "".join("ACGT"[i] for i in seqlet.sequence.argmax(axis=1))
    print(f"RECONSTRUCTED_SEQUENCE: {observed}")
    print(f"EXPECTED_SEQUENCE: {'C' * 20}")
    assert observed == "C" * 20
    assert (seqlet.start, seqlet.end) == (410, 430)
    np.testing.assert_array_equal(seqlet.hypothetical_contribs, scores[0, 410:430])
    np.testing.assert_array_equal(seqlet.contrib_scores, (onehot * scores)[0, 410:430])
    np.testing.assert_array_equal(state["union_onehot"], onehot)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("window, start, end", [(1000, 10, 30), (200, 410, 430)])
def test_full_and_cropped_windows_preserve_strand_tracks(tmp_path, reverse, window, start, end):
    entry, onehot, scores = _write_run(tmp_path, window=window,
                                     patterns=[[(0, 10, 30, reverse)]])
    _, (seqlet,) = _reconstruct([entry])
    expected_sequence = onehot[0, start:end]
    expected_scores = scores[0, start:end]
    if reverse:
        expected_sequence = expected_sequence[::-1, ::-1]
        expected_scores = expected_scores[::-1, ::-1]
    assert (seqlet.start, seqlet.end, seqlet.is_revcomp) == (start, end, reverse)
    np.testing.assert_array_equal(seqlet.sequence, expected_sequence)
    np.testing.assert_array_equal(seqlet.hypothetical_contribs, expected_scores)
    np.testing.assert_array_equal(seqlet.contrib_scores, expected_sequence * expected_scores)


def test_runs_with_different_windows_preserve_example_remapping(tmp_path):
    a, a_onehot, a_scores = _write_run(
        tmp_path, "a", n_examples=3,
        patterns=[[(2, 10, 30, False), (0, 20, 40, True)], [(2, 30, 50, False)]])
    b, b_onehot, b_scores = _write_run(
        tmp_path, "b", window=400, n_examples=4, patterns=[[(3, 10, 30, False)]])
    state, seqlets = _reconstruct([a, b])
    assert [(s.example_idx, s.start, s.end, bool(s.is_revcomp)) for s in seqlets] == [
        (1, 410, 430, False), (0, 420, 440, True),
        (1, 430, 450, False), (2, 310, 330, False)]
    np.testing.assert_array_equal(state["union_onehot"],
                                  np.array([a_onehot[0], a_onehot[2], b_onehot[3]]))
    np.testing.assert_array_equal(state["union_hypscores"],
                                  np.array([a_scores[0], a_scores[2], b_scores[3]]))


@pytest.mark.parametrize("length, window, origin", [(10, 4, 3), (11, 4, 3),
                                                     (10, 5, 3), (11, 5, 3), (12, 5, 4)])
def test_crop_origin_matches_cli_for_even_and_odd_windows(tmp_path, length, window, origin):
    entry, _, scores = _write_run(tmp_path, length=length, window=window,
                                 patterns=[[(0, 1, 3, False)]])
    _, (seqlet,) = _reconstruct([entry])
    assert calculate_window_offsets(length // 2, window)[0] == origin
    assert (seqlet.start, seqlet.end) == (origin + 1, origin + 3)
    np.testing.assert_array_equal(seqlet.hypothetical_contribs, scores[0, origin + 1:origin + 3])


def test_hdf5_window_metadata_takes_precedence_over_legacy_mapping(tmp_path):
    entry, _, _ = _write_run(tmp_path, window=400)
    _, (seqlet,) = _reconstruct([entry], {entry[0]: 200})
    assert (seqlet.start, seqlet.end) == (310, 330)


def test_legacy_window_is_explicit_per_run(tmp_path):
    a, _, _ = _write_run(tmp_path, "a", window=None)
    b, _, _ = _write_run(tmp_path, "b", window=None)
    _, seqlets = _reconstruct([a, b], {a[0]: 200, b[0]: 400})
    assert [(s.start, s.end) for s in seqlets] == [(410, 430), (310, 330)]


def test_missing_legacy_window_explains_how_to_reconstruct(tmp_path):
    entry, _, _ = _write_run(tmp_path, window=None)
    with pytest.raises(ValueError, match="input_window_sizes") as error:
        _reconstruct([entry])
    assert entry[0] in str(error.value)


@pytest.mark.parametrize("window", [0, -1, 1001, 2.5, "200", True])
def test_invalid_input_windows_are_rejected(tmp_path, window):
    entry, _, _ = _write_run(tmp_path, window=window)
    with pytest.raises(ValueError, match="window"):
        _reconstruct([entry])


@pytest.mark.parametrize("shape", [(1, 4, 999), (2, 4, 1000), (1, 3, 1000)])
def test_mismatched_input_shapes_are_rejected(tmp_path, shape):
    entry, _, _ = _write_run(tmp_path)
    np.savez(entry[2], np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        _reconstruct([entry])


def test_negative_pattern_group_uses_the_same_coordinate_translation(tmp_path):
    entry, _, _ = _write_run(tmp_path, pattern_group="neg_patterns")
    _, (seqlet,) = _reconstruct([entry], pattern_group="neg_patterns")
    assert (seqlet.start, seqlet.end) == (410, 430)


@pytest.mark.parametrize("window, origin", [
    pytest.param(np.int64(200), 400, id="int64-cropped"),
    pytest.param(np.uint64(200), 400, id="uint64-cropped"),
    pytest.param(np.int8(100), 450, id="int8-cropped"),
    pytest.param(np.uint64(1000), 0, id="uint64-full-window"),
])
@pytest.mark.parametrize("legacy", [False, True], ids=["metadata", "legacy"])
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_numpy_integer_windows_preserve_sequence_and_scores(
        tmp_path, window, origin, legacy, reverse):
    entry, onehot, scores = _write_run(
        tmp_path, window=None if legacy else window,
        patterns=[[(0, 10, 30, reverse)]])
    mapping = {entry[0]: window} if legacy else {}
    state, (seqlet,) = _reconstruct([entry], mapping)
    start, end = origin + 10, origin + 30
    assert (seqlet.start, seqlet.end) == (start, end)
    assert isinstance(seqlet.start, (int, np.integer))
    assert isinstance(seqlet.end, (int, np.integer))
    expected_sequence = onehot[0, start:end]
    expected_scores = scores[0, start:end]
    if reverse:
        expected_sequence = expected_sequence[::-1, ::-1]
        expected_scores = expected_scores[::-1, ::-1]
    np.testing.assert_array_equal(seqlet.sequence, expected_sequence)
    np.testing.assert_array_equal(seqlet.hypothetical_contribs, expected_scores)
    np.testing.assert_array_equal(seqlet.contrib_scores, expected_sequence * expected_scores)
    np.testing.assert_array_equal(state["union_onehot"], onehot)


@pytest.mark.parametrize("dtype,length,window,start,end,expected_start,expected_end", [
    pytest.param(np.int8, 1000, 200, 10, 30, 410, 430,
                 id="int8-offset-outside-dtype"),
    pytest.param(np.uint8, 1000, 200, 10, 30, 410, 430,
                 id="uint8-offset-outside-dtype"),
    pytest.param(np.int8, 400, 200, 100, 120, 200, 220,
                 id="int8-addition-wraps"),
    pytest.param(np.uint8, 600, 400, 200, 230, 300, 330,
                 id="uint8-addition-wraps"),
])
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_compact_seqlet_coordinates_do_not_overflow(
        tmp_path, dtype, length, window, start, end,
        expected_start, expected_end, reverse):
    entry, onehot, scores = _write_run(
        tmp_path, length=length, window=window,
        patterns=[[(0, dtype(start), dtype(end), reverse)]])
    with h5.File(entry[0]) as handle:
        assert handle["pos_patterns/pattern_0/seqlets/start"].dtype == np.dtype(dtype)
    _, (seqlet,) = _reconstruct([entry])
    assert (seqlet.start, seqlet.end) == (expected_start, expected_end)
    expected_sequence = onehot[0, expected_start:expected_end]
    expected_scores = scores[0, expected_start:expected_end]
    if reverse:
        expected_sequence = expected_sequence[::-1, ::-1]
        expected_scores = expected_scores[::-1, ::-1]
    np.testing.assert_array_equal(seqlet.sequence, expected_sequence)
    np.testing.assert_array_equal(seqlet.hypothetical_contribs, expected_scores)
    np.testing.assert_array_equal(seqlet.contrib_scores, expected_sequence * expected_scores)
