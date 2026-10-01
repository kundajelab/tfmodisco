"""Independent correctness probes added by the PR #136 review.

Production code is deliberately unchanged. These expected failures document
contracts, rather than requiring the current incorrect output.
"""
import numpy as np
import pytest

from modiscolite.io import save_hdf5, write_bed_from_h5, write_fasta_from_h5
from modiscolite.tfmodisco import _density_adaptation
from .synthetic import random_track_set, make_pattern, write_peaks


COORDS = [(0, 1, False), (0, 1, True), (39, 40, False),
          (39, 40, True), (2, 12, False), (2, 12, True)]
IDS = ['first-forward', 'first-reverse', 'last-forward', 'last-reverse',
       'ten-forward', 'ten-reverse']


@pytest.mark.parametrize('start,end,reverse', COORDS, ids=IDS)
@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S01: BED start must be zero-based')
def test_bed_interval_width(tmp_path, start, end, reverse):
    tracks = random_track_set(n=1, length=40)
    pattern = make_pattern(tracks, [(0, start, end, reverse)])
    save_hdf5(tmp_path / 'input.h5', [pattern], None, window_size=40)
    write_peaks(tmp_path / 'peaks.bed', 1)
    write_bed_from_h5(tmp_path / 'input.h5', tmp_path / 'peaks.bed',
                      tmp_path / 'out.bed', '*', None, True)
    row = (tmp_path / 'out.bed').read_text().splitlines()[1].split('\t')
    observed = (int(row[1]), int(row[2]))
    expected = (247 + start, 247 + end)
    assert observed == expected, f'BED coordinates {observed}; expected {expected} (width {end-start})'
    assert int(row[2]) - int(row[1]) == end - start
    assert row[5] == ('-' if reverse else '+')


@pytest.mark.parametrize('start,end,reverse', COORDS, ids=IDS)
@pytest.mark.xfail(strict=True, raises=AssertionError, reason='B11: FASTA must include every source base')
def test_fasta_complete_sequence(tmp_path, start, end, reverse):
    tracks = random_track_set(n=1, length=40)
    pattern = make_pattern(tracks, [(0, start, end, reverse)])
    save_hdf5(tmp_path / 'input.h5', [pattern], None, window_size=40)
    write_peaks(tmp_path / 'peaks.bed', 1)
    np.savez(tmp_path / 'sequences.npz', tracks.one_hot.transpose(0, 2, 1))
    write_fasta_from_h5(tmp_path / 'input.h5', tmp_path / 'peaks.bed',
                        tmp_path / 'sequences.npz', tmp_path / 'out.fa', '*', None, True)
    header, observed = (tmp_path / 'out.fa').read_text().split('\n')[:2]
    strand = '-' if reverse else '+'
    assert header == f'>chr1:{248+start}-{247+end} dir={strand} pattern_0.0'
    # This API exports forward-genomic sequence even when the motif is reverse-oriented.
    expected = ''.join(np.array(list('ACGT'))[tracks.one_hot[0, start:end].argmax(axis=1)])
    assert observed == expected, f'FASTA sequence {observed!r}; expected {expected!r}'
    assert len(observed) == end - start
    first, last = map(int, header.split(':')[1].split()[0].split('-'))
    assert len(observed) == last - first + 1


@pytest.mark.xfail(strict=True, raises=AssertionError, reason='S02: beta must be used consistently')
def test_density_requested_perplexity():
    affinities = np.full((4, 4), 0.5)
    np.fill_diagonal(affinities, 1.0)
    neighbors = np.tile(np.arange(4), (4, 1))
    result = _density_adaptation(affinities, neighbors, 2.0).toarray()
    # Symmetry preserves conditional row probabilities through symmetrization.
    probabilities = result[0] / result[0].sum()
    observed = np.exp(-np.sum(probabilities * np.log(probabilities)))
    assert abs(observed - 2.0) < 1e-4, f'Requested perplexity 2; observed {observed}; row {probabilities}'
