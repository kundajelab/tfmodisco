# conftest.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import os

# Both must be set before numba and matplotlib are first imported. numba
# otherwise starts one thread per core, which oversubscribes shared machines
# and gains nothing on inputs this small.
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("MPLBACKEND", "Agg")

import copy
import pathlib

import numpy
import pytest

from modiscolite.core import Seqlet
from modiscolite.core import TrackSet
from modiscolite.extract_seqlets import extract_seqlets
from modiscolite.io import save_hdf5
from modiscolite.meme_writer import MEMEWriter
from modiscolite.meme_writer import MEMEWriterMotif
from modiscolite.tfmodisco import TFMoDISco


DATA_DIR = pathlib.Path(__file__).parent / "data"


def pytest_sessionstart(session):
	"""Compile the lazily compiled numba kernels before any test is timed.

	The first call to each kernel compiles for 1-3 s, and every new dtype or
	memory layout compiles again, so these calls cover each signature that the
	library and the tests pass in. memelite caches its kernels on disk, so its
	call is only slow on the first run in a fresh environment.
	"""

	from memelite import tomtom
	from modiscolite import affinitymat
	from modiscolite import gapped_kmer

	for dtype in ('float32', 'float64'):
		affinitymat.pairwise_jaccard(numpy.ones((2, 3), dtype=dtype), 1)

	X = numpy.array([[[0, 0, 1.0], [1, 1, 1.0]]])
	gapped_kmer._extract_gkmers(X, 2, 2, 1, 2, 1)

	data = numpy.ones(1)
	indices = numpy.zeros(1, dtype='int64')
	indptr = numpy.array([0, 1], dtype='int64')
	affinitymat._sparse_mm_dot(data, indices, indptr, data, indices, indptr, 1)

	# tomtom compiles separately for C- and F-ordered queries; transposed
	# PPMs are F-ordered unless they are a single position long.
	pwm = numpy.random.RandomState(0).dirichlet(numpy.ones(4), size=6).T
	for Q in (numpy.ascontiguousarray(pwm), numpy.asfortranarray(pwm)):
		tomtom([Q], [pwm, pwm], n_nearest=1)


##


@pytest.fixture
def meme_writer_default():
	"""Clear the motif list that every default MEMEWriter shares.

	MEMEWriter's default `motifs=[]` is a single list shared by all instances,
	and write_meme_from_h5 relies on that default, so without this each call
	would also write the motifs from every earlier call in the session.
	"""

	default = MEMEWriter.__init__.__defaults__[0]
	default.clear()
	yield default
	default.clear()


##


@pytest.fixture(scope="session")
def _arrays():
	sequences = numpy.load(DATA_DIR / "sequences.npz")["arr_0"]
	attributions = numpy.load(DATA_DIR / "attributions.npz")["arr_0"]

	one_hot = numpy.ascontiguousarray(sequences.transpose(0, 2, 1),
		dtype='float32')
	hypothetical_contribs = numpy.ascontiguousarray(
		attributions.transpose(0, 2, 1), dtype='float32')

	one_hot.flags.writeable = False
	hypothetical_contribs.flags.writeable = False
	return one_hot, hypothetical_contribs


@pytest.fixture
def one_hot(_arrays):
	return _arrays[0].copy()


@pytest.fixture
def hypothetical_contribs(_arrays):
	return _arrays[1].copy()


@pytest.fixture
def contrib_scores(one_hot, hypothetical_contribs):
	return one_hot * hypothetical_contribs


@pytest.fixture
def track_set(one_hot, hypothetical_contribs, contrib_scores):
	return TrackSet(one_hot=one_hot, contrib_scores=contrib_scores,
		hypothetical_contribs=hypothetical_contribs)


##


@pytest.fixture(scope="session")
def _seqlet_coords(_arrays):
	one_hot, hypothetical_contribs = _arrays
	attributions = (one_hot * hypothetical_contribs).sum(axis=2)

	seqlets, threshold = extract_seqlets(attributions, window_size=20,
		flank=5, suppress=15, target_fdr=0.05, min_passing_windows_frac=0.03,
		max_passing_windows_frac=0.2, weak_threshold_for_counting_sign=0.8)

	coords = [(s.example_idx, s.start, s.end, s.is_revcomp) for s in seqlets]
	return coords, threshold


@pytest.fixture
def seqlets(track_set, _seqlet_coords):
	"""All seqlets called on the fixture with a 20 bp window and 5 bp flanks."""

	return track_set.create_seqlets([Seqlet(*c) for c in _seqlet_coords[0]])


@pytest.fixture
def pos_seqlets(seqlets, _seqlet_coords):
	"""The first 150 seqlets whose core passes the threshold, as TFMoDISco."""

	threshold = _seqlet_coords[1]
	return [s for s in seqlets if s.contrib_scores[5:-5].sum() > threshold][:150]


##


@pytest.fixture(scope="session")
def _signed_arrays(_arrays):
	"""The fixture with the attributions of the last 150 examples negated."""

	one_hot, hypothetical_contribs = _arrays
	hypothetical_contribs = hypothetical_contribs.copy()
	hypothetical_contribs[150:] *= -1
	hypothetical_contribs.flags.writeable = False
	return one_hot, hypothetical_contribs


@pytest.fixture
def signed_track_set(_signed_arrays):
	one_hot, hypothetical_contribs = _signed_arrays
	return TrackSet(one_hot=one_hot.copy(),
		contrib_scores=one_hot*hypothetical_contribs,
		hypothetical_contribs=hypothetical_contribs.copy())


@pytest.fixture(scope="session")
def _patterns(_signed_arrays):
	one_hot, hypothetical_contribs = _signed_arrays

	return TFMoDISco(one_hot=one_hot.copy(),
		hypothetical_contribs=hypothetical_contribs.copy(),
		max_seqlets_per_metacluster=150, sliding_window_size=20,
		flank_size=5, trim_to_window_size=30, initial_flank_to_add=10,
		target_seqlet_fdr=0.05, n_leiden_runs=2)


@pytest.fixture
def pos_patterns(_patterns):
	"""Positive patterns found on `signed_track_set`, copied per test."""

	return copy.deepcopy(_patterns[0])


@pytest.fixture
def neg_patterns(_patterns):
	"""Negative patterns found on `signed_track_set`, copied per test."""

	return copy.deepcopy(_patterns[1])


@pytest.fixture(scope="session")
def modisco_h5(_patterns, tmp_path_factory):
	"""A read-only results file holding both the positive and negative patterns."""

	filename = tmp_path_factory.mktemp("results") / "modisco_results.h5"
	save_hdf5(filename, _patterns[0], _patterns[1], window_size=300)
	return filename


@pytest.fixture(scope="session")
def meme_db(_patterns, tmp_path_factory):
	"""A MEME database holding the cores of the two positive patterns followed
	by three random motifs, named `db_motif_<i> NAME<i>`. The negative
	patterns carry the same two motifs as the positive ones."""

	rng = numpy.random.RandomState(0)
	ppms = [pattern.sequence[10:40] for pattern in _patterns[0]]
	ppms += [rng.dirichlet(numpy.ones(4), size=12) for _ in range(3)]

	writer = MEMEWriter(memesuite_version='5', motifs=[], alphabet='ACGT',
		background_frequencies='A 0.25 C 0.25 G 0.25 T 0.25')

	for i, ppm in enumerate(ppms):
		writer.add_motif(MEMEWriterMotif(name="db_motif_{} NAME{}".format(i, i),
			probability_matrix=ppm, source_sites=1, alphabet='ACGT'))

	filename = tmp_path_factory.mktemp("meme") / "motifs.meme"
	writer.write(filename)
	return filename
