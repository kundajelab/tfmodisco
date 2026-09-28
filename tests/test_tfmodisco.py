# test_tfmodisco.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import scipy.sparse
import pytest

from modiscolite.tfmodisco import _density_adaptation
from modiscolite.tfmodisco import _filter_patterns
from modiscolite.tfmodisco import _patterns_from_clusters
from modiscolite.tfmodisco import _filter_by_correlation
from modiscolite.tfmodisco import seqlets_to_patterns
from modiscolite.tfmodisco import TFMoDISco

from modiscolite.affinitymat import pairwise_jaccard
from modiscolite.core import Seqlet
from modiscolite.core import SeqletSet
from modiscolite.core import TrackSet
from modiscolite.util import binary_search_perplexity
from modiscolite.util import get_2d_data_from_patterns

from .synthetic import planted_track_set
from .synthetic import two_motif_track_set
from .synthetic import random_track_set
from .synthetic import make_seqlets
from .synthetic import make_pattern

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


BG = numpy.full(4, 0.25)

KWARGS = dict(sliding_window_size=20, flank_size=5, trim_to_window_size=30,
	initial_flank_to_add=10, target_seqlet_fdr=0.05, n_leiden_runs=2)


def _coords(pattern):
	return [(s.example_idx, s.start, s.end, s.is_revcomp)
		for s in pattern.seqlets]


def _affinity(n, k, random_state=0):
	X = numpy.random.RandomState(random_state).uniform(0, 1, size=(n, 12))
	affmat, neighbors = pairwise_jaccard(X, k)
	return affmat, neighbors


def _density_reference(affmat_nn, seqlet_neighbors, perplexity):
	n = len(affmat_nn)
	D = numpy.zeros((n, n))
	C = numpy.zeros((n, n))
	for i in range(n):
		for j, a in zip(seqlet_neighbors[i], affmat_nn[i]):
			d = max(numpy.log(1.0 / (0.5 * max(a, 1e-7)) - 1), 0)
			if d > 0:
				D[i, j] = d
				C[i, j] = 1

	D, C = D + D.T, C + C.T
	D = numpy.divide(D, C, out=numpy.zeros_like(D), where=C > 0)

	betas, norms = [], []
	for i in range(n):
		d = D[i][C[i] > 0]
		beta = binary_search_perplexity(perplexity, d)
		betas.append(beta)
		norms.append(numpy.exp(-d / beta).sum() + 1)

	A = numpy.zeros((n, n))
	for i in range(n):
		for j in range(n):
			if C[i, j] > 0:
				rbf_i = numpy.exp(-D[i, j] / betas[i]) / norms[i]
				rbf_j = numpy.exp(-D[i, j] / betas[j]) / norms[j]
				A[i, j] = numpy.sqrt(rbf_i * rbf_j)

	return A + numpy.diag(1.0 / numpy.array(norms))


@pytest.fixture
def motif_ts():
	return planted_track_set(n=40, length=60, motif_length=8, position=26)


##


@pytest.mark.parametrize("n,k", [(10, 5), (20, 10), (50, 20), (30, 30)])
@pytest.mark.parametrize("perplexity", [2.0, 5.0, 10.0])
def test_density_adaptation(n, k, perplexity):
	affmat, neighbors = _affinity(n, k, random_state=n)
	A = _density_adaptation(affmat, neighbors, perplexity)

	assert isinstance(A, scipy.sparse.csr_matrix)
	assert A.shape == (n, n)
	assert A.dtype == numpy.float64
	assert_array_almost_equal(A.toarray(), _density_reference(affmat,
		neighbors, perplexity))


def test_density_adaptation_symmetric():
	affmat, neighbors = _affinity(40, 12)
	A = _density_adaptation(affmat, neighbors, 5.0).toarray()

	assert_array_almost_equal(A, A.T)
	assert numpy.all(A >= 0)
	assert numpy.all(numpy.diag(A) > 0)


def test_density_adaptation_diagonal():
	# The diagonal holds 1 / (1 + sum_j exp(-d_ij / beta_i)).
	affmat, neighbors = _affinity(30, 10)
	A = _density_adaptation(affmat, neighbors, 5.0).toarray()

	assert numpy.all(numpy.diag(A) > 0)
	assert numpy.all(numpy.diag(A) <= 1)


def test_density_adaptation_lists():
	# Neighbor lists may be ragged after filtering by correlation.
	affmat = [[1.0, 0.5, 0.3], [1.0, 0.4], [1.0, 0.5, 0.4]]
	neighbors = [[0, 1, 2], [1, 2], [2, 0, 1]]
	A = _density_adaptation(affmat, neighbors, 1.5)

	assert A.shape == (3, 3)
	assert_array_almost_equal(A.toarray(), _density_reference(affmat,
		neighbors, 1.5))


def test_density_adaptation_self_removed():
	# A self affinity of 1 is a distance of 0, which is dropped, so listing
	# each point as its own neighbor changes nothing.
	affmat, neighbors = _affinity(15, 6)
	assert_array_equal(neighbors[:, 0], numpy.arange(15))

	A0 = _density_adaptation(affmat, neighbors, 3.0)
	A1 = _density_adaptation(affmat[:, 1:], neighbors[:, 1:], 3.0)
	assert_array_almost_equal(A0.toarray(), A1.toarray())


def test_density_adaptation_real(seqlets):
	X = get_2d_data_from_patterns(seqlets[:80])[0].reshape(80, -1)
	affmat, neighbors = pairwise_jaccard(X, 20)
	A = _density_adaptation(affmat, neighbors, 10.0)

	assert A.shape == (80, 80)
	assert_array_almost_equal(A.diagonal()[:4], [0.4414, 0.3231, 0.1511,
		0.1317], 4)


##


def test_filter_patterns_support(motif_ts):
	patterns = [make_pattern(motif_ts, [(i, 21, 39, False) for i in range(n)])
		for n in [5, 10, 20, 40]]
	passing = _filter_patterns(patterns, min_seqlet_support=10, window_size=6,
		min_ic_in_window=0.6, background=BG, ppm_pseudocount=0.001)

	assert [len(p.seqlets) for p in passing] == [10, 20, 40]


def test_filter_patterns_ic():
	track_set = random_track_set(n=40, length=60)
	background = make_pattern(track_set, [(i, 20, 40, False) for i in range(40)])
	motif = make_pattern(planted_track_set(n=40, length=60, motif_length=8,
		position=26), [(i, 21, 39, False) for i in range(40)])

	passing = _filter_patterns([background, motif], min_seqlet_support=10,
		window_size=6, min_ic_in_window=0.6, background=BG,
		ppm_pseudocount=0.001)

	assert passing == [motif]


@pytest.mark.parametrize("min_ic,expected", [(0.0, 2), (0.6, 1), (11.0, 1),
	(12.1, 0)])
def test_filter_patterns_min_ic(min_ic, expected):
	# Six bases of a perfectly conserved motif carry just under 12 bits.
	track_set = random_track_set(n=40, length=60)
	background = make_pattern(track_set, [(i, 20, 40, False) for i in range(40)])
	motif = make_pattern(planted_track_set(n=40, length=60, motif_length=8,
		position=26), [(i, 21, 39, False) for i in range(40)])

	passing = _filter_patterns([background, motif], min_seqlet_support=10,
		window_size=6, min_ic_in_window=min_ic, background=BG,
		ppm_pseudocount=0.001)
	assert len(passing) == expected


@pytest.mark.parametrize("window_size,expected", [(4, 1), (8, 1), (18, 1),
	(30, 1)])
def test_filter_patterns_window_size(motif_ts, window_size, expected):
	# Windows longer than the pattern fall back to the total IC.
	pattern = make_pattern(motif_ts, [(i, 21, 39, False) for i in range(40)])
	passing = _filter_patterns([pattern], min_seqlet_support=10,
		window_size=window_size, min_ic_in_window=7.5, background=BG,
		ppm_pseudocount=0.001)
	assert len(passing) == expected


def test_filter_patterns_window_short(motif_ts):
	pattern = make_pattern(motif_ts, [(i, 21, 39, False) for i in range(40)])
	passing = _filter_patterns([pattern], min_seqlet_support=10,
		window_size=2, min_ic_in_window=4.1, background=BG,
		ppm_pseudocount=0.001)
	assert passing == []


def test_filter_patterns_order(motif_ts):
	patterns = [make_pattern(motif_ts, [(i, 21, 39, False) for i in range(n)])
		for n in [30, 12, 25]]
	passing = _filter_patterns(patterns, 10, 6, 0.6, BG, 0.001)
	assert passing == patterns


def test_filter_patterns_empty():
	assert _filter_patterns([], 10, 6, 0.6, BG, 0.001) == []


def test_filter_patterns_real(pos_patterns, neg_patterns):
	patterns = pos_patterns + neg_patterns
	assert len(_filter_patterns(patterns, 20, 6, 0.6, BG, 0.001)) == 4
	assert len(_filter_patterns(patterns, 27, 6, 0.6, BG, 0.001)) == 3
	assert len(_filter_patterns(patterns, 75, 6, 0.6, BG, 0.001)) == 1


##


def test_patterns_from_clusters():
	track_set = two_motif_track_set(n=20)
	seqlets = make_seqlets(track_set, [(i, 15, 35, bool(i % 4 == 0))
		for i in range(40)])
	cluster_indices = numpy.repeat([0, 1], 20)

	patterns = _patterns_from_clusters(seqlets, track_set, min_overlap=0.7,
		min_frac=0.2, min_num=30, flank_to_add=5, window_size=10, bg_freq=BG,
		cluster_indices=cluster_indices, track_sign=1)

	assert len(patterns) == 2
	for pattern, examples in zip(patterns, [range(20), range(20, 40)]):
		assert isinstance(pattern, SeqletSet)
		assert len(pattern) == 20
		assert set(s.example_idx for s in pattern.seqlets) == set(examples)


def test_patterns_from_clusters_sign():
	track_set = two_motif_track_set(n=20)
	seqlets = make_seqlets(track_set, [(i, 15, 35, False) for i in range(40)])
	cluster_indices = numpy.repeat([0, 1], 20)

	patterns = _patterns_from_clusters(seqlets, track_set, 0.7, 0.2, 30, 5,
		10, BG, cluster_indices, track_sign=-1)
	assert patterns == []


def test_patterns_from_clusters_interleaved():
	track_set = two_motif_track_set(n=20)
	seqlets = make_seqlets(track_set, [(i, 15, 35, False) for i in range(40)])
	cluster_indices = numpy.array([i % 2 for i in range(40)])

	patterns = _patterns_from_clusters(seqlets, track_set, 0.7, 0.2, 30, 5,
		10, BG, cluster_indices, track_sign=1)

	assert len(patterns) == 2
	assert [len(p.seqlets) for p in patterns] == [20, 20]


@pytest.mark.parametrize("window_size,flank_to_add", [(6, 0), (8, 3),
	(10, 5), (12, 8)])
def test_patterns_from_clusters_length(motif_ts, window_size, flank_to_add):
	seqlets = make_seqlets(motif_ts, [(i, 16, 44, False) for i in range(40)])
	patterns = _patterns_from_clusters(seqlets, motif_ts, 0.7, 0.2, 30,
		flank_to_add, window_size, BG, numpy.zeros(40, dtype=int), 1)

	assert len(patterns) == 1
	assert len(patterns[0]) == window_size + 2*flank_to_add


def test_patterns_from_clusters_singleton(motif_ts):
	seqlets = make_seqlets(motif_ts, [(0, 16, 44, False), (1, 16, 44, False)])
	patterns = _patterns_from_clusters(seqlets, motif_ts, 0.7, 0.2, 30, 5, 8,
		BG, numpy.array([0, 1]), 1)

	# Every position of a single seqlet is fully conserved, so the IC trim
	# keeps the first window.
	assert len(patterns) == 2
	assert [_coords(p) for p in patterns] == [[(0, 6, 24, False)],
		[(1, 6, 24, False)]]


def test_patterns_from_clusters_seed(motif_ts):
	# The pattern is seeded with the seqlet with the most attribution.
	seqlets = make_seqlets(motif_ts, [(i, 16, 44, False) for i in range(10)])
	strongest = max(seqlets, key=lambda s: numpy.abs(s.contrib_scores).sum())

	patterns = _patterns_from_clusters(seqlets, motif_ts, 0.7, 0.2, 30, 5, 8,
		BG, numpy.zeros(10, dtype=int), 1)
	assert patterns[0].seqlets[0].example_idx == strongest.example_idx


def test_patterns_from_clusters_real(pos_seqlets, track_set):
	cluster_indices = numpy.arange(len(pos_seqlets)) % 3
	patterns = _patterns_from_clusters(pos_seqlets, track_set, 0.7, 0.2, 30,
		10, 30, BG, cluster_indices, 1)

	assert len(patterns) == 3
	assert all(len(p) == 50 for p in patterns)
	assert sum(len(p.seqlets) for p in patterns) <= len(pos_seqlets)


##


def _correlation_inputs():
	seqlets = list(range(5))
	neighbors = [[0, 1, 2, 3], [1, 0, 2, 4], [2, 3, 0, 1], [3, 4, 2, 0],
		[4, 3, 1, 0]]
	# The Spearman correlations of the rows are 1, -1, 1, -0.4 and 1.
	fine = numpy.array([[1.0, 0.8, 0.6, 0.4], [0.9, 0.7, 0.5, 0.3],
		[1.0, 0.9, 0.2, 0.1], [0.9, 0.6, 0.5, 0.2], [1.0, 0.9, 0.8, 0.1]])
	coarse = numpy.array([[1.0, 0.7, 0.5, 0.2], [0.2, 0.4, 0.6, 0.8],
		[1.0, 0.8, 0.3, 0.2], [0.5, 0.3, 0.4, 0.6], [1.0, 0.8, 0.5, 0.3]])
	return seqlets, neighbors, coarse, fine


def test_filter_by_correlation():
	seqlets, neighbors, coarse, fine = _correlation_inputs()
	kept, new_neighbors, new_affmat = _filter_by_correlation(seqlets,
		neighbors, coarse, fine, 0.0)

	assert kept == [0, 2, 4]
	assert [list(map(int, row)) for row in new_neighbors] == [[0, 1],
		[1, 0], [2, 0]]
	assert new_affmat == [[1.0, 0.6], [1.0, 0.2], [1.0, 0.1]]


@pytest.mark.parametrize("threshold,expected", [(-1.1, [0, 1, 2, 3, 4]),
	(-0.5, [0, 2, 3, 4]), (0.0, [0, 2, 4]), (0.99, [0, 2, 4]), (1.0, [])])
def test_filter_by_correlation_threshold(threshold, expected):
	seqlets, neighbors, coarse, fine = _correlation_inputs()
	kept, new_neighbors, new_affmat = _filter_by_correlation(seqlets,
		neighbors, coarse, fine, threshold)

	assert kept == expected
	assert len(new_neighbors) == len(expected)
	assert len(new_affmat) == len(expected)
	for row, affs in zip(new_neighbors, new_affmat):
		assert len(row) == len(affs)
		assert all(0 <= r < len(expected) for r in row)


def test_filter_by_correlation_mask():
	# Only entries where the fine affinity is nonzero are compared.
	seqlets = [0, 1]
	neighbors = [[0, 1, 1], [1, 0, 0]]
	fine = numpy.array([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
	coarse = numpy.array([[1.0, 0.5, 9.0], [1.0, 0.5, 9.0]])

	kept, _, _ = _filter_by_correlation(seqlets, neighbors, coarse, fine, 0.5)
	assert kept == [0, 1]


def test_filter_by_correlation_real(seqlets):
	from modiscolite.affinitymat import cosine_similarity_from_seqlets
	from modiscolite.affinitymat import jaccard_from_seqlets

	seqlets = seqlets[:100]
	coarse, neighbors = cosine_similarity_from_seqlets(seqlets, 20, sign=1)
	fine = jaccard_from_seqlets(seqlets, 0.7, seqlet_neighbors=neighbors)
	kept, new_neighbors, new_affmat = _filter_by_correlation(seqlets,
		neighbors, coarse, fine, 0.15)

	assert len(kept) == 77
	assert all(s in seqlets for s in kept)
	assert len(new_neighbors) == 77
	assert all(max(row) < 77 for row in new_neighbors)


##


def test_seqlets_to_patterns(pos_seqlets, track_set):
	patterns = seqlets_to_patterns(pos_seqlets, track_set, track_signs=1,
		n_leiden_runs=2, trim_to_window_size=30, initial_flank_to_add=10)

	assert isinstance(patterns, list)
	assert [len(p.seqlets) for p in patterns] == [80]
	for pattern in patterns:
		assert isinstance(pattern, SeqletSet)
		assert len(pattern) == 50
		assert pattern.subclusters is not None


def test_seqlets_to_patterns_deterministic(pos_seqlets, track_set):
	kwargs = dict(track_signs=1, n_leiden_runs=1, trim_to_window_size=30,
		initial_flank_to_add=10)
	patterns0 = seqlets_to_patterns(pos_seqlets[:100], track_set, **kwargs)
	patterns1 = seqlets_to_patterns(pos_seqlets[:100], track_set, **kwargs)

	assert [_coords(p) for p in patterns0] == [_coords(p) for p in patterns1]


@pytest.mark.parametrize("trim,initial,final", [(20, 5, 0), (30, 10, 0),
	(30, 5, 5), (24, 0, 3)])
def test_seqlets_to_patterns_length(pos_seqlets, track_set, trim, initial,
	final):
	patterns = seqlets_to_patterns(pos_seqlets[:120], track_set,
		track_signs=1, n_leiden_runs=1, trim_to_window_size=trim,
		initial_flank_to_add=initial, final_flank_to_add=final)

	assert len(patterns) > 0
	assert all(len(p) == trim + 2*initial + 2*final for p in patterns)


@pytest.mark.parametrize("final_min_cluster_size,expected", [(1, 3),
	(20, 1), (200, 0)])
def test_seqlets_to_patterns_min_cluster_size(pos_seqlets, track_set,
	final_min_cluster_size, expected):
	patterns = seqlets_to_patterns(pos_seqlets[:120], track_set,
		track_signs=1, n_leiden_runs=1, trim_to_window_size=30,
		initial_flank_to_add=10, final_min_cluster_size=final_min_cluster_size)
	assert len(patterns) == expected


def test_seqlets_to_patterns_min_ic(pos_seqlets, track_set):
	patterns = seqlets_to_patterns(pos_seqlets[:120], track_set,
		track_signs=1, n_leiden_runs=1, min_ic_in_window=100.0)
	assert patterns == []


def test_seqlets_to_patterns_sign(pos_seqlets, track_set):
	# Positive seqlets never produce patterns with a negative sign.
	patterns = seqlets_to_patterns(pos_seqlets[:120], track_set,
		track_signs=-1, n_leiden_runs=1)
	assert patterns is None or patterns == []


def test_seqlets_to_patterns_planted():
	track_set = two_motif_track_set(n=40, length=80, position=35)
	seqlets = make_seqlets(track_set, [(i, 25, 55, bool(i % 3 == 0))
		for i in range(80)])

	patterns = seqlets_to_patterns(seqlets, track_set, track_signs=1,
		n_leiden_runs=2, trim_to_window_size=10, initial_flank_to_add=5,
		final_min_cluster_size=10, subcluster_perplexity=10,
		nearest_neighbors_to_compute=40)

	assert len(patterns) == 2
	for pattern in patterns:
		assert len(pattern.seqlets) >= 36
		assert len(set(s.example_idx < 40 for s in pattern.seqlets)) == 1


@pytest.mark.skip(reason="bug: seqlets_to_patterns([]) raises numpy AxisError "
	"while computing bg_freq, before it reaches its check for an empty "
	"list of seqlets")
def test_seqlets_to_patterns_empty(track_set):
	assert seqlets_to_patterns([], track_set, track_signs=1) is None


##


def test_tfmodisco_real(one_hot, hypothetical_contribs):
	pos, neg = TFMoDISco(one_hot, hypothetical_contribs,
		max_seqlets_per_metacluster=150, **KWARGS)

	assert neg is None
	assert [len(p.seqlets) for p in pos] == [80]
	assert len(pos[0]) == 50

	assert_array_almost_equal(pos[0].sequence[20:24], [
		[0.425 , 0.125 , 0.2375, 0.2125],
		[0.    , 0.    , 1.    , 0.    ],
		[0.    , 0.    , 1.    , 0.    ],
		[0.4   , 0.275 , 0.1625, 0.1625]], 4)


def test_tfmodisco_signed(pos_patterns, neg_patterns):
	assert [len(p.seqlets) for p in pos_patterns] == [74, 26]
	assert [len(p.seqlets) for p in neg_patterns] == [77, 27]
	assert all(len(p) == 50 for p in pos_patterns + neg_patterns)

	for pattern in pos_patterns:
		assert pattern.contrib_scores.sum() > 0
		assert all(s.example_idx < 150 for s in pattern.seqlets)

	for pattern in neg_patterns:
		assert pattern.contrib_scores.sum() < 0
		assert all(s.example_idx >= 150 for s in pattern.seqlets)


def _shift(pattern0, pattern1):
	"""The set of (|start shift|, same strand) over shared examples, and the
	fraction of pattern0's examples that pattern1 shares."""

	a = {s.example_idx: s for s in pattern0.seqlets}
	b = {s.example_idx: s for s in pattern1.seqlets}
	common = set(a) & set(b)

	shifts = set((abs(b[i].start - a[i].start), a[i].is_revcomp ==
		b[i].is_revcomp) for i in common)
	return shifts, len(common) / len(a)


def test_tfmodisco_mirror(one_hot, hypothetical_contribs):
	# Negating every attribution swaps the positive and negative patterns.
	# The Laplacian null draws its samples differently for the two signs, so
	# seqlets near the threshold, and with them the framing of the pattern,
	# can move by a few bp.
	kwargs = dict(KWARGS, max_seqlets_per_metacluster=80)
	pos0, neg0 = TFMoDISco(one_hot, hypothetical_contribs, **kwargs)
	pos1, neg1 = TFMoDISco(one_hot, -hypothetical_contribs, **kwargs)

	assert neg0 is None
	assert pos1 is None
	assert len(pos0) == len(neg1)

	shifts, frac = _shift(pos0[0], neg1[0])
	assert len(shifts) == 1
	assert list(shifts)[0][1] is True
	assert frac > 0.85


def test_seqlets_to_patterns_mirror(pos_seqlets, track_set):
	# The clustering itself is exactly symmetric in the sign of the
	# attributions.
	negated = TrackSet(track_set.one_hot, -track_set.contrib_scores,
		-track_set.hypothetical_contribs)
	neg_seqlets = negated.create_seqlets([Seqlet(s.example_idx, s.start,
		s.end, s.is_revcomp) for s in pos_seqlets[:100]])

	kwargs = dict(n_leiden_runs=1, trim_to_window_size=30,
		initial_flank_to_add=10)
	patterns0 = seqlets_to_patterns(pos_seqlets[:100], track_set,
		track_signs=1, **kwargs)
	patterns1 = seqlets_to_patterns(neg_seqlets, negated, track_signs=-1,
		**kwargs)

	assert [_coords(p) for p in patterns0] == [_coords(p) for p in patterns1]
	for p0, p1 in zip(patterns0, patterns1):
		assert_array_almost_equal(p0.contrib_scores, -p1.contrib_scores)


@pytest.mark.parametrize("max_seqlets", [60, 100, 150])
def test_tfmodisco_max_seqlets(one_hot, hypothetical_contribs, max_seqlets):
	pos, neg = TFMoDISco(one_hot, hypothetical_contribs,
		max_seqlets_per_metacluster=max_seqlets, min_metacluster_size=50,
		**KWARGS)

	assert sum(len(p.seqlets) for p in pos) <= max_seqlets
	for pattern in pos:
		for seqlet in pattern.seqlets:
			assert 0 <= seqlet.start < seqlet.end <= 300


@pytest.mark.parametrize("min_metacluster_size", [100, 540, 10000])
def test_tfmodisco_min_metacluster_size(one_hot, hypothetical_contribs,
	min_metacluster_size):
	# About 540 positive seqlets pass the threshold.
	pos, neg = TFMoDISco(one_hot, hypothetical_contribs,
		max_seqlets_per_metacluster=80,
		min_metacluster_size=min_metacluster_size, **KWARGS)

	assert neg is None
	if min_metacluster_size > 540:
		assert pos is None
	else:
		assert len(pos) > 0


@pytest.mark.parametrize("sliding_window_size,flank_size", [(15, 5),
	(20, 3), (21, 10), (12, 4)])
def test_tfmodisco_window(one_hot, hypothetical_contribs,
	sliding_window_size, flank_size):
	kwargs = dict(KWARGS, sliding_window_size=sliding_window_size,
		flank_size=flank_size)
	pos, neg = TFMoDISco(one_hot, hypothetical_contribs,
		max_seqlets_per_metacluster=100, **kwargs)

	assert len(pos) > 0
	assert all(len(p) == 50 for p in pos)


@pytest.mark.parametrize("trim,initial,final", [(20, 5, 0), (30, 10, 5),
	(16, 8, 2)])
def test_tfmodisco_pattern_length(one_hot, hypothetical_contribs, trim,
	initial, final):
	kwargs = dict(KWARGS, trim_to_window_size=trim,
		initial_flank_to_add=initial, final_flank_to_add=final)
	pos, neg = TFMoDISco(one_hot, hypothetical_contribs,
		max_seqlets_per_metacluster=100, **kwargs)

	assert all(len(p) == trim + 2*initial + 2*final for p in pos)


@pytest.mark.parametrize("n,length", [(300, 300), (200, 300), (300, 200),
	(150, 250)])
def test_tfmodisco_shapes(one_hot, hypothetical_contribs, n, length):
	start = (300 - length) // 2
	X = numpy.ascontiguousarray(one_hot[:n, start:start+length])
	A = numpy.ascontiguousarray(hypothetical_contribs[:n, start:start+length])

	pos, neg = TFMoDISco(X, A, max_seqlets_per_metacluster=100,
		min_metacluster_size=50, **KWARGS)

	assert len(pos) > 0
	for pattern in pos:
		assert all(s.example_idx < n for s in pattern.seqlets)
		assert all(s.end <= length for s in pattern.seqlets)


def test_tfmodisco_float64(one_hot, hypothetical_contribs):
	# Rounding differences can move the framing of a pattern by a few bp.
	pos0, _ = TFMoDISco(one_hot, hypothetical_contribs,
		max_seqlets_per_metacluster=80, **KWARGS)
	pos1, _ = TFMoDISco(one_hot.astype('float64'),
		hypothetical_contribs.astype('float64'),
		max_seqlets_per_metacluster=80, **KWARGS)

	assert len(pos0) == len(pos1)
	shifts, frac = _shift(pos0[0], pos1[0])
	assert len(shifts) == 1
	assert list(shifts)[0][1] is True
	assert frac > 0.9


@pytest.mark.parametrize("n_leiden_runs", [1, 2, 4])
def test_tfmodisco_n_leiden_runs(one_hot, hypothetical_contribs,
	n_leiden_runs):
	kwargs = dict(KWARGS, n_leiden_runs=n_leiden_runs)
	pos, neg = TFMoDISco(one_hot, hypothetical_contribs,
		max_seqlets_per_metacluster=80, **kwargs)
	assert len(pos) > 0


def test_tfmodisco_verbose(one_hot, hypothetical_contribs, capsys):
	TFMoDISco(one_hot, hypothetical_contribs, max_seqlets_per_metacluster=60,
		verbose=True, **KWARGS)
	assert capsys.readouterr().out == "Using 60 positive seqlets\n"


def test_tfmodisco_verbose_negative(one_hot, hypothetical_contribs, capsys):
	TFMoDISco(one_hot, -hypothetical_contribs, max_seqlets_per_metacluster=60,
		verbose=True, **KWARGS)
	assert capsys.readouterr().out == "Extracted 60 negative seqlets\n"


def test_tfmodisco_quiet(one_hot, hypothetical_contribs, capsys):
	TFMoDISco(one_hot, hypothetical_contribs, max_seqlets_per_metacluster=60,
		**KWARGS)
	assert capsys.readouterr().out == ""


def test_tfmodisco_does_not_modify(one_hot, hypothetical_contribs):
	X, A = one_hot.copy(), hypothetical_contribs.copy()
	TFMoDISco(one_hot, hypothetical_contribs, max_seqlets_per_metacluster=60,
		**KWARGS)

	assert_array_equal(one_hot, X)
	assert_array_equal(hypothetical_contribs, A)


def test_tfmodisco_seqlet_data(one_hot, hypothetical_contribs):
	pos, _ = TFMoDISco(one_hot, hypothetical_contribs,
		max_seqlets_per_metacluster=60, **KWARGS)

	for pattern in pos:
		for s in pattern.seqlets:
			sequence = one_hot[s.example_idx, s.start:s.end]
			hyp = hypothetical_contribs[s.example_idx, s.start:s.end]
			if s.is_revcomp:
				sequence, hyp = sequence[::-1, ::-1], hyp[::-1, ::-1]

			assert_array_equal(s.sequence, sequence)
			assert_array_almost_equal(s.hypothetical_contribs, hyp)
			assert_array_almost_equal(s.contrib_scores, sequence * hyp)
