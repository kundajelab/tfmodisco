# synthetic.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy

from modiscolite.core import Seqlet
from modiscolite.core import SeqletSet
from modiscolite.core import TrackSet


def random_one_hot(shape, random_state=0):
	"""Return a float64 one-hot array with shape (n, length, 4)."""

	n, length = shape
	idxs = numpy.random.RandomState(random_state).randint(4, size=(n, length))
	return numpy.eye(4)[idxs]


def random_track_set(n=8, length=60, random_state=0):
	"""Return a TrackSet over random sequence with Gaussian attributions."""

	one_hot = random_one_hot((n, length), random_state=random_state)
	hypothetical_contribs = numpy.random.RandomState(random_state+1).randn(
		n, length, 4)

	return TrackSet(one_hot=one_hot, contrib_scores=one_hot*hypothetical_contribs,
		hypothetical_contribs=hypothetical_contribs)


def planted_track_set(n=12, length=80, motif_length=8, position=30,
	strength=3.0, random_state=0):
	"""Return a TrackSet with one strong motif at `position` in every example.

	The motif is the same sequence in every example and carries attribution
	`strength` per base, while the background carries small Gaussian noise.
	"""

	rng = numpy.random.RandomState(random_state)
	one_hot = random_one_hot((n, length), random_state=random_state)
	motif = numpy.eye(4)[rng.randint(4, size=motif_length)]
	one_hot[:, position:position+motif_length] = motif

	hypothetical_contribs = rng.randn(n, length, 4) * 0.05
	hypothetical_contribs[:, position:position+motif_length] += motif * strength

	return TrackSet(one_hot=one_hot, contrib_scores=one_hot*hypothetical_contribs,
		hypothetical_contribs=hypothetical_contribs)


def two_motif_track_set(n=20, length=60, motif_length=10, position=20,
	random_state=0):
	"""Return a TrackSet whose first n examples carry one motif and whose
	last n examples carry a different one, both at `position`."""

	ts0 = planted_track_set(n=n, length=length, motif_length=motif_length,
		position=position, random_state=random_state)
	ts1 = planted_track_set(n=n, length=length, motif_length=motif_length,
		position=position, random_state=random_state+100)

	one_hot = numpy.concatenate([ts0.one_hot, ts1.one_hot])
	hypothetical_contribs = numpy.concatenate([ts0.hypothetical_contribs,
		ts1.hypothetical_contribs])

	return TrackSet(one_hot=one_hot, contrib_scores=one_hot*hypothetical_contribs,
		hypothetical_contribs=hypothetical_contribs)


def make_seqlets(track_set, coords):
	"""Create seqlets from (example_idx, start, end, is_revcomp) tuples."""

	return track_set.create_seqlets([Seqlet(*coord) for coord in coords])


def make_pattern(track_set, coords):
	"""Create a SeqletSet from (example_idx, start, end, is_revcomp) tuples."""

	return SeqletSet(make_seqlets(track_set, coords))


def synthetic_patterns(track_set, n_patterns, n_seqlets, length,
	random_state=0):
	"""Return `n_patterns` SeqletSets of `n_seqlets` random seqlets each."""

	rng = numpy.random.RandomState(random_state)
	n = len(track_set.one_hot)

	patterns = []
	for _ in range(n_patterns):
		coords = []
		for _ in range(n_seqlets):
			idx = int(rng.randint(n))
			start = int(rng.randint(track_set.length - length + 1))
			coords.append((idx, start, start+length, bool(rng.randint(2))))

		patterns.append(make_pattern(track_set, coords))

	return patterns


def write_peaks(filename, n, chroms=('chr1', 'chr2', 'chrX'), width=500):
	"""Write `n` narrowPeak rows cycling through `chroms`.

	Row i starts at 1000*i + 17 on chromosome chroms[i % len(chroms)] and has
	score 100 + i. Returns the list of rows as written.
	"""

	rows = []
	for i in range(n):
		start = 1000*i + 17
		rows.append("\t".join(map(str, [chroms[i % len(chroms)], start,
			start+width, "peak_{}".format(i), 100+i, ".", 1.5, 2.5, 3.5,
			width // 2])))

	with open(filename, "w") as outfile:
		outfile.write("\n".join(rows) + "\n")

	return rows
