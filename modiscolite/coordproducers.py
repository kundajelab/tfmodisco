# coordproducers.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code originally written by Avanti Shrikumar 

import numpy as np

from .core import SeqletCoordinates
from .value_provider import AbsPercentileValTransformer

from sklearn.isotonic import IsotonicRegression

def _bin_mode(values, bins=1000):
	hist, bin_edges = np.histogram(values, bins=bins)
	peak = np.argmax(hist)
	l_edge = bin_edges[peak]
	r_edge = bin_edges[peak+1]
	return l_edge, r_edge, values[(l_edge < values) & (values < r_edge)]

def LaplaceNullDist(original_summed_score_track, window_size, num_to_samp, 
	verbose=True, percentiles_to_use=np.array([5*(x+1) for x in range(19)]),
	random_seed=1234):

	rng = np.random.RandomState()
	values = np.concatenate(original_summed_score_track, axis=0)

	# first estimate mu, using two level histogram to get to 1e-6
	_, _, top_values = _bin_mode(values)
	l_edge, r_edge, _ = _bin_mode(top_values)
	mu = (l_edge + r_edge) / 2

	pos_values = values[values >= mu]
	neg_values = values[values <=mu] 

	#Take the most aggressive lambda over all percentiles
	pos_laplace_lambda = np.max(
		-np.log(1-(percentiles_to_use/100.0))/
		(np.percentile(a=pos_values, q=percentiles_to_use)-mu))

	neg_laplace_lambda = np.max(
		-np.log(1-(percentiles_to_use/100.0))/
		(np.abs(np.percentile(a=neg_values,
							  q=100-percentiles_to_use)-mu)))

	rng.seed(random_seed)
	prob_pos = float(len(pos_values))/(len(pos_values)+len(neg_values)) 
	sampled_vals = []

	for i in range(num_to_samp):
		sign = 1 if (rng.uniform() < prob_pos) else -1
		if (sign == 1):
			sampled_cdf = rng.uniform()
			val = -np.log(1-sampled_cdf)/pos_laplace_lambda + mu 
		else:
			sampled_cdf = rng.uniform() 
			val = mu + np.log(1-sampled_cdf)/neg_laplace_lambda
		sampled_vals.append(val)
	return np.array(sampled_vals)

#identify_coords is expecting something that has already been processed
# with sliding windows of size window_size
def _identify_coords(score_track, pos_threshold, neg_threshold,
	window_size, flank, suppress, max_seqlets_total, sign_to_return):
	#cp_score_track = 'copy' of the score track, which can be modified as
	# coordinates are identified
	cp_score_track = score_track.copy()

	#if a position is less than the threshold, set it to -np.inf
	#Note that the threshold comparisons need to be >= and not just > for
	# cases where there are lots of ties at the high end (e.g. with an IR
	# tranformation that gives a lot of values that have a precision of 1.0)
	if sign_to_return is None:
		idxs = (cp_score_track >= pos_threshold) | (cp_score_track <= neg_threshold)
	elif sign_to_return == 1:
		idxs = cp_score_track >= pos_threshold
	elif sign_to_return == -1:
		idxs = cp_score_track <= neg_threshold

	cp_score_track[idxs] = np.abs(cp_score_track[idxs])
	cp_score_track[~idxs] = -np.inf

	# Filter out the flanks
	cp_score_track[:, :flank] = -np.inf
	cp_score_track[:, -flank] = -np.inf

	n, d = cp_score_track.shape
	coords = []
	for example_idx, single_score_track in enumerate(cp_score_track):
		while True:
			argmax = np.argmax(single_score_track, axis=0)
			max_val = single_score_track[argmax]

			#bail if exhausted everything that passed the threshold
			#and was not suppressed
			if max_val == -np.inf:
				break

			#need to be able to expand without going off the edge
			if argmax >= flank and argmax < (d-flank): 
				coord = SeqletCoordinates(
					example_idx=example_idx,
					start=argmax-flank,
					end=argmax+window_size+flank,
					revcomp=False,
					score=score_track[example_idx][argmax])

				coords.append(coord)

			#suppress the chunks within +- suppress
			l_idx = int(max(np.floor(argmax+0.5-suppress),0))
			r_idx = int(min(np.ceil(argmax+0.5+suppress), d))
			single_score_track[l_idx:r_idx] = -np.inf 

	if max_seqlets_total is not None and len(coords) > max_seqlets_total:
		coords = sorted(coords, key=lambda x: -np.abs(x.score))[:max_seqlets_total]
	
	return coords


def _smooth_and_split(tracks, window_size, subsample_cap=1000000):
	n = len(tracks)

	tracks = np.hstack([np.zeros((n, 1)), np.cumsum(tracks, axis=-1)])
	tracks = tracks[:, window_size:] - tracks[:, :-window_size]

	values = np.concatenate(tracks, axis=0)
	if len(values) > subsample_cap:
		values = np.random.RandomState(1234).choice(
			a=values, size=subsample_cap, replace=False)
	
	pos_values = values[values >= 0]
	pos_values = np.sort(pos_values)

	neg_values = values[values < 0]
	neg_values = np.sort(neg_values)[::-1]
	return pos_values, neg_values, tracks

def _isotonic_thresholds(values, null_values, increasing, target_fdr, 
	min_frac_neg=0.95):
	n1, n2 = len(values), len(null_values)

	X = np.concatenate([values, null_values], axis=0)
	y = np.concatenate([np.ones(n1), np.zeros(n2)], axis=0)

	w = len(values) / len(null_values)
	sample_weight = np.concatenate([np.ones(n1), np.ones(n2)*w], axis=0)

	model = IsotonicRegression(out_of_bounds='clip', increasing=increasing)
	model.fit(X, y, sample_weight=sample_weight)

	min_prec_x = model.X_min_ if increasing else model.X_max_
	min_precision = model.transform([min_prec_x])[0]
	implied_frac_neg = -1/(1-(1/max(min_precision, 1e-7)))
	if (implied_frac_neg > 1.0 or implied_frac_neg < min_frac_neg):
		implied_frac_neg = max(min(1.0,implied_frac_neg), min_frac_neg)

	precisions = np.minimum(np.maximum(1 + implied_frac_neg*(
		1 - (1 / np.maximum(model.transform(values), 1e-7))), 0.0), 1.0)
	precisions[-1] = 1
	return values[precisions >= (1 - target_fdr)][0]

def _refine_thresholds(vals, pos_threshold, neg_threshold,
	min_passing_windows_frac, max_passing_windows_frac):

	frac_passing_windows =(
		sum(vals >= pos_threshold)
		 + sum(vals <= neg_threshold))/float(len(vals))

	if frac_passing_windows < min_passing_windows_frac:
		pos_threshold = np.percentile(
			a=np.abs(vals),
			q=100*(1-min_passing_windows_frac)) 
		neg_threshold = -pos_threshold

	if frac_passing_windows > max_passing_windows_frac:
		pos_threshold = np.percentile(
			a=np.abs(vals),
			q=100*(1-max_passing_windows_frac)) 
		neg_threshold = -pos_threshold

	return pos_threshold, neg_threshold


def FixedWindowAroundChunks(attribution_scores, window_size, flank, suppress, 
	target_fdr, min_passing_windows_frac, max_passing_windows_frac, 
	max_seqlets_total=None, sign_to_return=None, verbose=True):

	pos_values, neg_values, smoothed_tracks = _smooth_and_split(
		attribution_scores, window_size)

	null_values = LaplaceNullDist(original_summed_score_track=smoothed_tracks,  
		window_size=window_size, num_to_samp=10000)
	pos_null_values = null_values[null_values >= 0]
	neg_null_values = null_values[null_values < 0]

	pos_threshold = _isotonic_thresholds(pos_values, pos_null_values, 
		increasing=True, target_fdr=target_fdr)
	neg_threshold = _isotonic_thresholds(neg_values, neg_null_values,
		increasing=False, target_fdr=target_fdr)

	pos_threshold, neg_threshold = _refine_thresholds(
		  vals=np.concatenate([pos_values, neg_values], axis=0),
		  pos_threshold=pos_threshold,
		  neg_threshold=neg_threshold,
		  min_passing_windows_frac=min_passing_windows_frac,
		  max_passing_windows_frac=max_passing_windows_frac) 

	val_transformer = AbsPercentileValTransformer(
		distribution=np.concatenate(smoothed_tracks, axis=0))

	coords = _identify_coords(
		score_track=smoothed_tracks,
		pos_threshold=pos_threshold,
		neg_threshold=neg_threshold,
		window_size=window_size,
		flank=flank,
		suppress=suppress,
		max_seqlets_total=max_seqlets_total,
		sign_to_return=sign_to_return)

	return {
		'coords': coords,
		'pos_threshold': pos_threshold,
		'neg_threshold': neg_threshold,
		'transformed_pos_threshold': val_transformer(pos_threshold),
		'transformed_neg_threshold': val_transformer(neg_threshold),
		'val_transformer': val_transformer,
	} 
