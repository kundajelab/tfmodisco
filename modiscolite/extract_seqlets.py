# extract_seqlets.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import numpy as np

from . import core
from sklearn.isotonic import IsotonicRegression

def _bin_mode(values, bins=1000):
	hist, bin_edges = np.histogram(values, bins=bins)
	peak = np.argmax(hist)
	l_edge = bin_edges[peak]
	r_edge = bin_edges[peak+1]
	return l_edge, r_edge, values[(l_edge < values) & (values < r_edge)]

def _laplacian_null(track, window_size, num_to_samp, random_seed=1234):
	percentiles_to_use = np.array([5*(x+1) for x in range(19)])

	rng = np.random.RandomState()
	values = np.concatenate(track, axis=0)

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
		if sign == 1:
			sampled_cdf = rng.uniform()
			val = -np.log(1-sampled_cdf)/pos_laplace_lambda + mu 
		else:
			sampled_cdf = rng.uniform() 
			val = mu + np.log(1-sampled_cdf)/neg_laplace_lambda
		sampled_vals.append(val)

	sampled_vals = np.array(sampled_vals)
	return sampled_vals[sampled_vals >= 0], sampled_vals[sampled_vals < 0]


def _iterative_extract_seqlets(score_track, window_size, flank, suppress):
	n, d = score_track.shape
	seqlets = []
	for example_idx, single_score_track in enumerate(score_track):
		while True:
			argmax = np.argmax(single_score_track, axis=0)
			max_val = single_score_track[argmax]

			#bail if exhausted everything that passed the threshold
			#and was not suppressed
			if max_val == -np.inf:
				break

			#need to be able to expand without going off the edge
			if argmax >= flank and argmax < (d-flank): 
				seqlet = core.Seqlet(example_idx=example_idx, 
					start=argmax-flank, end=argmax+window_size+flank,
					is_revcomp=False)

				seqlets.append(seqlet)

			#suppress the chunks within +- suppress
			l_idx = int(max(np.floor(argmax+0.5-suppress),0))
			r_idx = int(min(np.ceil(argmax+0.5+suppress), d))
			single_score_track[l_idx:r_idx] = -np.inf 

	return seqlets


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


def extract_seqlets(attribution_scores, window_size, flank, suppress, 
	target_fdr, min_passing_windows_frac, max_passing_windows_frac, 
	weak_threshold_for_counting_sign):

	pos_values, neg_values, smoothed_tracks = _smooth_and_split(
		attribution_scores, window_size)

	pos_null_values, neg_null_values = _laplacian_null(track=smoothed_tracks, 
		window_size=window_size, num_to_samp=10000)

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

	distribution = np.array(sorted(np.abs(np.concatenate(smoothed_tracks,
		axis=0))))

	transformed_pos_threshold = np.sign(pos_threshold)*np.searchsorted(
		a=distribution, v=abs(pos_threshold))/len(distribution)

	transformed_neg_threshold = np.sign(neg_threshold)*np.searchsorted(
		a=distribution, v=abs(neg_threshold))/len(distribution)

	idxs = (smoothed_tracks >= pos_threshold) | (smoothed_tracks <= neg_threshold)

	smoothed_tracks[idxs] = np.abs(smoothed_tracks[idxs])
	smoothed_tracks[~idxs] = -np.inf

	# Filter out the flanks
	smoothed_tracks[:, :flank] = -np.inf
	smoothed_tracks[:, -flank:] = -np.inf

	seqlets = _iterative_extract_seqlets(score_track=smoothed_tracks,
		window_size=window_size,
		flank=flank,
		suppress=suppress)

	#find the weakest transformed threshold used across all tasks
	weak_thresh = min(min(transformed_pos_threshold, 
		abs(transformed_neg_threshold)) - 0.0001, 
			weak_threshold_for_counting_sign)

	threshold = distribution[int(weak_thresh * len(distribution))]

	return seqlets, threshold
