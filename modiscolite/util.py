import numpy as np
import h5py
import scipy.sparse

from numba import njit

def coo_matrix_from_neighborsformat(entries, neighbors, ncols):
	coo_mat = scipy.sparse.coo_matrix(
			(np.concatenate(entries, axis=0),
			 (np.array([i for i in range(len(neighbors))
						   for j in neighbors[i]]).astype("int"),
			  np.concatenate(neighbors, axis=0)) ),
			shape=(len(entries), ncols)) 
	return coo_mat


def save_string_list(string_list, dset_name, grp):
	dset = grp.create_dataset(dset_name, (len(string_list),),
							  dtype=h5py.special_dtype(vlen=bytes))
	dset[:] = string_list

def save_patterns(patterns, grp):
    all_pattern_names = []
    for idx, pattern in enumerate(patterns):
        pattern_name = "pattern_"+str(idx)
        all_pattern_names.append(pattern_name)
        pattern_grp = grp.create_group(pattern_name) 
        pattern.save_hdf5(pattern_grp)
    save_string_list(all_pattern_names, dset_name="all_pattern_names",
                     grp=grp)

def save_seqlet_coords(seqlets, dset_name, grp):
	coords_strings = [str(x.coor) for x in seqlets] 
	save_string_list(string_list=coords_strings,
					 dset_name=dset_name, grp=grp)

def save_seqlet_coords(seqlets, dset_name, grp):
	coords_strings = [str(x.coor) for x in seqlets] 
	save_string_list(string_list=coords_strings,
					 dset_name=dset_name, grp=grp)

def save_list_of_objects(grp, list_of_objects):
	grp.attrs["num_objects"] = len(list_of_objects) 
	for idx,obj in enumerate(list_of_objects):
		obj.save_hdf5(grp=grp.create_group("obj"+str(idx)))

#TODO: this can prob be replaced with np.sum(
# util.rolling_window(a=arr, window=window_size), axis=-1)  
def cpu_sliding_window_sum(arr, window_size):
	assert len(arr) >= window_size, str(len(arr))+" "+str(window_size)
	to_return = np.zeros(len(arr)-window_size+1)
	current_sum = np.sum(arr[0:window_size])
	to_return[0] = current_sum
	idx_to_include = window_size
	idx_to_exclude = 0
	while idx_to_include < len(arr):
		current_sum += (arr[idx_to_include] - arr[idx_to_exclude]) 
		to_return[idx_to_exclude+1] = current_sum
		idx_to_include += 1
		idx_to_exclude += 1
	return to_return


@njit('float64(float64, float64[:])')
def binary_search_perplexity(desired_perplexity, distances):
	EPSILON_DBL = 1e-8
	PERPLEXITY_TOLERANCE = 1e-5
	n_steps = 100
	
	desired_entropy = np.log(desired_perplexity)
	
	beta_min = -np.inf
	beta_max = np.inf
	beta = 1.0
	
	for l in range(n_steps):
		ps = np.exp(-distances * beta)
		sum_ps = np.sum(ps) + 1
		ps = ps/(max(sum_ps,EPSILON_DBL))
		sum_disti_Pi = np.sum(distances*ps)
		entropy = np.log(sum_ps) + beta * sum_disti_Pi
		
		entropy_diff = entropy - desired_entropy
		if np.abs(entropy_diff) <= PERPLEXITY_TOLERANCE:
			break
		
		if entropy_diff > 0.0:
			beta_min = beta
			if beta_max == np.inf:
				beta *= 2.0
			else:
				beta = (beta + beta_max) / 2.0
		else:
			beta_max = beta
			if beta_min == -np.inf:
				beta /= 2.0
			else:
				beta = (beta + beta_min) / 2.0
	
	return beta


def compute_per_position_ic(ppm, background, pseudocount):
	"""Compute information content at each position of ppm.

	Arguments:
		ppm: should have dimensions of length x alphabet. Entries along the
			alphabet axis should sum to 1.
		background: the background base frequencies
		pseudocount: pseudocount to be added to the probabilities of the ppm
			to prevent overflow/underflow.

	Returns:
		total information content at each positon of the ppm.
	"""

	if (not np.allclose(np.sum(ppm, axis=1), 1.0, atol=1.0e-5)):
		ppm = ppm/np.sum(ppm, axis=1)[:,None]

	alphabet_len = len(background)
	ic = ((np.log((ppm+pseudocount)/(1 + pseudocount*alphabet_len))/np.log(2))
		  *ppm - (np.log(background)*background/np.log(2))[None,:])
	return np.sum(ic,axis=1)


#rolling_window is from this blog post by Erik Rigtorp:
# https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
#The last axis of a will be subject to the windowing
def rolling_window(a, window):
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def subsample_pattern(pattern, num_to_subsample):
	from . import core
	seqlets_and_alnmts_list = list(pattern._seqlets_and_alnmts)
	subsample = [seqlets_and_alnmts_list[i]
				 for i in
				 np.random.RandomState(1234).choice(
					 a=np.arange(len(seqlets_and_alnmts_list)),
					 replace=False,
					 size=num_to_subsample)]
	return core.AggregatedSeqlet(seqlets_and_alnmts_arr=subsample) 

