from __future__ import division, print_function
import os
import signal
import deeplift
import numpy as np
import theano
import theano.tensor.signal.conv
import h5py
import traceback
from sklearn.neighbors.kde import KernelDensity


def save_patterns(patterns, grp):
    for idx, pattern in enumerate(patterns):
        pattern_grp = grp.create_group("pattern_"+str(idx)) 
        pattern.save_hdf5(pattern_grp)


def save_seqlet_coords(seqlets, dset_name, grp):
    dset = grp.create_dataset(dset_name, (len(self.seqlets),),
                              dtype=h5py.special_dtype(vlen=bytes))
    coords_strings = [str(x.coor) for x in seqlets] 
    dset[:] = coords


def factorial(val):
    to_return = 1
    for i in range(1,val+1):
        to_return *= i
    return to_return


def first_curvature_max(values, bins, bandwidth):
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
                [[x,0] for x in values])
    midpoints = np.min(values)+((np.arange(bins)+0.5)
                                *(np.max(values)-np.min(values))/bins)
    densities = np.exp(kde.score_samples([[x,0] for x in midpoints]))

    global_max_x = max(zip(densities,midpoints), key=lambda x: x[0])[1]
    firstd_x, firstd_y = angle_firstd(x_values=midpoints, y_values=densities) 
    secondd_x, secondd_y = firstd(x_values=firstd_x, y_values=firstd_y)
    thirdd_x, thirdd_y = firstd(x_values=secondd_x, y_values=secondd_y)
    #find curvature maxima i.e. points where thirdd crosses 0
    maxima_x = [0.5*(prev_x+after_x) for (prev_x, after_x),(prev_y,after_y)
                  in zip(zip(thirdd_x[0:-1], thirdd_x[1:]),
                         zip(thirdd_y[0:-1], thirdd_y[1:]))
                  if (prev_y > 0 and after_y < 0)
                  and 0.5*(prev_x+after_x)]
    maxima_x_after_global_max = [x for x in maxima_x if x > global_max_x]
    maxima_x_before_global_max = [x for x in maxima_x if x < global_max_x]
    threshold_before = maxima_x_before_global_max[-1] if\
                        len(maxima_x_before_global_max) > 0 else global_max_x
    threshold_after = maxima_x_after_global_max[0] if\
                        len(maxima_x_after_global_max) > 0 else global_max_x

    from matplotlib import pyplot as plt
    hist_y, _, _ = plt.hist(values, bins=100)
    max_y = np.max(hist_y)
    plt.plot(midpoints, densities*(max_y/np.max(densities)))
    plt.plot([threshold_before, threshold_before], [0, max_y])
    plt.plot([threshold_after, threshold_after], [0, max_y])
    plt.show()

    return threshold_before, threshold_after


def cosine_firstd(x_values, y_values):
    x_differences = x_values[1:] - x_values[:-1]
    x_midpoints = 0.5*(x_values[1:] + x_values[:-1])
    y_differences = y_values[1:] - y_values[:-1]
    hypotenueses = np.sqrt(np.square(y_differences) + np.square(x_differences))
    cosine_first_d = x_differences/hypotenueses 
    return x_midpoints, cosine_first_d


def angle_firstd(x_values, y_values):
    x_differences = x_values[1:] - x_values[:-1]
    x_midpoints = 0.5*(x_values[1:] + x_values[:-1])
    y_differences = y_values[1:] - y_values[:-1]
    return x_midpoints, np.arctan2(y_differences, x_differences)


def angle_curvature(x_values, y_values):
    x_midpoints, y_angles = angle_firstd(x_values, y_values)
    y_midpoints = 0.5*(y_values[1:] + y_values[:-1])
    x_midmidpoints, y_anglechange = firstd(x_midpoints, y_angles)
    x_differences = x_midpoints[1:] - x_midpoints[:-1] 
    y_differences = y_midpoints[1:] - y_midpoints[:-1]
    distance_travelled = np.sqrt(np.square(x_differences)+
                                 np.square(y_differences))
    angle_change_w_dist = y_anglechange/distance_travelled
    return x_midmidpoints, angle_change_w_dist


def firstd(x_values, y_values):
    x_differences = x_values[1:] - x_values[:-1]
    x_midpoints = 0.5*(x_values[1:] + x_values[:-1])
    y_differences = y_values[1:] - y_values[:-1]
    rise_over_run = y_differences/x_differences
    return x_midpoints, rise_over_run


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


def identify_peaks(arr):
    #use a state machine to identify peaks
    #"peaks" as defined by larger than neighbours
    #for tied region, take the middle of the tie.
    #return tuples of idx + peak val
    previous_val = None
    potential_peak_start_idx = None
    found_peaks = []
    for idx, val in enumerate(arr):
        if (previous_val is not None):
            if (val > previous_val):
                potential_peak_start_idx = idx
            elif (val < previous_val):
                if (potential_peak_start_idx is not None):
                    #peak found!
                    found_peaks.append(
        (int(0.5*(potential_peak_start_idx+(idx-1))), previous_val))
                potential_peak_start_idx = None
                potential_peak_start_val = None
            else:
                #tie...don't change anything.
                pass
        previous_val = val
    return found_peaks


def create_detector_from_subset_of_sequential_layers(sequential_container,
                                                    idx_of_layer_of_interest,
                                                    channel_indices,
                                                    multipliers_on_channels):
    import deeplift.backend as B
    layers = []  
    #this adds in all the layers preceeding idx_of_layer_of_interest
    #(remember zero-based indexing...)
    for layer_idx in range(idx_of_layer_of_interest):
        layers.append(
         sequential_container.get_layers()[layer_idx].copy_blob_keep_params()
        )
    #add in the layer of interest, but with the channels subsetted to
    #the channels of interest
    layer_to_subset = sequential_container.get_layers()\
                       [idx_of_layer_of_interest]
    assert hasattr(layer_to_subset, "W"), "Layer does not have weights - "\
        +" make sure you have supplied the correct index for the conv layer?"
    subsetted_weights = layer_to_subset.W[channel_indices]
    subsetted_biases = layer_to_subset.b[channel_indices]
    layer_kwargs = layer_to_subset.get_yaml_compatible_object_kwargs()
    layer_kwargs['W'] = subsetted_weights 
    layer_kwargs['b'] = subsetted_biases
    subsetted_layer = layer_to_subset.\
                      load_blob_from_yaml_contents_only(**layer_kwargs)
    layers.append(subsetted_layer)
    #check if the next layer is an activation layer
    layer_after_layer_of_interest =\
      sequential_container.get_layers()[idx_of_layer_of_interest+1]
    if isinstance(layer_after_layer_of_interest, deeplift.blobs.Activation):
        layers.append(layer_after_layer_of_interest.copy_blob_keep_params())
    #multipliers_layer = sequential_container.get_layers()[layer_idx+1].copy_blob_keep_params()
    #add in a layer with a conv filter that is the multipliers
    #need to be reversed because this is doing a convolution, not cross corr
    multipliers_layer = deeplift.blobs.Conv2D(
                    name="multipliers_layer",
                    W=multipliers_on_channels[:,:,::-1,::-1].astype('float32'),
                    b=np.zeros(multipliers_on_channels.shape[0])\
                      .astype('float32'),
                    strides=(1,1), 
                    border_mode=B.BorderMode.valid)
    layers.append(multipliers_layer)
    deeplift.util.connect_list_of_layers(layers)
    layers[-1].build_fwd_pass_vars()
    model_to_return = deeplift.models.SequentialModel(layers=layers)
    model_to_return.get_layers()

    return model_to_return

def get_conv_out_symbolic_var(input_var,
                              set_of_2d_patterns_to_conv_with,
                              normalise_by_magnitude,
                              take_max,
                              mode='full'):
    assert len(set_of_2d_patterns_to_conv_with.shape)==3
    if (normalise_by_magnitude):
        set_of_2d_patterns_to_conv_with =\
         set_of_2d_patterns_to_conv_with/\
          (np.sqrt(np.sum(np.sum(np.square(set_of_2d_patterns_to_conv_with),
                               axis=-1),
                        axis=-1))[:,None,None])
    filters = theano.tensor.as_tensor_variable(
               x=set_of_2d_patterns_to_conv_with,
               name="filters")
    conv_out = theano.tensor.signal.conv.conv2d(
                input=input_var,
                filters=filters,
                border_mode=mode)
    if (normalise_by_magnitude):
        sum_squares_per_pos =\
                   theano.tensor.signal.conv.conv2d(
                    input=theano.tensor.square(input_var),
                    filters=np.ones(set_of_2d_patterns_to_conv_with.shape)\
                            .astype("float32"),
                    border_mode=mode) 
        per_pos_magnitude = theano.tensor.sqrt(sum_squares_per_pos)
        per_pos_magnitude += 0.0000001*(per_pos_magnitude < 0.0000001)
        conv_out = conv_out/per_pos_magnitude
    if (take_max):
        conv_out = theano.tensor.max(
                    theano.tensor.max(conv_out, axis=-1), #max over cols
                    axis=-1) #max over rows
    return conv_out 


def compile_conv_func_with_theano(set_of_2d_patterns_to_conv_with,
                                  normalise_by_magnitude=False,
                                  take_max=False,
                                  mode='full'):
    input_var = theano.tensor.TensorType(dtype=theano.config.floatX,
                                         broadcastable=[False]*3)("input")
    conv_out = get_conv_out_symbolic_var(input_var,
                                 set_of_2d_patterns_to_conv_with,
                                 normalise_by_magnitude=normalise_by_magnitude,
                                 take_max=take_max,
                                 mode=mode)
    func = theano.function([input_var],
                           conv_out,
                           allow_input_downcast=True)
    return func 


def get_max_cross_corr(filters, things_to_scan,
                           verbose=True, batch_size=10,
                           func_params_size=1000000,
                           progress_update=1000,
                           min_overlap=0.3):
    """
        func_params_size: when compiling functions
    """
    #reverse the patterns as the func is a conv not a cross corr
    filters = filters.astype("float32")[:,::-1,::-1]
    to_return = np.zeros((filters.shape[0], len(things_to_scan)))
    #compile the number of filters that result in a function with
    #params equal to func_params_size 
    params_per_filter = np.prod(filters[0].shape)
    filter_batch_size = int(func_params_size/params_per_filter)
    filter_length = filters.shape[-1]
    filter_idx = 0 
    while filter_idx < filters.shape[0]:
        if (verbose):
            print("On filters",filter_idx,"to",(filter_idx+filter_batch_size))
        filter_batch = filters[filter_idx:(filter_idx+filter_batch_size)]
        cross_corr_func = compile_conv_func_with_theano(
                           set_of_2d_patterns_to_conv_with=filter_batch,
                           normalise_by_magnitude=False,
                           take_max=True)  
        padding_amount = int((filter_length)*(1-min_overlap))
        padded_input = [np.pad(array=x,
                              pad_width=((padding_amount, padding_amount)),
                              mode="constant") for x in things_to_scan]
        max_cross_corrs = np.array(deeplift.util.run_function_in_batches(
                            func=cross_corr_func,
                            input_data_list=[padded_input],
                            batch_size=batch_size,
                            progress_update=(None if verbose==False else
                                             progress_update)))
        assert len(max_cross_corrs.shape)==2, max_cross_corrs.shape
        to_return[filter_idx:
                  (filter_idx+filter_batch_size),:] =\
                  np.transpose(max_cross_corrs)
        filter_idx += filter_batch_size
        
    return to_return

def get_full_cross_corr(filters, things_to_scan,
                           verbose=True, batch_size=10,
                           func_params_size=1000000,
                           progress_update=1000,
                           min_overlap=1,
                           mode='valid'):
    """
        func_params_size: when compiling functions
    """
    #reverse the patterns as the func is a conv not a cross corr
    filters = filters.astype("float32")[:,::-1,::-1]
    # padding_amount0 = int((filters[0].shape[-1])*(1-min_overlap))
    if mode == 'valid':
      num_xcor_sites = things_to_scan[0].shape[-1]-filters[0].shape[-1]+1
    elif mode == 'full':
      num_xcor_sites = things_to_scan[0].shape[-1]+filters[0].shape[-1]-1      
    to_return = np.zeros((filters.shape[0], len(things_to_scan), num_xcor_sites))
    #compile the number of filters that result in a function with
    #params equal to func_params_size 
    params_per_filter = np.prod(filters[0].shape)
    filter_batch_size = int(func_params_size/params_per_filter)
    filter_length = filters.shape[-1]
    filter_idx = 0 
    while filter_idx < filters.shape[0]:
        if (verbose):
            print("On filters",filter_idx,"to",(filter_idx+filter_batch_size))
        filter_batch = filters[filter_idx:(filter_idx+filter_batch_size)]
        cross_corr_func = compile_conv_func_with_theano(
                           set_of_2d_patterns_to_conv_with=filter_batch,
                           normalise_by_magnitude=False,
                           take_max=False,
                           mode=mode) 
        padding_amount = int((filter_length)*(1-min_overlap))
        padded_input = [np.pad(array=x,
                              pad_width=((padding_amount, padding_amount)),
                              mode="constant") for x in things_to_scan]
        all_cross_corrs = np.array(deeplift.util.run_function_in_batches(
                            func=cross_corr_func,
                            input_data_list=[padded_input],
                            batch_size=batch_size,
                            progress_update=(None if verbose==False else
                                             progress_update)))
        all_cross_corrs_max = all_cross_corrs.max(axis=2)
        assert len(all_cross_corrs_max.shape)==3, all_cross_corrs_max.shape
        to_return[filter_idx:
                  (filter_idx+filter_batch_size),:,:] =\
                  np.transpose(all_cross_corrs_max, axes=[1,0,2])
        filter_idx += filter_batch_size
        
    return to_return


def get_top_N_scores_per_region(scores, N, exclude_hits_within_window):
    scores = scores.copy()
    assert len(scores.shape)==2, scores.shape
    if (N==1):
        return np.max(scores, axis=1)[:,None]
    else:
        top_n_scores = []
        for i in range(scores.shape[0]):
            top_n_scores_for_region=[]
            for n in range(N):
                max_idx = np.argmax(scores[i]) 
                top_n_scores_for_region.append(scores[i][max_idx])
                scores[i][max_idx-exclude_hits_within_window:
                          max_idx+exclude_hits_within_window-1] = -np.inf
            top_n_scores.append(top_n_scores_for_region) 
        return np.array(top_n_scores)


def phenojaccard_sim_mat(sim_mat, k):
    from collections import defaultdict
    node_to_nearest = defaultdict(set)
    for node,neighbours_affs in enumerate(sim_mat):
        sorted_neighbours_affs = sorted(enumerate(neighbours_affs), key=lambda x: -x[1])
        node_to_nearest[node].update([x[0] for x in sorted_neighbours_affs[:k]])
    new_sim_mat = np.zeros_like(sim_mat)
    for node1 in node_to_nearest:
        for node2 in node_to_nearest:
            intersection = set(node_to_nearest[node1])
            intersection.intersection_update(node_to_nearest[node2])
            union = set(node_to_nearest[node1])
            union.update(node_to_nearest[node2])
            jaccard = float(len(intersection))/float(len(union))
            new_sim_mat[node1,node2] = jaccard
    return new_sim_mat
 

def jaccardify_sim_mat(sim_mat, verbose=True, power=1):
    print("Seriously consider using phenojaccard")
    if (verbose):
        print("calling jaccardify")
    sim_mat = np.power(sim_mat, power)
    import time
    t1 = time.time()
    minimum_sum = np.sum(np.minimum(sim_mat[:,None,:],
                         sim_mat[None,:,:]), axis=-1)
    maximum_sum = np.sum(np.maximum(sim_mat[:,None,:],
                         sim_mat[None,:,:]), axis=-1)
    ratio = minimum_sum/maximum_sum
    t2 = time.time()
    if (verbose):
        print("time taken in jaccardify",t2-t1)
    return ratio 


def compute_jaccardify(sim_mat, start_job, end_job):
    num_nodes = sim_mat.shape[0]
    distances = []
    for job_num in xrange(start_job, end_job):
        row_idx = int(job_num/num_nodes)
        col_idx = job_num%num_nodes
        minimum_sum = np.sum(np.minimum(sim_mat[row_idx,:],
                                        sim_mat[col_idx,:]))
        maximum_sum = np.sum(np.maximum(sim_mat[row_idx,:],
                                        sim_mat[col_idx,:]))
        ratio = minimum_sum/maximum_sum
        distances.append(ratio)
    return distances


#might be speed-upable further by recognizing that the distance is symmetric
def gpu_jaccardify(sim_mat, power=1,
                   func_params_size=1000000,
                   batch_size=100,
                   progress_update=1000,
                   verbose=True):
    if (power != 1):
        print("taking the power")
        sim_mat = np.power(sim_mat, power)
        print("took the power")
    num_nodes = sim_mat.shape[0]
    cols_batch_size = int(func_params_size/num_nodes) 
    print("cols_batch_size is",cols_batch_size)
    assert cols_batch_size > 0, "Please increase func_params_size; a single"+\
                                " col can't fit in the function otherwise"

    to_return = np.zeros(sim_mat.shape)

    col_idx = 0
    while col_idx < num_nodes:
        if (verbose):
            print("cols idx:",col_idx)
        #compile a function for computing distance to
        #nodes col_idx:(col_idx + cols_batch_size)

        #input var will store a batch of input data points
        input_var = theano.tensor.TensorType(
                        dtype=theano.config.floatX,
                        broadcastable=[False]*2)("input")
        
        end_col_idx = min(col_idx + cols_batch_size, num_nodes)
        minimum_sum = theano.tensor.sum(
                         theano.tensor.minimum(
                            input_var[:,None,:],
                            sim_mat[None,
                                     col_idx:end_col_idx,:]),
                                     axis=-1)
        maximum_sum = theano.tensor.sum(
                         theano.tensor.maximum(
                            input_var[:,None,:],
                            sim_mat[None,
                                     col_idx:end_col_idx,:]),
                                     axis=-1)
        ratios = minimum_sum/maximum_sum #the "jaccardified" distance

        #compile the function which takes input_var as the input tensor
        #and returns ratios 
        func = theano.function([input_var], ratios, allow_input_downcast=True)

        #apply the function in batches to all the nodes
        row_idx = 0
        while row_idx < num_nodes: 
            if (verbose):
                if (row_idx%progress_update == 0):
                    print("Done",row_idx)
            end_row_idx = row_idx+batch_size
            distances = func(sim_mat[row_idx:end_row_idx,:])
            to_return[row_idx:end_row_idx, col_idx:end_col_idx] = distances
            row_idx = end_row_idx
        col_idx = end_col_idx
    return to_return


#should be speed-upable further by recognizing that the distance is symmetric
def parallel_jaccardify(sim_mat, num_processes=4,
                        verbose=True, power=1,
                        temp_file_dir="tmp",
                        temp_file_prefix="jaccardify_h5"):

    if (os.path.isdir(temp_file_dir)==False):
        os.system("mkdir "+temp_file_dir)
    sim_mat = np.power(sim_mat, power)

    num_nodes = sim_mat.shape[0]
    total_tasks = num_nodes**2
    tasks_per_job = int(np.ceil(total_tasks/num_processes))

    launched_pids = []
    print(num_processes)
    for i in xrange(num_processes):
        pid = os.fork() 
        print(pid)
        if pid==0:
            try:
                #set a signal handler for interrupt signals
                signal.signal(signal.SIGINT,
                              (lambda signum, frame: os._exit(os.EX_TEMPFAIL)))
                start_job = tasks_per_job*i
                end_job = min(total_tasks, tasks_per_job*(i+1))
                distances = compute_jaccardify(sim_mat, start_job, end_job) 
                #write the distances to an h5 file
                h5_file_name = temp_file_dir+"/"\
                               +temp_file_prefix+"_"+str(i)+".h5"
                f = h5py.File(h5_file_name, "w")
                dset = f.create_dataset("/distances", data=distances)
                f.close()
                print("Exit!")
                os._exit(os.EX_OK) #exit the child
            except Exception, _:
                raise RuntimeError("Exception in job "+str(i)+\
                                   "\n"+traceback.format_exc()) 
                os._exit(os.EX_SOFTWARE)
        else:
            launched_pids.append(pid)

    try:
        while len(launched_pids) > 0:
            pid, return_code = os.wait()
            if return_code != os.EX_OK:  
                raise RuntimeError(
                "pid "+str(pid)+" gave error code "+str(return_code))
            if pid in launched_pids:
                launched_pids.remove(pid)

        #child processes would have all exited
        collated_distances = []
        #now collate all the stuff written to the various h5 files
        for i in xrange(num_processes):
            h5_file_name = temp_file_dir+"/"\
                           +temp_file_prefix+"_"+str(i)+".h5"
            f = h5py.File(h5_file_name)
            collated_distances.extend(f['/distances'])
            f.close()
            os.system("rm "+h5_file_name)
        assert len(collated_distances) == total_tasks 
        to_return = np.zeros((num_nodes, num_nodes))
        #now reshape the collated distances into a numpy array
        for i in xrange(len(collated_distances)):
            row_idx = int(i/num_nodes)
            col_idx = i%num_nodes
            to_return[row_idx, col_idx] = collated_distances[i]
        return to_return
    except KeyboardInterrupt, OSError:
        for pid in launched_pids:
            try:
                os.kill(pid, signal.SIGHUP)
            except:
                pass
        raise


def make_graph_from_sim_mat(sim_mat):
    import networkx as nx
    G = nx.Graph()
    print("Adding nodes")
    for i in range(len(sim_mat)):
        G.add_node(i)
    print("nodes added")
    edges_to_add = []
    print("Preparing edges")
    for i in range(len(sim_mat)):
        for j in range(i+1,len(sim_mat)):
            edges_to_add.append((i,j,{'weight':sim_mat[i,j]})) 
    print("Done preparing edges")
    G.add_edges_from(edges_to_add)
    print("Done adding edges")
    return G


def cluster_louvain(sim_mat):
    import community
    graph = make_graph_from_sim_mat(sim_mat)
    print("making partition")
    partition = community.best_partition(graph)
    print("done making partition")
    louvain_labels = [partition[i] for i in range(len(partition.keys()))]
    return louvain_labels


def scan_regions_with_filters(filters, regions_to_scan,
                              batch_size=50, progress_update=1000):
    """
        filters: for PWMs, use log-odds matrices.
        Will be cross-correlated with sequence
    
        set_of_regions: either one-hot-encoded or deeplift score tracks.
    """ 
    if (len(filters.shape)==3):
        filters = filters[:,None,:,:]
    assert filters.shape[1]==1 #input channels=1
    assert filters.shape[2]==4 #acgt
    assert len(regions_to_scan.shape)==4
    assert regions_to_scan.shape[1]==1 #input channels=1
    assert regions_to_scan.shape[2]==4 #acgt

    #set up the theano convolution 
    fwd_filters = filters[:,:,::-1,::-1] #convolutions reverse things
    rev_comp_filters = filters

    input_var = theano.tensor.TensorType(dtype=theano.config.floatX,
                                         broadcastable=[False]*4)("input") 
    fwd_filters_var = theano.tensor.as_tensor_variable(
                        x=fwd_filters, name="fwd_filters")
    rev_comp_filters_var = theano.tensor.as_tensor_variable(
                             x=rev_comp_filters, name="rev_comp_filters")
    fwd_conv_out = theano.tensor.nnet.conv2d(
                    input=input_var, filters=fwd_filters_var)
    rev_comp_conv_out = theano.tensor.nnet.conv2d(
                         input=input_var, filters=rev_comp_filters_var)

    #concatenate the results to take an elementwise max
    #remember that the lenght of the row dimension is 1, so this is a
    #good dimension to take advantage of for concatenation
    concatenated_fwd_and_rev_results = theano.tensor.concatenate(
        [fwd_conv_out, rev_comp_conv_out], axis=2) 

    fwd_or_rev = theano.tensor.argmax(concatenated_fwd_and_rev_results, axis=2, keepdims=True)
    scores = theano.tensor.max(concatenated_fwd_and_rev_results, axis=2, keepdims=True)

    #concatenated score and complementation
    concatenated_score_and_orientation = theano.tensor.concatenate(
        [scores,fwd_or_rev], axis=2)

    compiled_func = theano.function([input_var],
                                    concatenated_score_and_orientation,
                                    allow_input_downcast=True)

    #run function in batches
    conv_results = np.array(deeplift.util.run_function_in_batches(
                            func=compiled_func,
                            input_data_list=[regions_to_scan],
                            batch_size=batch_size,
                            progress_update=progress_update))

    return conv_results


def product_of_cosine_distances(filters, track1, track2,
                                batch_size=50, progress_update=1000):
    """
        filters: 3d array: filter_idx x ACGT x length
        Will take cosine distance of filter with track1 and
         as track2 at each position and return product of cosines

        track1: 3-d array: samples x length x channels(ACGT)
        track2: 3-d array: samples x length x channels(ACGT)
    """ 
    assert len(filters.shape)==3
    assert len(track1.shape)==3
    assert len(track2.shape)==3
    #for DNA sequences, filters.shape[1] will be 4 (ACGT)
    assert filters.shape[1] == track1.shape[2] #channel axis has same dims
    assert filters.shape[1] == track2.shape[2]

    #Normalize filters by magnitude (makes cosine dist computation easier)
    #first flatten the last two dimensions of the filters, then compute norm
    filter_magnitudes = np.linalg.norm(filters.reshape(filters.shape[0],-1),
                                       axis=1)
    #'None' inserts dummy dimensions of length 1
    filters = filters/filter_magnitudes[:,None,None]
    scanning_window_area = filters.shape[1]*filters.shape[2]

    #insert dummy dimensions to convert shapes from 1d format to
    #the theano 2d format
    #for input tracks:
    #keras 1d format is: samples x length x ACGT (channels)
    #theano conv2d wants: samples x 1 (channels) x ACGT (height) x length
    #for filters:
    #start with: num_filters x ACGT (channels) x filter len
    #conv2d wants: num_filters x 1 (channels) x ACGT (height) x filter len
    #transpose is necessary to get the ACGT into axis index 2
    filters = filters[:,None,:,:]
    track1 = track1[:,None,:,:].transpose(0,1,3,2)
    track2 = track2[:,None,:,:].transpose(0,1,3,2)

    #prepare the forward and reverse complement filters for the conv2d call
    fwd_filters = filters[:,:,::-1,::-1] #convolutions reverse things
    rev_comp_filters = filters

    #create theano variables for the inputs and filters
    #set the channel axis to be broadcastable because that should come
    #in handy when dividing the convolutions for each filter
    #by the magnitude of the underlying track computed using average pooling
    track1_var = theano.tensor.TensorType(
                    dtype=theano.config.floatX,
                    broadcastable=[False,True,False,False])("track1") 
    track2_var = theano.tensor.TensorType(
                  dtype=theano.config.floatX,
                  broadcastable=[False,True,False,False])("track2") 
    fwd_filters_var = theano.tensor.as_tensor_variable(
                        x=fwd_filters, name="fwd_filters")
    rev_comp_filters_var = theano.tensor.as_tensor_variable(
                             x=rev_comp_filters, name="rev_comp_filters")

    #get variables representing the results of the convolutions,
    #which are equal to the the dot products at each sliding window
    fwd_track1_conv_out_var = theano.tensor.nnet.conv2d(
                         input=track1_var, filters=fwd_filters_var)
    fwd_track2_conv_out_var = theano.tensor.nnet.conv2d(
                         input=track2_var, filters=fwd_filters_var)
    rev_track1_conv_out_var = theano.tensor.nnet.conv2d(
                         input=track1_var, filters=rev_comp_filters_var)
    rev_track2_conv_out_var = theano.tensor.nnet.conv2d(
                         input=track2_var, filters=rev_comp_filters_var)

    #cosine distance is dot product divided by magnitude; we compute
    #per-window magnitude by squaring the tracks, then summing in
    #sliding windows, then taking the square root
    #squaring:
    track1_squared_var = track1_var*track1_var 
    track2_squared_var = track2_var*track2_var
    #compute per-position sliding window sums using average and then
    #scaling up by window size
    track1_squared_sumpool_var = theano.tensor.signal.pool.pool_2d(
                                input=track1_squared_var,
                                ws=(filters.shape[-2], filters.shape[-1]),
                                ignore_border=False,
                                stride=(1,1),
                                pad=(0,0),
                                mode='average_exc_pad')*scanning_window_area
    track2_squared_sumpool_var = theano.tensor.signal.pool.pool_2d(
                                input=track2_squared_var,
                                ws=(filters.shape[-2], filters.shape[-1]),
                                ignore_border=False,
                                stride=(1,1),
                                pad=(0,0),
                                mode='average_exc_pad')*scanning_window_area

    #take square roots to get magnitudes
    track1_magnitude_var = theano.tensor.sqrt(track1_squared_sumpool_var)
    track2_magnitude_var = theano.tensor.sqrt(track2_squared_sumpool_var)

    pseudocount=0.0000001
    #compute product of cosine distances. Add pseudocount to avoid div by 0
    #filters were already normalized to have magnituded 1, so don't have
    #to worry about them
    fwd_scores_var = ((fwd_track1_conv_out_var/
                    (track1_magnitude_var+pseudocount))
                  *(fwd_track2_conv_out_var/
                    (track2_magnitude_var+pseudocount)))
    rev_scores_var = ((rev_track1_conv_out_var/
                    (track1_magnitude_var+pseudocount))
                  *(rev_track2_conv_out_var/
                    (track2_magnitude_var+pseudocount))) 

    #concatenate the results to take an elementwise max
    #The length of the height dimension becomes 1 after convolution,
    #so this is a good dimension to take advantage of for concatenation
    concatenated_fwd_and_rev_scores_var = theano.tensor.concatenate(
        [fwd_scores_var, rev_scores_var], axis=2) 
    #use argmax to determine whether the fwd or rev match is stronger
    fwd_or_rev_var = theano.tensor.argmax(concatenated_fwd_and_rev_scores_var,
                                          axis=2, keepdims=True)
    #final score is max of fwd and rev result
    final_scores_var = theano.tensor.max(concatenated_fwd_and_rev_scores_var,
                                         axis=2, keepdims=True)
    #concatenate the score with variable determining fwd or rev
    concatenated_score_and_orientation_var = theano.tensor.concatenate(
        [final_scores_var,fwd_or_rev_var], axis=2)

    #compile the function linking inputs and outputs
    compiled_func = theano.function([track1_var, track2_var],
                                    concatenated_score_and_orientation_var,
                                    allow_input_downcast=True)

    #run function in batches
    final_scores = np.array(deeplift.util.run_function_in_batches(
                            func=compiled_func,
                            input_data_list=[track1, track2],
                            batch_size=batch_size,
                            progress_update=progress_update))

    return final_scores


def project_onto_nonzero_filters(filters, track,
                                 batch_size=50, progress_update=1000):
    """
        filters: 3d array: filter_idx x ACGT x length
        At each position, gives you a value equivalent to what you get if
         you eliminate all positions of the filter that are 0 in the
         underlying track, and then projected the deeplift track onto
         the filter

        track: 3-d array: samples x length x channels(ACGT)
    """ 
    assert len(filters.shape)==3
    assert len(track.shape)==3
    #for DNA sequences, filters.shape[1] will be 4 (ACGT)
    assert filters.shape[1] == track.shape[2] #channel axis has same dims
    scanning_window_area = filters.shape[1]*filters.shape[2]

    #insert dummy dimensions to convert shapes from 1d format to
    #the theano 2d format
    #for input tracks:
    #keras 1d format is: samples x length x ACGT (channels)
    #theano conv2d wants: samples x 1 (channels) x ACGT (height) x length
    #for filters:
    #start with: num_filters x ACGT (channels) x filter len
    #conv2d wants: num_filters x 1 (channels) x ACGT (height) x filter len
    #transpose is necessary to get the ACGT into axis index 2
    filters = filters[:,None,:,:]
    track = track[:,None,:,:].transpose(0,1,3,2)

    #prepare the forward and reverse complement filters for the conv2d call
    fwd_filters = filters[:,:,::-1,::-1] #convolutions reverse things
    rev_comp_filters = filters

    #create theano variables for the inputs and filters
    #set the channel axis to be broadcastable because that should come
    #in handy when dividing the convolutions for each filter
    #by the magnitude of the underlying track computed using average pooling
    track_var = theano.tensor.TensorType(
                    dtype=theano.config.floatX,
                    broadcastable=[False,True,False,False])("track") 
    nonzeros_track_var = theano.tensor.gt(
                          theano.tensor.abs_(track_var),0.0)*1.0

    fwd_filters_var = theano.tensor.as_tensor_variable(
                        x=fwd_filters, name="fwd_filters")
    squared_fwd_filters_var = fwd_filters_var*fwd_filters_var
    nonzero_fwd_filters_var = theano.tensor.gt(
                               theano.tensor.abs_(fwd_filters_var),0.0)*1.0
    rev_comp_filters_var = theano.tensor.as_tensor_variable(
                             x=rev_comp_filters, name="rev_comp_filters")
    squared_rev_comp_filters_var = rev_comp_filters_var*rev_comp_filters_var
    nonzero_rev_filters_var = theano.tensor.gt(
                              theano.tensor.abs_(rev_comp_filters_var),0.0)*1.0

    #get variables representing the results of the convolutions,
    #which are equal to the the dot products at each sliding window
    fwd_track_conv_out_var = theano.tensor.nnet.conv2d(
                         input=track_var, filters=fwd_filters_var)
    rev_track_conv_out_var = theano.tensor.nnet.conv2d(
                         input=track_var, filters=rev_comp_filters_var)
    #also convolve the squared filters with the nonzeros track, to get
    #the square of the magnitude for all the positions of the filter that
    #are nonzero in the underlying track
    squaredfwd_nonzerotrack_conv_out_var = theano.tensor.nnet.conv2d(
                               input=nonzeros_track_var,
                               filters=squared_fwd_filters_var)
    squaredrev_nonzerotrack_conv_out_var = theano.tensor.nnet.conv2d(
                               input=nonzeros_track_var,
                               filters=squared_rev_comp_filters_var)

    #take square roots to get magnitudes
    fwd_nonzero_magnitude_var =\
     theano.tensor.sqrt(squaredfwd_nonzerotrack_conv_out_var)
    rev_nonzero_magnitude_var =\
     theano.tensor.sqrt(squaredrev_nonzerotrack_conv_out_var)

    pseudocount=0.0000001
    #compute projection. Add pseudocount to avoid div by 0
    fwd_scores_var = (fwd_track_conv_out_var/
                      (fwd_nonzero_magnitude_var+pseudocount))
    rev_scores_var = (rev_track_conv_out_var/
                      (rev_nonzero_magnitude_var+pseudocount))

    #concatenate the results to take an elementwise max
    #The length of the height dimension becomes 1 after convolution,
    #so this is a good dimension to take advantage of for concatenation
    concatenated_fwd_and_rev_scores_var = theano.tensor.concatenate(
        [fwd_scores_var, rev_scores_var], axis=2) 
    #use argmax to determine whether the fwd or rev match is stronger
    fwd_or_rev_var = theano.tensor.argmax(concatenated_fwd_and_rev_scores_var,
                                          axis=2, keepdims=True)
    #final score is max of fwd and rev result
    final_scores_var = theano.tensor.max(concatenated_fwd_and_rev_scores_var,
                                         axis=2, keepdims=True)
    #concatenate the score with variable determining fwd or rev
    concatenated_score_and_orientation_var = theano.tensor.concatenate(
        [final_scores_var,fwd_or_rev_var], axis=2)

    #compile the function linking inputs and outputs
    compiled_func = theano.function([track_var],
                                     concatenated_score_and_orientation_var,
                                     allow_input_downcast=True)

    #run function in batches
    final_scores = np.array(deeplift.util.run_function_in_batches(
                            func=compiled_func,
                            input_data_list=[track],
                            batch_size=batch_size,
                            progress_update=progress_update))

    return final_scores


def get_betas_from_tsne_conditional_probs(conditional_probs,
                                          original_affmat, aff_to_dist_mat):
    dist_mat = aff_to_dist_mat(original_affmat)
    betas = []
    for i,(prob_row, distances, affinities) in\
        enumerate(zip(conditional_probs,
                      dist_mat, original_affmat)):
        nonzero_probs = prob_row[prob_row > 0.0]
        nonzero_distances = distances[prob_row > 0.0]
        prob1, dist1 = max(zip(nonzero_probs, nonzero_distances),
                           key=lambda x: x[1])
        prob2, dist2 = min(zip(nonzero_probs, nonzero_distances),
                           key=lambda x: x[1])
        beta = np.log(prob2/prob1)/(dist1-dist2)
        betas.append(beta)
        #sanity check
        recomputed_probs = np.exp(-beta*(distances))*(affinities > 0.0)
        recomputed_probs[i] = 0
        recomputed_probs = recomputed_probs/np.sum(recomputed_probs)
        test_recomputed_probs = recomputed_probs[prob_row > 0.0]/\
                                 np.sum(recomputed_probs[prob_row > 0.0])
        maxdiff = np.max(np.abs(prob_row[prob_row > 0.0]
                                - test_recomputed_probs))
        assert maxdiff < 10**-5,\
               (np.sum(prob_row), maxdiff, test_recomputed_probs)
    return np.array(betas)


def convert_to_percentiles(vals):
    to_return = np.zeros(len(vals))
    sorted_vals = sorted(enumerate(vals), key=lambda x: x[1])
    for sort_idx,(orig_idx,val) in enumerate(sorted_vals):
        to_return[orig_idx] = sort_idx/float(len(vals))
    return to_return


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
        sum_ps = np.sum(ps)
        ps = ps/(max(sum_ps,EPSILON_DBL))
        sum_disti_Pi = np.sum(distances*ps)
        entropy = np.log(sum_ps) + beta * sum_disti_Pi
        
        entropy_diff = entropy - desired_entropy
        #print(beta, np.exp(entropy), entropy_diff)
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
    return beta, ps
