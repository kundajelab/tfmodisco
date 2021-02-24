from __future__ import division, print_function
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf
import numpy as np
import sys


_SESS = None

def get_session():
    try:
        #use the keras session if there is one
        import keras.backend as K
        return K.get_session()
    except:
        #Warning: I haven't really tested this behaviour out...
        global _SESS 
        if _SESS is None:
            print("MAKING A SESSION")
            _SESS = tf.Session()
            _SESS.run(tf.global_variables_initializer()) 
        return _SESS


def compile_func(inputs, outputs):
    if (isinstance(inputs, list)==False):
        print("Wrapping the inputs in a list...")
        inputs = [inputs]
    assert isinstance(inputs, list)
    def func_to_return(inp):
        if len(inp) > len(inputs) and len(inputs)==1:
            print("Wrapping the inputs in a list...")
            inp = [inp]
        assert len(inp)==len(inputs),\
            ("length of provided list should be "
             +str(len(inputs))+" for tensors "+str(inputs)
             +" but got input of length "+str(len(inp)))
        feed_dict = {}
        for input_tensor, input_val in zip(inputs, inp):
            feed_dict[input_tensor] = input_val 
        sess = get_session()
        return sess.run(outputs, feed_dict=feed_dict)  
    return func_to_return


def run_function_in_batches(func,
                            input_data_list,
                            learning_phase=None,
                            batch_size=10,
                            progress_update=1000,
                            multimodal_output=False):
    #func has a return value such that the first index is the
    #batch. This function will run func in batches on the inputData
    #and will extend the result into one big list.
    #if multimodal_output=True, func has a return value such that first
    #index is the mode and second index is the batch
    assert isinstance(input_data_list, list), "input_data_list must be a list"
    #input_datas is an array of the different input_data modes.
    to_return = [];
    i = 0;
    while i < len(input_data_list[0]):
        if (progress_update is not None):
            if (i%progress_update == 0):
                print("Done",i)
        func_output = func(([x[i:i+batch_size] for x in input_data_list]
                                +([] if learning_phase is
                                   None else [learning_phase])
                        ))
        if (multimodal_output):
            assert isinstance(func_output, list),\
             "multimodal_output=True yet function return value is not a list"
            if (len(to_return)==0):
                to_return = [[] for x in func_output]
            for to_extend, batch_results in zip(to_return, func_output):
                to_extend.extend(batch_results)
        else:
            to_return.extend(func_output)
        i += batch_size
    return to_return


def get_gapped_kmer_embedding_func(filters, biases, require_onehot_match):

    #filters should be: out_channels, rows, ACGT
    filters = filters.astype("float32").transpose((1,2,0))
    biases = biases.astype("float32")
    if (require_onehot_match):
        onehot_var = tf.placeholder(dtype=tf.float32,
                                    shape=(None,None,None),
                                    name="onehot")
    toembed_var = tf.placeholder(dtype=tf.float32,
                                 shape=(None,None,None),
                                 name="toembed")
    tf_filters = tf.convert_to_tensor(value=filters, name="filters")
    if (require_onehot_match):
        onehot_out = 1.0*(tf.cast(tf.greater(tf.nn.conv1d(
                        value=onehot_var,
                        filters=tf_filters,
                        stride=1,
                        padding='VALID') + biases[None,None,:], 0.0),
                        tf.float32))
    embedding_out = tf.reduce_sum(
                        input_tensor=(
                            tf.nn.conv1d(
                                value=toembed_var,
                                filters=tf_filters,
                                stride=1,
                                padding='VALID'))*
                                (onehot_out if require_onehot_match else 1.0),
                        axis=1)
    if (require_onehot_match):
        func = compile_func(inputs=[onehot_var, toembed_var],
                            outputs=embedding_out)
        def batchwise_func(onehot, to_embed, batch_size, progress_update):
            return np.array(run_function_in_batches(
                                func=func,
                                input_data_list=[onehot, to_embed],
                                batch_size=batch_size,
                                progress_update=progress_update))
    else:
        func = compile_func(inputs=[toembed_var],
                            outputs=embedding_out)
        def batchwise_func(to_embed, batch_size, progress_update):
            return np.array(run_function_in_batches(
                                func=func,
                                input_data_list=[to_embed],
                                batch_size=batch_size,
                                progress_update=progress_update))
    return batchwise_func


def max_cross_corrs(filters, things_to_scan, min_overlap,
                       batch_size=50,
                       func_params_size=1000000,
                       progress_update=1000):
    """
        func_params_size: when compiling functions
    """
    #reverse the patterns as the func is a conv not a cross corr
    assert len(filters.shape)==3,"Did you pass in filters of unequal len?"
    assert filters.shape[-1]==things_to_scan.shape[-1]
    filters = filters.astype("float32")[:,:,:]
    to_return = np.zeros((filters.shape[0], len(things_to_scan)))
    #compile the number of filters that result in a function with
    #params equal to func_params_size 
    params_per_filter = np.prod(filters[0].shape)
    filter_batch_size = int(func_params_size/params_per_filter)
    filter_length = filters.shape[1]
    filter_idx = 0 
    while filter_idx < filters.shape[0]:
        if (progress_update is not None):
            print("On filters",filter_idx,"to",
                  min((filter_idx+filter_batch_size),len(filters)))
            sys.stdout.flush()

        filter_batch = filters[filter_idx:
                              min((filter_idx+filter_batch_size),len(filters))]

        padding_amount = int((filter_length)*(1-min_overlap))
        padded_input = [np.pad(array=x,
                              pad_width=((padding_amount, padding_amount),
                                         (0,0)),
                              mode="constant") for x in things_to_scan]

        onehot_var = tf.placeholder(dtype=tf.float32,
                                    shape=(None,None,None),
                                    name="onehot")
        input_var = tf.placeholder(dtype=tf.float32,
                                   shape=(None,None,None),
                                   name="input")
        tf_filters = tf.convert_to_tensor(value=filter_batch,
                                          name="filters")
        conv_out = tf.nn.conv1d(
                    value=input_var[:,None,:,:],
                    filters=tf_filters[:,None,:,:],
                    stride=1,
                    padding='VALID')[:,:,:,0]

        max_out = tf.reduce_max(input_tensor=conv_out, axis=-1)

        max_cross_corr_func = compile_func(inputs=[input_var],
                                           outputs=max_out)

        max_cross_corrs = np.array(run_function_in_batches(
                            func=max_cross_corr_func,
                            input_data_list=[padded_input],
                            batch_size=batch_size,
                            progress_update=progress_update))
        assert len(max_cross_corrs.shape)==2, max_cross_corrs.shape
        to_return[filter_idx:
                  min((filter_idx+filter_batch_size),len(filters)),:] =\
                  np.transpose(max_cross_corrs)
        filter_idx += filter_batch_size
        
    return to_return
