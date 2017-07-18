import theano
from theano import tensor as T


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
        func_output = func(*([x[i:i+batch_size] for x in input_data_list]
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
        i += batch_size;
    return to_return


def tensor_with_dims(num_dims, name):
    return T.TensorType(dtype=theano.config.floatX,
                        broadcastable=[False]*num_dims)(name)


def get_window_sum_function(window_size, same_size_return):
    """
        Returns a function for smoothening inputs with a window
         of size window_size.

        Returned function has arguments of inp,
         batch_size and progress_update
    """
    inp_tensor = tensor_with_dims(2, "inp_tensor") 
    inp_tensor = inp_tensor[:,None,None,:]

    if (same_size_return):
        border_mode='same'
    else:
        border_mode='valid'

    averaged_inp = theano.pool2d(
                        inp=inp_tensor,
                        pool_size=(1,window_size),
                        strides=(1,1),
                        border_mode=border_mode,
                        ignore_border=True,
                        pool_mode='avg_exc_pad') 

    #if window_size is even, then we have an extra value in the output,
    #so kick off the value from the front
    if (window_size%2==0 and same_size_return):
        averaged_inp = averaged_inp[:,:,:,1:]

    averaged_inp = averaged_inp[:,0,0,:]
    smoothen_func = theano.function([inp_tensor], averaged_inp*window_size)

    def smoothen(inp, batch_size, progress_update=None):
       return run_function_in_batches(
                func=smoothen_func,
                input_data_list=[inp],
                batch_size=batch_size,
                progress_update=progress_update)

    return smoothen


def get_argmax_function(): 
    inp_tensor = tensor_with_dims(2, "inp_tensor") 
    argmaxes = T.argmax(inp_tensor, axis=1) 
    argmax_func = theano.function([inp_tensor], argmaxes)
    def argmax_func(inp, batch_size, progress_update=None):
        return run_function_in_batches(
                func=argmax_func,
                input_data_list=[inp],
                batch_size=batch_size,
                progress_update=progress_update)
    return argmax_func


