import deeplift
import numpy as np
import deeplift.backend as B

def create_detector_from_subset_of_sequential_layers(sequential_container,
                                                    idx_of_layer_of_interest,
                                                    channel_indices,
                                                    multipliers_on_channels):
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
