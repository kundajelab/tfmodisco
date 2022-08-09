from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.stats
from modisco import util

class TransformCentralWindowValueProvider():

    def __init__(self, track_name, central_window, val_transformer):
        if isinstance(track_name, str):
            self.track_name = track_name
        else: 
            self.track_name = track_name.decode('utf-8')
        self.central_window = central_window
        self.val_transformer = val_transformer

    def __call__(self, seqlet):
        val = self.get_val(seqlet=seqlet)
        return self.val_transformer(val=val)

    def get_imp_around_central_window(self, seqlet, central_window):
        flank_to_ignore = int(0.5*(len(seqlet)-central_window))
        track_values = seqlet[self.track_name]\
                        .fwd[flank_to_ignore:(len(seqlet)-flank_to_ignore)]
        return np.sum(track_values)

    def get_val(self, seqlet):
        if (hasattr(self.central_window, '__iter__')):
            vals = []
            for window_width in self.central_window:
                imp = self.get_imp_around_central_window(
                        seqlet=seqlet, central_window=window_width) 
                vals.append(imp)
            return vals
        else:
            return self.get_imp_around_central_window(seqlet=seqlet,
                            central_window=self.central_window)

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.attrs["track_name"] = self.track_name
        if (hasattr(self.central_window, '__iter__')):
            grp.create_dataset("central_window",
                               data=np.array(self.central_window))
        else:
            grp.attrs["central_window"] = self.central_window
        self.val_transformer.save_hdf5(grp.create_group("val_transformer")) 




def valatmaxabs(arrs):
    idxs = np.argmax(np.abs(arrs), axis=0)
    return arrs[idxs, np.arange(len(arrs[0]))], idxs
    

class PrecisionValTransformer():
    def __init__(self, sliding_window_sizes, pos_irs, neg_irs):
        assert len(pos_irs)==len(neg_irs)
        self.sliding_window_sizes = sliding_window_sizes
        self.pos_irs = pos_irs
        self.neg_irs = neg_irs

    #I have the transform_score_track function in addition to the __call__
    # function because in the case of the transform_score_track function, the
    # total importance for a given window size doesn't have to be retained
    # until the very last step; it is computed and immediately subject to
    # transformation.
    def transform_score_track(self, score_track): 
        from .coordproducers import get_simple_window_sum_function
        percentile_transformed_tracks = []
        for sliding_window_size, pos_ir, neg_ir in zip(
                        self.sliding_window_sizes, self.pos_irs, self.neg_irs):
            window_sum_function = get_simple_window_sum_function(
                                        sliding_window_size)
            window_sums_rows = window_sum_function(arrs=score_track)
            transformed_track = []
            for row_idx, window_sums_row in enumerate(window_sums_rows): 
                transformed_row = np.zeros_like(window_sums_row)

                pos_val_indices = np.nonzero(window_sums_row >= 0)[0] 
                pos_vals = window_sums_row[pos_val_indices]
                transformed_pos_vals = pos_ir.transform(pos_vals)
                transformed_row[pos_val_indices] = transformed_pos_vals

                neg_val_indices = np.nonzero(window_sums_row < 0)[0]
                if (len(neg_val_indices) > 0 and neg_ir is not None):
                    neg_vals = window_sums_row[neg_val_indices]
                    transformed_neg_vals = neg_ir.transform(neg_vals)
                    transformed_row[neg_val_indices] = -transformed_neg_vals

                #add padding to make up for entries lost due to the sliding
                # windows
                transformed_row = np.pad(transformed_row,
                    pad_width=(
                        (int((sliding_window_size-1)/2.0),
                         (sliding_window_size-1)
                          -int((sliding_window_size-1)/2.0))),
                    mode='constant')
                assert len(transformed_row)==len(score_track[row_idx]),\
                    (len(transformed_row), len(score_track[row_idx]))
                transformed_track.append(transformed_row) 
            percentile_transformed_tracks.append(transformed_track)
        #ultimately, return the result of taking the value that has
        # the maximum absolute value over all the different transformed tracks
        bestwindowvals = [valatmaxabs(
                          np.array([percentile_transformed_tracks[i][j]
                          for i in range(len(percentile_transformed_tracks))]))
                        for j in range(len(score_track))]
        #return both the best window values AND the idx of the window size,
        # as I think the latter is also helpful to know
        return [x[0] for x in bestwindowvals], [x[1] for x in bestwindowvals] 

    #In the case of __call__, val is a list of the total importance for
    # different window sizes
    def __call__(self, val): 
        assert len(val)==len(self.pos_irs)
        transformed_vals = []
        for (a_val, pos_ir, neg_ir) in zip(val, self.pos_irs, self.neg_irs):
            if (a_val >= 0):
                transformed_val = pos_ir.transform([a_val])[0]
            else:
                transformed_val = -neg_ir.transform([a_val])[0] 
            transformed_vals.append(transformed_val)
        return transformed_vals[np.argmax(np.abs(transformed_vals))] 

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.create_dataset("sliding_window_sizes",
                           data=np.array(self.sliding_window_sizes))
        util.save_list_of_objects(grp=grp.create_group("pos_irs"),
                                  list_of_objects=self.pos_irs)
        util.save_list_of_objects(grp=grp.create_group("neg_irs"),
                                  list_of_objects=self.neg_irs)



class AbsPercentileValTransformer():
    def __init__(self, distribution):
        self.distribution = np.array(sorted(np.abs(distribution)))

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.create_dataset("distribution", data=self.distribution)

    def __call__(self, val):
        return np.sign(val)*np.searchsorted(
                 a=self.distribution,
                 v=abs(val))/float(len(self.distribution))

