from __future__ import division, print_function
from collections import namedtuple
import numpy as np
from joblib import Parallel, delayed


#Seqlet data for imputation
_SeqlDatForImput = namedtuple("_SeqlDatForImput",
                              ["corelen", "flanklen", "onehot", "hyp"])


#fake constructor for tuple
def SeqlDatForImput(corelen, onehot, hyp):
    assert (len(onehot)-corelen)%2 == 0
    assert onehot.shape==hyp.shape
    flanklen = int((len(onehot)-corelen)/2)
    return _SeqlDatForImput(corelen=corelen, flanklen=flanklen,
                            onehot=onehot, hyp=hyp)


def compute_sim_on_pairs(oneseql_corelen, oneseql_onehot, oneseql_hyp,
                         seqlset_corelen, seqlset_onehot, seqlset_hyp,
                         min_overlap_frac, pair_sim_metric):

    assert oneseql_onehot.shape==oneseql_hyp.shape
    assert len(oneseql_onehot.shape)==2
    assert seqlset_onehot.shape==seqlset_hyp.shape
    assert len(seqlset_onehot.shape)==3

    assert (oneseql_onehot.shape[0]-oneseql_corelen)%2==0
    assert (seqlset_onehot.shape[1]-seqlset_corelen)%2==0
    oneseql_flanklen = int((oneseql_onehot.shape[0]-oneseql_corelen)/2)
    seqlset_flanklen = int((seqlset_onehot.shape[1]-oneseql_corelen)/2)

    min_overlap = int(np.ceil(min(oneseql_corelen, seqlset_corelen)
                              *min_overlap_frac))

    oneseql_actual = oneseql_onehot*oneseql_hyp
    seqlset_actual = seqlset_onehot*seqlset_hyp
    
    #iterate over all possible offsets of oneseql relative to seqlset
    startoffset = -(oneseql_corelen-min_overlap)
    endoffset = (seqlset_corelen-min_overlap)
    possible_offsets = np.array(range(startoffset, endoffset+1))
    #init the array that will store the similarity results
    sim_results = np.zeros((seqlset_onehot.shape[0], len(possible_offsets)))
    for offsetidx,offset in enumerate(possible_offsets):
        #compute the padding needed for the offset seqlets to be comparable
        oneseql_leftpad = max(offset, 0)  
        oneseql_rightpad = max(seqlset_corelen-(oneseql_corelen+offset),0) 
        #based on the padding, figure out how we would need to slice into
        # the available numpy arrays
        oneseql_slicestart = oneseql_flanklen-oneseql_leftpad
        oneseql_sliceend = oneseql_flanklen+oneseql_corelen+oneseql_rightpad

        #do the same for seqlset
        seqlset_leftpad = max(-offset, 0) 
        seqlset_rightpad = max((oneseql_corelen+offset)-seqlset_corelen, 0)
        seqlset_slicestart = seqlset_flanklen-seqlset_leftpad
        seqlset_slicesend = seqlset_flanklen+seqlset_corelen+seqlset_rightpad

        #slice to get the underlying data
        oneseqlactual_slice = (oneseql_actual[oneseql_slicestart:
                                              oneseql_sliceend])[None,:,:]
        oneseqlonehot_slice = oneseql_onehot[oneseql_slicestart:
                                             oneseql_sliceend] 
        oneseqlhyp_slice = oneseql_hyp[oneseql_slicestart:oneseql_sliceend]

        seqlsetactual_slice = seqlset_actual[:,seqlset_slicestart:
                                               seqlset_slicesend]
        seqlsetonehot_slice = seqlset_onehot[:,seqlset_slicestart:
                                               seqlset_slicesend]
        seqlsethyp_slice = seqlset_hyp[:,seqlset_slicestart:seqlset_slicesend]

        oneseql_imputed = oneseqlhyp_slice[None,:,:]*seqlsetonehot_slice
        seqlset_imputed = seqlsethyp_slice*oneseqlonehot_slice[None,:,:]
       
        sim_results[:,offsetidx] = (
            0.5*pair_sim_metric(oneseqlactual_slice, seqlset_imputed)
          + 0.5*pair_sim_metric(oneseql_imputed, seqlsetactual_slice)) 
    argmax = np.argmax(sim_results, axis=-1)
    return sim_results[np.arange(len(argmax)),argmax], possible_offsets[argmax]


class SequenceAffmatComputer_Impute(object):

    def __init__(self, metric, n_jobs, min_overlap_frac):
        self.min_overlap_frac = min_overlap_frac 
        self.pair_sim_metric = metric
        self.n_jobs = n_jobs

    def __call__(self, seqlets, onehot_trackname, hyp_trackname):

        hasrev = seqlets[0][onehot_trackname].hasrev

        seqlet_corelengths = [len(x) for x in seqlets]
        #for now, will just deal with case where all seqlets are of equal len
        assert len(set(seqlet_corelengths))==1; the_corelen=seqlet_corelengths[0]

        #max_seqlet_len will return the length of the longest seqlet core
        max_seqlet_len = max(seqlet_corelengths)
        #for each seqlet, figure out the maximum size of the flank needed
        # on each size. This is determined by the length of the longest
        # seqlet that each seqlet could be compared against
        flank_sizes = [max_seqlet_len-int(corelen*self.min_overlap_frac)
                       for corelen in seqlet_corelengths] 

        allfwd_onehot = (np.array( #I do the >0 at end to binarize
                            [seqlet[onehot_trackname].get_core_with_flank(
                             left=flank, right=flank, is_revcomp=False)
                             for seqlet,flank in zip(seqlets,flank_sizes)])>0) 
        allfwd_hyp = np.array(
                            [seqlet[hyp_trackname].get_core_with_flank(
                             left=flank, right=flank, is_revcomp=False)
                             for seqlet,flank in zip(seqlets,flank_sizes)]) 
        if (hasrev):
            allrev_onehot = allfwd_onehot[:,::-1,::-1]
            allrev_hyp = allfwd_hyp[:,::-1,::-1]

        assert allfwd_onehot.shape==allfwd_hyp.shape
        assert all([len(single_onehot)==(len(seqlet)+2*flanksize)
                    for (seqlet, single_onehot, flanksize) in
                    zip(seqlets, allfwd_onehot, flank_sizes)])

        indices = [(i,j) for i in range(len(seqlets))
                         for j in range(len(seqlets))]
        fwdresults = Parallel(n_jobs=self.n_jobs, verbose=True)(
                                delayed(compute_sim_on_pairs)(
                                    oneseql_corelen=the_corelen,
                                    oneseql_onehot=allfwd_onehot[i],
                                    oneseql_hyp=allfwd_hyp[i],
                                    seqlset_corelen=the_corelen,
                                    seqlset_onehot=allfwd_onehot,
                                    seqlset_hyp=allfwd_hyp,
                                    min_overlap_frac=self.min_overlap_frac,
                                    pair_sim_metric=self.pair_sim_metric)
                                for i in range(len(seqlets)))
        affmat = np.array([x[0] for x in fwdresults])
        assert np.max(np.abs(affmat.T-affmat)==0)
        offsets = np.array([x[1] for x in fwdresults])
        del fwdresults
        import gc
        gc.collect()

        if (hasrev):
            revresults = Parallel(n_jobs=self.n_jobs, verbose=True)(
                                delayed(compute_sim_on_pairs)(
                                    oneseql_corelen=the_corelen,
                                    oneseql_onehot=allfwd_onehot[i],
                                    oneseql_hyp=allfwd_hyp[i],
                                    seqlset_corelen=the_corelen,
                                    seqlset_onehot=allrev_onehot,
                                    seqlset_hyp=allrev_hyp,
                                    min_overlap_frac=self.min_overlap_frac,
                                    pair_sim_metric=self.pair_sim_metric)
                                for i in range(len(seqlets)))
            revaffmat = np.array([x[0] for x in revresults])
            revoffsets = np.array([x[1] for x in revresults])
            assert np.max(np.abs(revaffmat.T-revaffmat)==0)
            isfwdmat = affmat > revaffmat
            affmat = isfwdmat*affmat + (isfwdmat==False)*revaffmat
            offsets = isfwdmat*offsets + (isfwdmat==False)*revoffsets
            del revresults
            import gc
            gc.collect()

        return affmat, offsets, isfwdmat

