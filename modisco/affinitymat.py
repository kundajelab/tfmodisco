from __future__ import division, print_function
from collections import namedtuple


#Seqlet data for imputation
_SeqlDatForImput = namedtuple("_SeqlDatForImput",
                              [corelen, flanklen, onehot, hyp])

#fake constructor for tuple
def SeqlDatForImput(corelen, onehot, hyp):
    assert (len(onehot)-corelen)%2 == 0
    assert onehot.shape==hyp.shape
    flanklen = int((len(onehot)-corelen)/2)
    return _SeqlDatForImput(corelen=corelen, flanklen=flanklen,
                            onehot=onehot, hyp=hyp) 


class ImputedSimComputer(object):

    def __init__(self, sim_metric):
        self.sim_metric = sim_metric

    def __call__(self, shortr_seqldat, longer_seqldat, min_overlap_frac):
        assert (shortr_seqldat.coreseqletlen <= longer_seqldat.coreseqletlen) 
        min_overlap = int(shortr_seqldat.coreseqletlen*min_overlap_frac) 
        flank_shortr = (longer_seqldat.coreseqletlen - min_overlap) 
        flank_longer = (shortr_seqldat.coreseqletlen - min_overlap)

        #TODO: CALL COMPUTE SIM ON SLICES

    def compute_sim_on_slices(shortronehot, shortrhyp, shortrflanklen,
                              longeronehot, longerhyp, longerflanklen,
                              sim_metric):
        assert len(shortronehot) <= len(longeronehot)
        shortractual = shortronehot*shortrhyp
        longeractual = longeronehot*longerhyp

        shortrcorelen = len(shortronehot)-2*shortrflanklen 
        longercorelen = len(longeronehot)-2*longerflanklen
        min_overlap = int(shortrdatacorelen*min_overlap_frac)

        #iterate over all possible offsets of shortr's core relative to longerdata
        leftoffset = -(shortrcorelen-min_overlap) 
        rightoffset = longercorelen-min_overlap

        possible_offsets = list(range(leftoffset, rightoffset+1))
        sim_results = []
        for offset in possible_offsets:
            shortr_leftflank = max(offset,0) 
            shortr_rightflank = max(longercorelen-(offset+shortrcorelen),0)

            shortr_slicestart = shortrflanklen-shortr_leftflank
            shortr_sliceend = shortr_slicestart+shortrcorelen+shortr_rightflank

            longer_leftflank = max(-offset,0)
            longer_rightflank = max((offset+shortrcorelen)-longercorelen,0)
            longer_slicestart = longerflanklen-longer_leftflank
            longer_sliceend = longer_slicestart+longercorelen+longer_rightflank

            shortractual_slice = shortractual[shortr_slicestart:shortr_sliceend] 
            shortronehot_slice = shortronehot[shortr_slicestart:shortr_sliceend] 
            shortrhyp_slice = shortrhyp[shortr_slicestart:shortr_sliceend] 
            
            longeractual_slice = longeractual[longer_slicestart:longer_sliceend] 
            longeronehot_slice = longeronehot[longer_slicestart:longer_sliceend] 
            longerhyp_slice = longerhyp[longer_slicestart:longer_sliceend] 

            shortr_imputed = shortrhyp_slice*longeronehot_slice
            longer_imputed = longerhyp_slice*shortronehot_slice

            #TODO: put the imputed and actual together in one track
            raise RuntimeError("Do the thing above!")
            sim_results.append(
                 0.5*self.sim_metric(shortractual_slice, longer_imputed),
               + 0.5*self.sim_metric(shortr_imputed, longeractual_slice))

        return possible_offsets, sim_results


class SequenceAffmatComputer_ImputeActual(object):

    def __init__(self, min_overlap_frac=0.67, imputed_simcomputer):
        self.min_overlap_frac = min_overlap_frac 
        self.imputed_simcomputer = imputed_simcomputer

    def __call__(self, seqlets,
                 onehot_seqletdataextractor,
                 hyp_seqletdataextractor):

        #max_seqlet_len will return the length of the longest seqlet core
        max_seqlet_len = max(len(x) for x in seqlets)
 
        #for each seqlet, figure out the maximum size of the flank needed
        # on each size. This is determined by the length of the longest
        # seqlet that each seqlet could be compared against
        flank_sizes = [max_seqlet_len-
                       (len(seqlet)-int(len(seqlet)*self.min_overlap_frac))
                       for seqlet in seqlets] 
        all_onehot_seqletdata = np.array(
                            [onehot_seqletdataextractor(seqlet,
                             left_flank=flank, right_flank=flank)
                             for seqlet,flank in zip(seqlets,flank_sizes)]) 
        all_hyp_seqletdata = np.array(
                            [hyp_seqletdataextractor(seqlet,
                             left_flank=flank, right_flank=flank)
                             for seqlet,flank in zip(seqlets,flank_sizes)])
        assert onehot_seqletdata.shape==hyp_seqletdata.shape
        assert all([len(single_onehot_seqletdata)==(len(seqlet)+2*flanksize)
                    for (seqlet, single_onehot_seqletdata, flanksize) in
                    zip(seqlets, all_onehot_seqletdata, flank_sizes)])

        affmat = np.zeros(len(seqlets), len(seqlets))
        for i in range(len(seqlets)):
            for j in range(len(seqlets)):
                if (j >= i):
                    seqldat1 = _SeqlDatForImput( 
                        coreseqletlen=len(seqlet[i]),
                        onehot=all_onehot_seqletdata[i],
                        hyp=all_hyp_seqletdata[i])
                    seqldat2 = _SeqlDatForImput( 
                        coreseqletlen=len(seqlet[j]),
                        onehot=all_onehot_seqletdata[j],
                        hyp=all_hyp_seqletdata[j])
                    if (seqldat1.coreseqletlen <= seqldat2.coreseqletlen):
                        shortr_seqldat = seqldat1 
                        longer_seqldat = seqldat2
                    else:
                        shortr_seqldat = seqldat2 
                        longer_seqldat = seqldat1
                    sim = pair_simcomputer(
                            shortr_seqldat=shortr_seqldat,
                            longer_seqldat=longer_seqldat,
                            min_overlap_frac=min_overlap_frac)
                    affmat[i,j] = sim
                    affmat[j,i] = sim

        return affmat 
