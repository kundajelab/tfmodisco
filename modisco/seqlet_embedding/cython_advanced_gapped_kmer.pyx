
def fast_recursively_get_gappedkmersandimp(posbaseimptuples, int max_k,
                                           int max_gap, int max_len):
    #preceding_posbaseimptuples: [(0-based-position, base, imp)...]
    #A gapped kmer representation: [(gapbefore, base)...]
    #Gappedkmersandimp: [
    #   (gapped_kmer_representation, totalimp)] <-- smallest first
    #startposandgappedkmersandimp: [ (startpos, gappedkmersandimp) ]
    #endpos_and_startposandgappedkmersandimp: [
    #   (endpos, startpos_and_gappedkmersandimp) ] <- earliest end first
    if (len(posbaseimptuples)==0):
        return []
    else:
        lastbasepos, lastbase, lastbaseimp = posbaseimptuples[-1]
        endpos_and_startposandgappedkmersandimp =\
            fast_recursively_get_gappedkmersandimp(
                posbaseimptuples[:-1], max_k=max_k,
                max_gap=max_gap, max_len=max_len)
        
        #fill out startposandgappedkmersandimp for this ending position
        startposandgappedkmersandimp_endingatthispos = []
        
        #maintain the property of 'latest start first';
        # lastbasepos is the start for the kmer of k=1
        startposandgappedkmersandimp_endingatthispos.append(
            (lastbasepos, [ ([(0, lastbase)], lastbaseimp) ] ) )
        #iterate in order of latest end first, as this will
        # allow us to 'break' early when
        # we get to endpositions that would violate the 'max_gap' criterion
        for (endpos, startposandgappedkmersandimp)\
                in endpos_and_startposandgappedkmersandimp[::-1]:     
            if ( (lastbasepos-endpos)+1 <= max_gap ):
                #iterate through startposandgappedkmersandimp in order.
                # This will go through latest start first. As a result, we
                # will be able to 'break' early when we encounter a startpos
                # that would violate the max_len criterion.
                for startpos, gappedkmersandimp in\
                        startposandgappedkmersandimp:
                    gappedkmersandimp_startingatthispos = []
                    if ( (lastbasepos-startpos)+1 <= max_len):
                        #iterate through gappedkmersandimp in forward order.
                        # This iterates through in order of smallest
                        # gappedkmer_rep first. As a result, we can break out
                        # of the loop early when we encounter a
                        # len(gappedkmer_rep) that would violate max_k
                        for (gappedkmer_rep, totalimp) in gappedkmersandimp:
                            if (len(gappedkmer_rep) < max_k):
                                #because we iterate through gappedkmersandimp
                                # in order of smallest gappedkmer_rep first,
                                # gappedkmersandimp_startingatthispos will
                                # also maintain that property.
                                gappedkmersandimp_startingatthispos.append(
                                    (gappedkmer_rep
                                     +[(lastbasepos-endpos, lastbase)],
                                     totalimp+lastbaseimp) )
                            else:
                                break
                    if len(gappedkmersandimp_startingatthispos) > 0:
                        #would need to sort this later to make sure property of
                        # being sorted in descending order of startpos is
                        # preserved
                        startposandgappedkmersandimp_endingatthispos.append(
                            (startpos, gappedkmersandimp_startingatthispos) )
            else:
                break #can stop iterating through
                      # endpos_and_startposandgappedkmersandimp
        
        endpos_and_startposandgappedkmersandimp.append(
            (lastbasepos+1, startposandgappedkmersandimp_endingatthispos  ) )
        
        return endpos_and_startposandgappedkmersandimp


def unravel_fast_recursively_get_gappedkmersandimp(
    posbaseimptuples, int max_k, int max_gap, int max_len):
    endpos_and_startposandgappedkmersandimp =\
        fast_recursively_get_gappedkmersandimp(
            posbaseimptuples=posbaseimptuples, max_k=max_k,
            max_gap=max_gap, max_len=max_len)
    return [(tuple(x[0]), x[1]) for endpos,startposandgappedkmersandimp
            in endpos_and_startposandgappedkmersandimp
            for startpos,gappedkmersandimp in startposandgappedkmersandimp
            for x in gappedkmersandimp]


def get_agkmer_to_totalseqimp(gappedkmersandimp, int min_k):
    gapped_kmer_to_totalseqimp = {}
    for gapped_kmer_rep, gapped_kmer_imp in gappedkmersandimp:
        assert gapped_kmer_rep[0][0]==0 #no superfluous pre-padding
        if (len(gapped_kmer_rep) >= min_k):
            gapped_kmer_to_totalseqimp[gapped_kmer_rep] = (
                gapped_kmer_to_totalseqimp.get(gapped_kmer_rep, 0)
                + gapped_kmer_imp)
    return gapped_kmer_to_totalseqimp
