#!/usr/bin/env python
from __future__ import print_function, division

import logging

import numpy as np

# ### Class to calculate average deeplift scores among overlapping sequences
class MergeOverlaps:

    def __init__(self, merged_hyp_scores_list, merged_seq_list,
                 chrom=None, merged_st=0, merged_en=0, core_size=400):
        ''' 
        constructor
        self.merged_hyp_scores_list : newly calculated average hyp scores list
        self.merged_seq_list : new sequence list (after merging/averaging overlaps)
        self.merged_st       : start of the new interval
        self.merged_en       : end   of the new interval
        self.merged_scores   : new scores array, will be add to merged_hyp_scores_list
        self.merged_counts   : counts array, has the same len as the self.merged_scores
        self.merged_seq      : new sequence, will be add to merged_seq_list
        self.max_seq_size    : max seq size this obj has seen
        '''
        self.merged_hyp_scores_list = merged_hyp_scores_list
        self.merged_seq_list        = merged_seq_list
        self.chrom         = chrom      # rest init at new interval
        self.merged_st     = merged_st
        self.merged_en     = merged_en
        self.merged_seq    = ""
        self.merged_scores = np.zeros((0, 4))
        self.merged_counts = np.zeros(0)
        self.core_size     = core_size
        self.max_seq_size  = 0
        self.in_seq_count  = 0
        self.out_seq_count = 0

        #logging.debug("MergeOverlaps: %s", 
        #              "append to seq_list" if self.merged_seq_list != None else "-" )

    def close_interval(self):

        if self.merged_st == self.merged_en: return # nothing to close

        ### generate a sequence between merged_st, merged_en ###

        self.merged_scores /= self.merged_counts[:, None] # calculate the average

        self.merged_hyp_scores_list.append(self.merged_scores)
        if self.merged_seq_list != None:
            self.merged_seq_list.append(self.merged_seq)

        if self.merged_en - self.merged_st > self.max_seq_size : # keeping track of the max sequence size
            self.max_seq_size = self.merged_en - self.merged_st

        '''
        logging.debug("close interval: %s:%d-%d, len=%d max=%d %s", 
                      self.chrom, self.merged_st, self.merged_en, 
                      self.merged_en - self.merged_st, self.max_seq_size,
                      "append to seq_list" if self.merged_seq_list != None else "-" )
        # write the interval to file
        if self.ofh :
            self.ofh.write(self.chrom + "\t" + str(self.merged_st) + "\t" + \
                           str(self.merged_en) + "\n")
        logging.debug("scores = ")
        logging.debug(self.merged_scores)
        logging.debug("counts = ")
        logging.debug(self.merged_counts)
        '''

        # reset for calculating the merged of the next set of overlapping intervals
        self.merged_st = self.merged_en
        self.merged_scores = np.zeros((0, 4))
        self.merged_counts = np.zeros(0)
        self.merged_seq    = ""

    def append_interval(self, chrom, st, en, hyp_scores, seq):
        '''
        process the next interval <st, en>

        Args:
            chrom:      chromsome of the interval to be processed
            st:         start of the interval
            en:         end   of the interval
            hyp_scores: hypothetical scores of the interval [st:en]
            seq:        sequences of the interval

        Returns:
            self.merged_en: the new ending position of the merged interval

                self.merged_st     self.merged_en
                             v     v
        self.merged_scores:  SSSSSS
                  O=Overlap    OOOOAA    A=Append
        hyp_scores:            HHHHHH
                               ^     ^
                               st    en
        '''
        
        assert(st >= self.merged_st) # intervals must be sorted
        if chrom != self.chrom or st > self.merged_en : # handle the non-overlapping case
            self.chrom  = chrom
            self.merged_st = st
            self.merged_en = st

        '''
        length of the overlap = self.merged_en - st
        overlap is self.merged_scores[st - self.merged_st : self.merged_en - self.merged_st]
        overlap is hyp_scores[0 : self.merged_en - st]
        '''
        if st > self.merged_st : # non-empty overlap
            self.merged_scores[st - self.merged_st : self.merged_en - self.merged_st] += \
                hyp_scores[0:self.merged_en - st]
            self.merged_counts[st - self.merged_st : self.merged_en - self.merged_st] += \
                np.ones(self.merged_en - st)
        
        '''
        part to append, len = en - self.merged_en
        hyp_scores[self.merged_en-st : en-st    ]
        '''
        if en > self.merged_en : # non-empty append
            self.merged_scores = np.concatenate((self.merged_scores,
                                              hyp_scores[self.merged_en - st : en - st]), 
                                              axis = 0)
            self.merged_counts = np.concatenate((self.merged_counts, np.ones(en-self.merged_en)), 
                                              axis = 0)
            if self.merged_seq_list != None:
                self.merged_seq    += seq[self.merged_en - st : en - st]
            self.merged_en = en

        return self.merged_en


    def process_one_interval(self, chrom, st, en, hyp_scores, seq):
        
        seq_size = en - st
        left     = int((seq_size - self.core_size) / 2)
        right    = left + self.core_size

        if chrom != self.chrom or st > self.merged_en: # start a new interval
            self.close_interval()

        self.append_interval(chrom, st + left, st + right, hyp_scores[left:right], 
                             seq[left:right])

        #logging.debug("processed interval %s:%d-%d, reduce to %d-%d",
        #              chrom, st, en, st+left, st+right)

def merge_overlaps(in_tsv_fn, hyp_scores_all, merged_hyp_scores_list, seq_list, merged_seq_list):

    with open(in_tsv_fn,'r') as tsvin:
        merged = MergeOverlaps(merged_hyp_scores_list, merged_seq_list)
        for idx, line in enumerate(tsvin):
            row = line.split()
            chrom = row[0]
            st    = int(row[1])
            en    = int(row[2])

            hyp_scores = hyp_scores_all[idx]
            merged.process_one_interval(chrom, st, en, hyp_scores, seq_list[idx])

        logging.debug("merged overlaps based on in_tsv %s, %d seqs merged into %d seqs, max len %d" %
                      (in_tsv_fn, len(hyp_scores_all ), len(merged_hyp_scores_list), 
                       merged.max_seq_size))



