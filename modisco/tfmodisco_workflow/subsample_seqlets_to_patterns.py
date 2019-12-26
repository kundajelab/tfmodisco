from __future__ import division, absolute_import, print_function
from .. import core
from .. import util
from ..util import print_memory_use
import logging


class TfModiscoSubsampleSeqletsToPatterns(object):

    def __init__(self, subsample_size, logging_level=logging.DEBUG):
        self.subsample_size = subsample_size 
        self.logger = logging.getLogger("SubsampleSeqletsToPatterns") 
        self.logger.setLevel(logging_level)

    def __build__(self):
        pass 

    def __call__(self, seqlets, track_set, runtime_settings):
        
        all_patterns = []
        seqlets_left = seqlets
        iteration = 0
        while (True):
            self.logger.info("On iteration "+str(iteration))
            subampled_seqets = #TODO: subsample from seqlets_left
            patterns_this_iter = self.run_on_subsample(seqlets=subsampled_seqlets) 
            if (len(patterns_this_iter)==0):
                self.logger.info("No patterns found in iteration "
                                 +str(iteration))
                break
            seqlets_left, new_patterns = self.soak(patterns=patterns_this_iter,
                                                   seqlets=seqlets_left)
            

    def run_on_subsample(self, seqlets):
    
        #TODO: compute affinity matrix
        affmat = #TODO: compute fine-grained affmat
        leiden_clustering = #TODO: apply leiden to affmat
        patterns = #TODO: aggregate according to clustering 
        return patterns
