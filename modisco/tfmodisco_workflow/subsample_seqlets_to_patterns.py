from __future__ import division, absolute_import, print_function
from .. import core
from .. import util as modiscoutil
from ..util import print_memory_use
import logging
from collections import namedtuple
from . import affinitymat
import numpy as np


class SubsampleSeqletsToPatterns_SequenceBased(object):

    def __init__(self, subsample_size, min_overlap_frac,
                       seed,
                       logging_level=logging.DEBUG):
        self.subsample_size = subsample_size 
        self.min_overlap_frac = min_overlap_frac
        self.logger = logging.getLogger("SubsampleSeqletsToPatterns") 
        self.logger.setLevel(logging_level)
        self.seed = seed

    def _build(self, onehot_track_name, hyp_track_name):
        self.affmat_computer = SequenceAffmatComputer_Impute(
                min_overlap_frac=self.min_overlap_frac,
                metric=modiscoutil.l1norm_contin_jaccard_sim)
        self.onehot_track_name = onehot_track_name
        self.hyp_track_name = hyp_track_name
        self.rng = np.random.RandomState(self.seed) 

    def __call__(self, seqlets, track_set, **buildkwargs):

        self._build(**buildkwargs)

        all_patterns = []
        seqlets_left = seqlets
        iteration = 0
        while (True):
            self.logger.info("On iteration "+str(iteration))
            subampled_seqets = [seqlets_left[i] for i in
                              self.rng.choice(a=np.arange(self.subsample_size),
                                              size=self.subsampled_seqlets,
                                              replace=False)]
            patterns_this_iter = self.run_on_subsample(
                                       seqlets=subsampled_seqlets) 
            if (len(patterns_this_iter)==0):
                self.logger.info("No patterns found in iteration "
                                 +str(iteration))
                break
            seqlets_left, new_patterns = self.soak(patterns=patterns_this_iter,
                                                   seqlets=seqlets_left)
            

    def run_on_subsample(self, seqlets):
        affmat = self.affmat_computer(seqlets=seqlets,
                     onehot_track_name=self.onehot_track_name,
                     hyp_track_name=self.hyp_track_name)
        leiden_clustering = #TODO: apply leiden to affmat
        patterns = #TODO: aggregate according to clustering 
        return patterns
