from __future__ import division, print_function, absolute_import
from ..affinitymat.core import MeanNormalizer
from .. import backend as B
from .core import (AbstractSeqletsToOnedEmbedder,
                   AbstractSeqletsToOnedEmbedderFactory)
from .. import core as modiscocore
import itertools
import numpy as np
import sys


class GappedKmerEmbedderFactory(object):

    def __init__(self, alphabet_size=4, kmer_len=8, num_gaps=3,
                       num_mismatches=2, gpu_batch_size=20,
                       num_filters_to_retain=None,
                       #note: I should experiment to see whether this
                       # mean normalization helps results...it might not
                       mean_normalize=True):
        self.alphabet_size = alphabet_size
        self.kmer_len = kmer_len
        self.num_gaps = num_gaps
        self.num_mismatches = num_mismatches
        self.gpu_batch_size = gpu_batch_size
        self.num_filters_to_retain = num_filters_to_retain
        self.mean_normalize = mean_normalize
        if (self.mean_normalize):
            self.normalizer = MeanNormalizer()
        else:
            self.normalizer = lambda x: x

    def get_jsonable_config(self):
        return OrderedDict([
                ('alphabet_size', self.alphabet_size),
                ('kmer_len', self.kmer_len),
                ('num_gaps', self.num_gaps),
                ('num_mismatches', self.alphabet_size),
                ('gpu_batch_size', self.gpu_batch_size),
                ('num_filters_to_retain', self.num_filters_to_retain),
                ('mean_normalize', self.mean_normalize),
                ])

    def __call__(self, onehot_track_name, toscore_track_names_and_signs):
        return GappedKmerEmbedder(
                alphabet_size=self.alphabet_size,
                kmer_len=self.kmer_len,
                num_gaps=self.num_gaps,
                num_mismatches=self.num_mismatches,
                onehot_track_name=onehot_track_name,
                toscore_track_names_and_signs=toscore_track_names_and_signs,
                normalizer=self.normalizer,
                batch_size=self.gpu_batch_size,
                num_filters_to_retain=self.num_filters_to_retain) 


class GappedKmerEmbedder(AbstractSeqletsToOnedEmbedder):
    
    def __init__(self, alphabet_size,
                       kmer_len,
                       num_gaps,
                       num_mismatches,
                       toscore_track_names_and_signs,
                       normalizer,
                       batch_size,
                       num_filters_to_retain=None,
                       onehot_track_name=None,
                       progress_update=None):
        self.alphabet_size = alphabet_size
        self.kmer_len = kmer_len
        self.num_gaps = num_gaps
        self.num_mismatches = num_mismatches
        self.num_filters_to_retain = num_filters_to_retain
        self.filters, self.biases = self.prepare_gapped_kmer_filters()
        self.onehot_track_name = onehot_track_name
        self.toscore_track_names_and_signs = toscore_track_names_and_signs
        assert len(toscore_track_names_and_signs) >= 0,\
            "toscore_track_names_and_signs length is 0"
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.progress_update = progress_update
        self.require_onehot_match = (True if self.onehot_track_name
                                     is not None else False)
        self.gapped_kmer_embedding_func =\
            B.get_gapped_kmer_embedding_func(
                    filters=self.filters,
                    biases=self.biases,
                    require_onehot_match=self.require_onehot_match)

    def prepare_gapped_kmer_filters(self):
        nonzero_position_combos = list(itertools.combinations(
                            iterable=range(self.kmer_len),
                            r=(self.kmer_len-self.num_gaps)))
        letter_permutations = list(itertools.product(
                                *[list(range(self.alphabet_size)) for x in
                                  range(self.kmer_len-self.num_gaps)]))
        filters = []
        biases = []
        ##removed: unique_nonzero_positions = set()
        for nonzero_positions in nonzero_position_combos:
            string_representation = [" " for x in range(self.kmer_len)]
            for nonzero_position in nonzero_positions:
                string_representation[nonzero_position] = "X"
            nonzero_positions_string =\
                ("".join(string_representation)).lstrip().rstrip()
            #The logic for using 'unique_nonzero_positions' was that
            # ' XX' and 'XX ' are in principle equivalent, so I did not want
            # to double-count them. However, ' XX' and 'XX ' would generate
            # slightly different embeddings, and if we don't include both, then 
            # the forward and reverse-complement version of the same seqlet 
            # would not wind up generating equivalent embeddings... 
            #So I have decided to just accept the double-counting and
            # include both in order to preserve reverse-complement symmetry
            ##removed: if (nonzero_positions_string not in unique_nonzero_positions):
            ##removed: unique_nonzero_positions.add(nonzero_positions_string) 
            for letter_permutation in letter_permutations:
                assert len(nonzero_positions)==len(letter_permutation)
                the_filter = np.zeros((self.kmer_len, self.alphabet_size))
                for nonzero_position, letter\
                    in zip(nonzero_positions, letter_permutation):
                    the_filter[nonzero_position, letter] = 1 
                filters.append(the_filter)
                biases.append(-(len(nonzero_positions)-1
                                -self.num_mismatches))
        return np.array(filters), np.array(biases)

    def __call__(self, seqlets):
        print("Computing embeddings")
        sys.stdout.flush()
        assert self.require_onehot_match #legacy, now just assume it's True
        onehot_track_fwd, onehot_track_rev =\
            modiscocore.get_2d_data_from_patterns(
                patterns=seqlets,
                track_names=[self.onehot_track_name],
                track_transformer=None)
            

        data_to_embed_fwd = np.zeros(
            (len(seqlets),
             len(list(seqlets)[0]), self.alphabet_size)).astype("float32")
        data_to_embed_rev = (np.zeros(
            (len(seqlets),
             len(list(seqlets)[0]), self.alphabet_size)).astype("float32")
            if (onehot_track_rev is not None) else None)
        for (track_name, sign) in self.toscore_track_names_and_signs:
            fwd_data, rev_data = modiscocore.get_2d_data_from_patterns(
                patterns=seqlets,
                track_names=[track_name], track_transformer=None)  
            data_to_embed_fwd += fwd_data*sign
            if (rev_data is not None):
                data_to_embed_rev += (rev_data*sign if
                                      (rev_data is not None) else None)
        data_to_embed_fwd = np.array([self.normalizer(x) for x in
                                      data_to_embed_fwd])
        if (data_to_embed_rev is not None):
            data_to_embed_rev = np.array([self.normalizer(x) for x in
                                          data_to_embed_rev])
        common_args = {'batch_size': self.batch_size,
                       'progress_update': self.progress_update}
        if (self.require_onehot_match):
            embedding_fwd = self.gapped_kmer_embedding_func(
                                  onehot=onehot_track_fwd,
                                  to_embed=data_to_embed_fwd,
                                  **common_args)
            embedding_rev = (self.gapped_kmer_embedding_func(
                                  onehot=onehot_track_rev,
                                  to_embed=data_to_embed_rev,
                                  **common_args)
                             if (onehot_track_rev is not None) else None)
        else:
            embedding_fwd = self.gapped_kmer_embedding_func(
                                  to_embed=data_to_embed_fwd,
                                  **common_args)
            embedding_rev = (self.gapped_kmer_embedding_func(
                                  to_embed=data_to_embed_rev,
                                  **common_args)
                             if (onehot_track_rev is not None) else None)
        if (self.num_filters_to_retain is not None):
            all_embeddings = (np.concatenate(
                              [embedding_fwd, embedding_rev], axis=0)
                              if (embedding_rev is not None)
                              else np.array(embedding_fwd))
            embeddings_denominators =\
                np.sum(np.abs(all_embeddings) > 0, axis=0).astype("float")
            embeddings_denominators += 10.0
            embeddings_mean_impact =\
                (np.sum(np.abs(all_embeddings), axis=0)/
                 embeddings_denominators)
            top_embedding_indices = [
                x[0] for x in sorted(enumerate(embeddings_mean_impact),
                key=lambda x: -x[1])][:self.num_filters_to_retain]
            embedding_fwd = embedding_fwd[:,top_embedding_indices]
            embedding_rev = (embedding_rev[:,top_embedding_indices]
                             if (embedding_rev is not None) else None)
        return embedding_fwd, embedding_rev
