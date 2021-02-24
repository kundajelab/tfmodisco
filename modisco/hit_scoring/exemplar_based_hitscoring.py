from __future__ import division, print_function
from collections import defaultdict, OrderedDict, namedtuple
import numpy as np
import time
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.isotonic import IsotonicRegression
from .. import affinitymat
from .. import aggregator
from .. import core
from .. import util
from .. import visualization
from matplotlib import pyplot as plt
import sklearn
from joblib import Parallel, delayed


MotifHitAndCoord = namedtuple("MotifHitAndCoord",
                    ["motif_idx", "motif_score", "precision_at_motif_score",
                     "example_idx", "start", "end", "is_revcomp"]) 


def facility_locator(distmat, num_exemplars):
    exemplars = [] 
    current_bestrep = np.inf*np.ones(distmat.shape[0])
    for i in range(min(num_exemplars, len(distmat))):
        candidate_newbestrep = np.minimum(distmat, current_bestrep[None,:])  
        candidate_objective = np.sum(candidate_newbestrep, axis=-1) 
        next_best_exemplar = np.argmin(candidate_objective) 
        exemplars.append(next_best_exemplar)
        current_bestrep = candidate_newbestrep[next_best_exemplar]
    return exemplars


def compute_continjacc_sims_1vmany(vec1, vecs2, vecs2_weighting):
    sign_vec1, signs_vecs2 = np.sign(vec1), np.sign(vecs2)
    abs_vec1, abs_vecs2 = np.abs(vec1), np.abs(vecs2)
    intersection = np.sum((np.minimum(abs_vec1[None,:], abs_vecs2[:,:])
                 *sign_vec1[None,:]*signs_vecs2[:,:])*vecs2_weighting, axis=-1)
    union = np.sum(np.maximum(abs_vec1[None,:],
                   abs_vecs2[:,:])*vecs2_weighting, axis=-1)
    return intersection/union


def compute_pairwise_continjacc_sims(vecs1, vecs2, n_jobs,
                                     vecs2_weighting=None):
    #normalize vecs2_weighting to sum to 1
    if (vecs2_weighting is None):
        vecs2_weighting = np.ones_like(vecs2)
    assert np.min(vecs2_weighting) >= 0
    return np.array(Parallel(n_jobs=n_jobs, verbose=True)(
            delayed(compute_continjacc_sims_1vmany)(
                     vec1, vecs2, vecs2_weighting) for vec1 in vecs1))
  

def make_aggregated_seqlet(seqlets):
    seqletsandalignments = core.SeqletsAndAlignments()
    [seqletsandalignments.append(core.SeqletAndAlignment(
        seqlet=seqlet,
        alnmt=0)) for seqlet in seqlets if seqlet not in seqletsandalignments]
    reconstructed_motif = core.AggregatedSeqlet(seqletsandalignments)
    return reconstructed_motif

    
def get_exemplar_motifs(seqlets, pattern_comparison_settings,
                            seqlets_per_exemplar, max_exemplars,
                            affmat_min_frac_of_median, n_jobs):
    """This identifies the exemplars among seqlets

    Args:
        seqlets: A list of :class:`modisco.core.Seqlet` that are already
            aligned relative to each other
        pattern_comparison_settings: determines which tracks to fetch,
            as well as normalization settings.
            Instance of modisco.affinitymat.core.PatternComparisonSettings.
        seqlets_per_exemplar: determines how many exemplars to
            look for by looking for the total number of seqlets and dividing by
            this; number of exemplars will be capped at max_exemplars
        max_exemplars: caps the maximum number of exemplars
        affmat_min_frac_of_median: Kick out seqlets that have poor
            within-cluster similarity relative to the median within-cluster
            similarity.
        n_jobs: number of jobs to launch when computing similarities

    Returns
        motifs: Exemplar motifs, sorted by number of seqlets (only exemplars
            for the passing seqlets are considered)
        affmat: Pairwise affinity matrix for all the *passing* seqlets
        filtered_orig_motif: The result of kicking out weak seqlets from
            the original motif
        sum_orig_affmat: For visualization purposes, can plot the
            distribution of the sum of the within-cluster similarity for
            each seqlet in the original motif. Is a vector.
    """
    print("Numseqles:", len(seqlets))
    #seqlets should already be aligned relative to each other.
    # Extract the importance score information.
    fwd_seqlet_data, _ = core.get_2d_data_from_patterns(
        patterns=seqlets,
        track_names=pattern_comparison_settings.track_names,
        track_transformer=
         pattern_comparison_settings.track_transformer)
    #flatten the fwd_seqlet_data (they are aligned so it's ok to flatten
    # them before doing comparisons)
    fwd_seqlet_data_vectors = util.flatten_seqlet_impscore_features(
                                        fwd_seqlet_data)
    #compute the affinity matrix
    orig_affmat = compute_pairwise_continjacc_sims(
        vecs1=fwd_seqlet_data_vectors,
        vecs2=fwd_seqlet_data_vectors,
        n_jobs=n_jobs)
    #Let's kick out seqlets for which the sum of the affmat across all
    # neighbors is less than affmat_min_frac_of_median
    sum_orig_affmat = np.sum(orig_affmat, axis=-1)
    median_sum_affmat = np.median(sum_orig_affmat)
    passing_mask = (sum_orig_affmat >
                    affmat_min_frac_of_median*median_sum_affmat)
    
    #get a new affmat and seqlets that are subsetted
    affmat = orig_affmat[passing_mask,:][:,passing_mask]
    seqlets = [seqlet for seqlet,is_passing in zip(seqlets, passing_mask)
               if is_passing]
    filtered_orig_motif = make_aggregated_seqlet(seqlets)
    
    #convert to distance matrix
    distmat = 1/( np.maximum(affmat,1e-7) )
    #get exemplars
    seqlet_exemplar_indices = facility_locator(
        distmat=distmat,
        num_exemplars=min(max_exemplars,
                          int(np.ceil(len(seqlets)/seqlets_per_exemplar)) ))
    #aggregate over the similar ones, return the aggseqlets
    representive_exemplars = np.argmax(affmat[:, seqlet_exemplar_indices],
                                         axis=-1)
    exemplar_to_seqletsandalignments = OrderedDict()
    for seqlet, representive_exemplar in zip(seqlets, representive_exemplars):
        if (representive_exemplar not in exemplar_to_seqletsandalignments):
            exemplar_to_seqletsandalignments[representive_exemplar] = []
        exemplar_to_seqletsandalignments[representive_exemplar].append(
            core.SeqletAndAlignment(seqlet=seqlet, alnmt=0) )
    exemplar_to_motif = OrderedDict([
        (exemplar, core.AggregatedSeqlet(seqletsandalignments))
        for exemplar,seqletsandalignments in
        exemplar_to_seqletsandalignments.items()])
    #return the list of motifs, sorted by the number of seqlets
    motifs = sorted(list(exemplar_to_motif.values()),
                    key=lambda x: len(x.seqlets))
    return motifs, affmat, filtered_orig_motif, sum_orig_affmat


def get_exemplar_motifs_for_all_patterns(
    patterns, pattern_comparison_settings,
    affmat_min_frac_of_median, n_jobs):

    print("Getting the exemplar motifs")
    #Take each pattern
    #Identify some number of exemplars and aggregate around them
    exemplarmotifs_foreach_pattern = []
    #indices that mark when the exemplars for one pattern starts
    # and another pattern ends
    exemplarmotifs_indices = [0] 
    withinpattern_affmats = []
    filt_patterns = []
    for pattern in patterns:
        (exemplarmotifs, patternaffmat,
         filtered_orig_motif, sum_orig_affmat) = get_exemplar_motifs(
          seqlets=pattern.seqlets,
          pattern_comparison_settings=pattern_comparison_settings,
          seqlets_per_exemplar=30,
          max_exemplars=10,
          affmat_min_frac_of_median=affmat_min_frac_of_median,
          n_jobs=n_jobs)
        exemplarmotifs_foreach_pattern.append(exemplarmotifs)
        exemplarmotifs_indices.append(len(exemplarmotifs_foreach_pattern))
        withinpattern_affmats.append(patternaffmat)
        filt_patterns.append(filtered_orig_motif)
        
        visualization.viz_sequence.plot_weights(pattern["sequence"].fwd)
        plt.hist(sum_orig_affmat, bins=20)
        plt.show()
        print("After filtering: numseqlets", len(filtered_orig_motif.seqlets))
        visualization.viz_sequence.plot_weights(
            filtered_orig_motif["sequence"].fwd)

    return (exemplarmotifs_foreach_pattern, exemplarmotifs_indices,
            withinpattern_affmats, filt_patterns)


#gets the shifts of the provided coordinate - everything except 0
def get_shifts(seqlet_coordinate, shift_fraction, max_seq_len):
    shift_size_in_bp = int((seqlet_coordinate.end-
                            seqlet_coordinate.start)*shift_fraction)
    coordinates_to_return = []
    for shift_size in range(-shift_size_in_bp,shift_size_in_bp+1):
        for is_revcomp in [True, False]:
            new_start = seqlet_coordinate.start + shift_size
            new_end = seqlet_coordinate.end + shift_size
            if (new_start >= 0 and new_end <= max_seq_len):
                coordinates_to_return.append(core.SeqletCoordinates(
                    example_idx=seqlet_coordinate.example_idx,
                    start=new_start,
                    end=new_end,
                    is_revcomp=is_revcomp))
    return coordinates_to_return


def get_coordinates_and_labels(shift_fraction, patterns, track_set):
    """
    Get coordinates from shifting the seqlet instances by shift_fraction,
        and get labels for shifts that align with the original seqlets
    """
    print("Getting labels")
    all_coordinates = [
        coor
        for pattern in patterns
        for seqlet in pattern.seqlets
        for coor in get_shifts(
            seqlet_coordinate=seqlet.coor,
            shift_fraction=shift_fraction,
            max_seq_len=track_set.get_example_idx_len(seqlet.coor.example_idx),
            )
    ]

    patternidx_to_positivecoordinates = OrderedDict([
        (patternidx, set(str(seqlet.coor) for seqlet
                         in patterns[patternidx].seqlets))
        for patternidx in range(len(patterns))
    ])

    #get the labels for the coordinates depending on the patterns
    # the very last column is the 'no pattern' class
    labels = np.zeros((len(all_coordinates), 1+len(patterns)))
    for patternidx in range(len(patterns)):
        labels[:,patternidx] = np.array([
            1 if str(coor) in
              patternidx_to_positivecoordinates[patternidx] else 0
            for coor in all_coordinates ])
    #fill in last col as a 1 if nothing else is
    labels[:,-1] = np.array([1 if x==0 else 0 for x
                             in np.sum(labels, axis=-1)])

    return all_coordinates, labels


class FeaturesProducer(object):

    def __init__(self, motifs, pattern_comparison_settings,
                       onehot_track_name, bg_freq, n_jobs):
        self.motifs = motifs
        self.pattern_comparison_settings = pattern_comparison_settings
        self.onehot_track_name = onehot_track_name
        self.bg_freq = bg_freq
        self.n_jobs = n_jobs

        #Get imp scores data
        (allexemplarmotifs_impscoresdata_fwd,
         allexemplarmotifs_impscoresdata_rev) =\
            core.get_2d_data_from_patterns(
                patterns=motifs,
                track_names=pattern_comparison_settings.track_names,
                track_transformer=pattern_comparison_settings.track_transformer)
        #Flatten the importance score data into vectors
        self.allexemplarmotifs_impscoresdata_fwd = (
            util.flatten_seqlet_impscore_features(
                allexemplarmotifs_impscoresdata_fwd))

        #Do the same for per-position IC (for weighting exemplar sim
        # computation). First, get the one-hot encoded sequence data
        allexemplarmotifs_sequence_fwd, allexemplarmotifs_sequence_rev =\
            core.get_2d_data_from_patterns(
                patterns=motifs,
                track_names=[onehot_track_name],
                track_transformer=lambda x: x)
        #compute the per-position IC, then tile (for ACGT) and flatten to
        # get it into vector form.
        self.per_position_ic_allexemplarmotifs_fwd =\
            np.maximum(util.flatten_seqlet_impscore_features(np.array([
                np.tile(util.compute_per_position_ic(
                    ppm=x,
                    background=bg_freq,
                    pseudocount=0.001)[:,None],
                  (1,4*len(pattern_comparison_settings.track_names)))
                for x in allexemplarmotifs_sequence_fwd])),0)

    def __call__(self, coordinates, track_set):
        print("Getting impscores data")
        seqlets = track_set.create_seqlets(coords=coordinates)
        impscoresdata_fwd, _ =\
            core.get_2d_data_from_patterns(
               patterns=seqlets,
               track_names=self.pattern_comparison_settings.track_names,
               track_transformer=
                self.pattern_comparison_settings.track_transformer)
        #Flatten the importance score data into vectors
        impscoresdata_fwd = (util.flatten_seqlet_impscore_features(
                             impscoresdata_fwd))

        start = time.time()
        print("Computing fwd sims")
        features_matrix_fwd = compute_pairwise_continjacc_sims(
            vecs1=impscoresdata_fwd,
            vecs2=self.allexemplarmotifs_impscoresdata_fwd,
            vecs2_weighting=self.per_position_ic_allexemplarmotifs_fwd,
            n_jobs=self.n_jobs)
        print("Took",time.time()-start,"s")

        #We ignore the rc because we want to annotate seqlets as
        # matches *for a specific orientation*
        return features_matrix_fwd


class InstanceScorer(object):

    def __init__(self, features_producer, classifier):
        self.features_producer = features_producer
        self.classifier = classifier

    def _call_batch(self, coordinates, track_set):
        features_matrix = self.features_producer(coordinates=coordinates,
                                                 track_set=track_set) 
        if (hasattr(self.classifier, 'predict_proba')):
            return self.classifier.predict_proba(features_matrix)
        else:
            return self.classifier.predict(features_matrix)
   
    def __call__(self, coordinates, track_set, batch_size=None):
        if (batch_size is None):
            batch_size = len(coordinates)
        to_return = []
        for idx in range(0,len(coordinates), batch_size):
           to_return.extend(self._call_batch(
            coordinates=coordinates[idx:idx+batch_size], track_set=track_set))
        return np.array(to_return)

    def get_prec_for_threshold(self, motif_idx, threshold):
        if (hasattr(threshold, '__iter__')==False):
            return self.prec_ir_list[motif_idx].transform([threshold])[0]
        else:
            return self.prec_ir_list[motif_idx].transform(threshold)

    def compute_precrecthres_list(self, coordinates, track_set, labels):
        """
        Prepare the attribute self.precrecthres_list which, for each
            pattern, has (precision, recall, threshold) as returned
            by scipy's precision_recall_curve  function. The
            precision recall curve is computed accoridng to
            coordiantes and labels. The last column of labels
            corresponds to the "no pattern" class.
        """
        preds = self(coordinates=coordinates, track_set=track_set)
        assert np.min(preds) >= 0 #relevant when assuming min threshold is 0
        prec_ir_list = []
        precision_list = []
        recall_list = []
        thresholds_list = []
        for pattern_idx in range(labels.shape[1]):
            ir = IsotonicRegression(out_of_bounds='clip').fit(
                X=preds[:,pattern_idx], y=labels[:,pattern_idx])
            prec_ir_list.append(ir)
            precision, recall, thresholds = precision_recall_curve(
                    y_true=labels[:,pattern_idx],
                    probas_pred=preds[:,pattern_idx]) 
            precision_list.append(precision)
            recall_list.append(recall)
            thresholds_list.append(thresholds)

        self.prec_ir_list = prec_ir_list
        self.precision_list = precision_list
        self.recall_list = recall_list
        self.thresholds_list = thresholds_list

        return (precision_list, recall_list, thresholds_list)


def prepare_instance_scorer(
    patterns,
    trim_window_size,
    task_names,
    bg_freq,
    track_set,
    affmat_min_frac_of_median=0.6,
    classifier_to_fit_factory=(
    lambda: sklearn.linear_model.LogisticRegression(
                class_weight='balanced',
                multi_class='multinomial',
                verbose=5,
                random_state=1234,
                n_jobs=10,
                max_iter=3000)),
    shift_fraction=0.3,
    min_overlap=0.7,
    n_jobs=10):

    onehot_track_name = "sequence"
    score_track_names = ([task_name+"_hypothetical_contribs"
                          for task_name in task_names]
                         +[task_name+"_contrib_scores"
                           for task_name in task_names])

    pattern_comparison_settings =\
        affinitymat.core.PatternComparisonSettings(                         
                track_names=score_track_names,                      
                track_transformer=affinitymat.L1Normalizer(),                   
                min_overlap=min_overlap)

    #start by trimming the patterns to the lengths of the original seqlets
    prefilt_trimmed_patterns = aggregator.TrimToBestWindowByIC(                                    
                        window_size=trim_window_size,                           
                        onehot_track_name=onehot_track_name,                            
                        bg_freq=bg_freq)(patterns)
    (exemplarmotifs_foreach_pattern,
     exemplarmotifs_indices,
     withinpattern_affmats,
     filt_trimmed_patterns) = get_exemplar_motifs_for_all_patterns(
            patterns=prefilt_trimmed_patterns,
            pattern_comparison_settings=pattern_comparison_settings,
            affmat_min_frac_of_median=affmat_min_frac_of_median,
            n_jobs=n_jobs)


    #get the flattened list of exemplar motifs and make FeaturesProducer
    all_exemplarmotifs = [exemplarmotif
        for patternidx in range(len(filt_trimmed_patterns)) 
        for exemplarmotif in exemplarmotifs_foreach_pattern[patternidx]]
    features_producer = FeaturesProducer(
        motifs=all_exemplarmotifs,
        pattern_comparison_settings=pattern_comparison_settings,
        onehot_track_name=onehot_track_name,
        bg_freq=bg_freq,
        n_jobs=n_jobs)

    #get coordinates, labels and their features
    all_coordinates, labels = get_coordinates_and_labels(
                               shift_fraction=shift_fraction,
                               patterns=filt_trimmed_patterns,
                               track_set=track_set)
    features_matrix = features_producer(coordinates=all_coordinates,
                                        track_set=track_set)
    
    classifier = classifier_to_fit_factory().fit(
                    features_matrix, np.argmax(labels, axis=-1))

    instance_scorer = InstanceScorer(features_producer=features_producer,
                                     classifier=classifier)
    instance_scorer.compute_precrecthres_list(coordinates=all_coordinates,
                                              track_set=track_set,
                                              labels=labels)
    return instance_scorer


def get_windows_to_be_scanned_interior(
    transformed_scoretrack, transformed_scoretrack_bestwindowwidth,
    val_transformer, scanning_window_width, cutoff_value, plot_save_dir="."):
    sliding_window_sizes = val_transformer.sliding_window_sizes
    #boolean arrays containing which values are above the cutoff
    values_above_cutoff = [(x >= cutoff_value) for x in transformed_scoretrack]
    frac_vals_above_cutoff =\
        np.sum(np.concatenate(values_above_cutoff, axis=0))/sum(
        [len(x) for x in values_above_cutoff])
    print("Fraction of values above cutoff:", frac_vals_above_cutoff)    
    #prepare the coordinates for the windows to be scanned
    coordinates_to_be_scanned = []
    for rowidx, above_cutoff_mask in enumerate(values_above_cutoff):
        #a mask of which positions are start positions (given
        # window length scanning_window_width)
        window_start_mask = np.zeros(len(above_cutoff_mask)
            -(scanning_window_width-1)).astype(bool)   
        colindices = np.nonzero(above_cutoff_mask)[0]
        bestslidingwindowwidths =\
            transformed_scoretrack_bestwindowwidth[rowidx][colindices]
        bestslidingwindow_startindices = (
            colindices-(((bestslidingwindowwidths-1)/2.0).astype(int)))
        for slidingwindowstart, slidingwindowwidth in zip(bestslidingwindow_startindices,
                                                          bestslidingwindowwidths):
            scanning_window_begin_startrange = max((slidingwindowstart
                - max(scanning_window_width-slidingwindowwidth,0)),0)
            scanning_window_begin_endrange = min((slidingwindowstart
                + max(slidingwindowwidth-scanning_window_width,0)),
                len(window_start_mask)-1)
            window_start_mask[scanning_window_begin_startrange:
                              scanning_window_begin_endrange] = True
        coordinates_to_be_scanned.extend([
            core.SeqletCoordinates(
                example_idx=rowidx, start=start_idx,
                end=start_idx+scanning_window_width,
                is_revcomp=is_revcomp)
            for start_idx in np.nonzero(window_start_mask)[0]
            for is_revcomp in [True, False]
        ])     
    return coordinates_to_be_scanned, transformed_scoretrack
    

def get_windows_to_be_scanned(contrib_scores, val_transformer,
                              scanning_window_width,
                              cutoff_value, plot_save_dir="."):
    print("computing the transformed score track")
    transformed_scoretrack, transformed_scoretrack_bestwindowwidth =(
        val_transformer.transform_score_track(np.sum(contrib_scores, axis=-1)))
    print("done computing the transformed score track")
    return get_windows_to_be_scanned_interior(
     transformed_scoretrack=transformed_scoretrack,
     transformed_scoretrack_bestwindowwidth=
       transformed_scoretrack_bestwindowwidth,
     val_transformer=val_transformer,
     scanning_window_width=scanning_window_width,
     cutoff_value=cutoff_value, plot_save_dir=plot_save_dir)


def collect_coordinates_by_regionidx(coordinates):
    regionidx_to_motifmatchandcoords = defaultdict(list)
    for coordinate in coordinates:
        regionidx_to_motifmatchandcoords[
            coordinate.example_idx].append(coordinate)
    return regionidx_to_motifmatchandcoords


def scan_and_process_results(instance_scorer, track_set, coordinates,
                             batch_size=None):
    scan_results = instance_scorer(coordinates, track_set=track_set,
                                   batch_size=batch_size)
    #convert scan_results into a tracks that have dimensions of
    # num_regions x num_motifs x region_len. Nan in locations that aren't hits.
    # Four such tracks for: score, prec, recall, pos_strand
    #In each case, take the max score over all scores mapping to that strand

    #for now, we are assuming windowlen is the same for all the motifs
    windowlens = (set([x.end-x.start for x in coordinates]))
    assert len(windowlens)==1
    windowlen = list(windowlens)[0]
    del windowlens

    def initialize_return_track():
        return [np.full([track_set.get_example_idx_len(i)-(windowlen-1),
                         scan_results.shape[1]-1],
                        0.0)
                for i in range(track_set.num_examples)] 

    motif_scores = initialize_return_track()
    motif_precisions = initialize_return_track()
    besthit_isrevcomp = initialize_return_track()
    for coordinate,row_of_scan_results in zip(coordinates,scan_results):
        row_of_scan_results = row_of_scan_results[:-1]
        precs = np.array([instance_scorer.get_prec_for_threshold(
                            motif_idx=motifidx,
                            threshold=row_of_scan_results[motifidx])
                          for motifidx in range(scan_results.shape[1]-1)])
        precs = np.array(precs)
        is_revcomp = coordinate.is_revcomp
        existing_scores = motif_scores[coordinate.example_idx][
                                       coordinate.start] 
        new_score_is_better = row_of_scan_results > existing_scores
        motif_scores[coordinate.example_idx][
                     coordinate.start][new_score_is_better] =(
                    row_of_scan_results[new_score_is_better])
        motif_precisions[coordinate.example_idx][
                     coordinate.start][new_score_is_better] = (
                    precs[new_score_is_better])
        besthit_isrevcomp[coordinate.example_idx][
                     coordinate.start][new_score_is_better] = is_revcomp

    ###
    #get motifmatch to coordinates
    motifmatch_to_coordinates = defaultdict(list)
    matching_motifidx = np.argmax(scan_results, axis=-1)
    for motifidx,coordinate,row_of_scan_results in zip(
            matching_motifidx,coordinates,scan_results):
        if (motifidx < scan_results.shape[-1]):
            motif_score = row_of_scan_results[motifidx]
            prec = instance_scorer.get_prec_for_threshold(
                        motif_idx=motifidx, threshold=motif_score)
            motifmatch_to_coordinates[motifidx].append(
                MotifHitAndCoord(
                    motif_idx=motifidx,
                    motif_score=motif_score,
                    precision_at_motif_score=prec,
                    example_idx=coordinate.example_idx,
                    start=coordinate.start,
                    end=coordinate.end,
                    is_revcomp=coordinate.is_revcomp)
            )
    motifmatch_to_coordinatesbyregionidx = {}
    for motifmatch in motifmatch_to_coordinates:
        motifmatch_to_coordinatesbyregionidx[motifmatch] =\
            collect_coordinates_by_regionidx(
                motifmatch_to_coordinates[motifmatch])
    
    return (motifmatch_to_coordinates,
            motifmatch_to_coordinatesbyregionidx,
            motif_scores, motif_precisions,
            besthit_isrevcomp)
