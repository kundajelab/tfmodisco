from modisco.sliding_similarities import parallel_sliding_continousjaccard


# Old API
def score_trackset_with_pattern(track_set, pattern,
                                track_names, track_transformer, n_cores):
    pass


def parallelcpu_score_full_regionarrs_with_perpos_continjaccard(regionarrs_to_scan, arr_to_scan_with, n_cores):
    return parallel_sliding_continousjaccard(arr_to_scan_with, regionarrs_to_scan, n_cores)
