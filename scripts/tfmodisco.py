from __future__ import division, print_function, absolute_import
import modisco
import modisco.core
import argparse


def get_seqlets(task_names, contrib_scores, hypothetical_contribs,
                one_hot, coord_producer, overlap_resolver):

    contrib_scores_tracks = [
        modisco.core.DataTrack(
            name=key+"_contrib_scores",
            fwd_tracks=contrib_scores[key],
            rev_tracks=contrib_scores[key][:,::-1,::-1],
            has_pos_axis=True) for key in task_names] 

    hypothetical_contribs_tracks = [
        modisco.core.DataTrack(name=key+"_hypothetical_contribs",
                       fwd_tracks=hypothetical_contribs[key],
                       rev_tracks=hypothetical_contribs[key][:,::-1,::-1],
                       has_pos_axis=True)
                       for key in task_names]

    onehot_track = modisco.core.DataTrack(name="sequence", fwd_tracks=onehot,
                               rev_tracks=onehot[:,::-1,::-1],
                               has_pos_axis=True)

    track_set = modisco.core.TrackSet(data_tracks=
                                contrib_scores_tracks+hypothetical_contribs_tracks
                                +[onehot_track])

    per_position_contrib_scores = dict([
        (x, np.sum(contrib_scores[x],axis=2)) for x in task_names])

    task_name_to_labeler = dict([
        (task_name, modisco.core.SignedContribThresholdLabeler(
            flank_to_ignore=flank,
            name=task_name+"_label",
            track_name=task_name+"_contrib_scores"))
         for task_name in task_names]) 

    seqlets = modisco.core.MultiTaskSeqletCreation(
        coord_producer=coord_producer,
        track_set=track_set,
        overlap_resolver=overlap_resolver)(
            task_name_to_score_track=per_position_contrib_scores,
            task_name_to_labeler=task_name_to_labeler)

    return seqlets



if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--onehot_hdf5", required=True,
                        help="Path to .hdf5 with one-hot encoded seq data."
                             +" The data should be stored under a dataset"
                             +" named 'onehot'")
    parser.add_argument("--hypothetical_contribs_hdf5", required=True,
                        help="Path to the .hdf5 with the hypothetical"
                             " contribs. The dataset names should correspond"
                             " to the different tasks to analyze")
    parser.add_argument("--clustering_config", required=True,
                        help="Path to file with clustering config") 
