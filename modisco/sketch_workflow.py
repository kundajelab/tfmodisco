
score_track = core.DataTrack(name=..., fwd_tracks=..., 
                             rev_tracks=..., has_pos_axis=...)
track_set = core.TrackSet(data_tracks=[score_track, ...])

seqlets = TrackSet.create_seqlets(
            coordproducers.FixedWindowAroundPeaks(...)
                           .get_coords(score_track))

clusters = cluster.core.PhenographCluster().cluster(
            affinitymat.MaxCrossCorrAffinityMatrixFromSeqlets(...)
                       .get_affinity_matrix(seqlets))
