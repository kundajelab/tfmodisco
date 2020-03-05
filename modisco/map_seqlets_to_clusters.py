from __future__ import division, print_function
import numpy as np
from . import util


def get_clusters_from_seqlettoclusterscore(seqlet_to_cluster_score):
    clusters = -1*np.ones(len(seqlet_to_cluster_score))
    best_cluster = np.argmax(seqlet_to_cluster_score, axis=-1)
    cluster_score = seqlet_to_cluster_score[(np.arange(len(best_cluster)),
                                             best_cluster)]
    cluster_assigned_mask = (cluster_score > -1)
    clusters[cluster_assigned_mask] = best_cluster[cluster_assigned_mask]
    return clusters


#define API
class MapSeqletsToClusters(object):

    def __call__(seqlets):
        raise NotImplementedError()


class ExemplarBasedSeqletToClusterMapper(object):

    def __init__(self, cluster_to_exemplars, cluster_to_minthresh):
        self.num_clusters = max(cluster_to_exemplars.keys())+1
        self.cluster_to_exemplars = cluster_to_exemplars
        self.cluster_to_minthresh = cluster_to_minthresh

    @classmethod
    def build(cls, cluster_to_exemplars,
              cluster_to_exemplar_sims,
              orig_cluster_membership,
              fprthresh, tprthresh, precthresh,
              clustersizefoldincreasethresh):
        cluster_to_minthresh = {}
        for idx in cluster_to_exemplars:
            within_cluster_mask = orig_cluster_membership==idx
            exemplar_sims, _, _ = cluster_to_exemplar_sims[idx]
            scores = np.median(exemplar_sims, axis=0)

            scores_tprthresh =\
                np.percentile(a=scores[within_cluster_mask],
                              q=(1-tprthresh)*100)
            scores_fprthresh =\
                max(np.percentile(
                      a=scores[within_cluster_mask==False],
                      q=(1-fprthresh)*100),
                    np.min(scores[within_cluster_mask]))
            scores_precthresh =\
                max(util.get_precision_threshold(
                      y_true=within_cluster_mask,
                      y_pred=scores,
                      precision_threshold=precthresh),
                    np.min(scores[within_cluster_mask]))
            scores_clustersizefoldincreasethresh =\
                np.percentile(a=scores,
                              q=max((1-((np.sum(within_cluster_mask)
                                        *clustersizefoldincreasethresh)/
                                        len(scores)))*100,0))
            ##take the minimum of the 3 thresholds, but make sure
            # cluster size does not explode
            finalthresh = max(min(scores_tprthresh,
                                  scores_fprthresh,
                                  scores_precthresh),
                              scores_clustersizefoldincreasethresh)
            cluster_to_minthresh[idx] = finalthresh
        return cls(cluster_to_exemplars=cluster_to_exemplars,
                            cluster_to_minthresh=cluster_to_minthresh) 

    def map_exemplar_sims_to_cluster(self, cluster_to_exemplar_sims):
        num_seqlets = cluster_to_exemplar_sims[
                       list(cluster_to_exemplar_sims.keys())[0]][0].shape[1]
        seqlet_to_cluster_score = np.ones((num_seqlets, self.num_clusters))*-1
        for idx in cluster_to_exemplar_sims:
            exemplar_sims, _, _ = cluster_to_exemplar_sims[idx]
            scores = np.median(exemplar_sims, axis=0)
            minthresh = self.cluster_to_minthresh[idx]
            passing_thresh = scores > minthresh
            seqlet_to_cluster_score[passing_thresh, idx] =\
                scores[passing_thresh] 
        return seqlet_to_cluster_score
