from html_helpers import *


class VDataset(object):
    """
    Inputs: 
    all are optional. 
    metaclusters_heatmap -- object of type VAllMetaclusterHeatmap
    per_task_histograms -- list of VHistogram objects 
    metaclusters -- list of VMetaCluster objects 
    """

    def __init__(self,
                 metaclusters_heatmap=None,
                 per_task_histograms=[],
                 metaclusters=[],
                 title=None
                 
                 
    ):
        self.metaclusters_heatmap=metaclusters_heatmap
        self.per_task_histograms=per_task_histograms
        self.metaclusters=metaclusters
        self.title=title
        
        
class VCluster(object):
    """
    Inputs: 
    all are optional. 
    tsne_embedding -- VTsne object 
    tsne_embedding_denoised -- VTsne_denoised
 object 
    aggregate_motif -- VPattern object 
    example_seqlets -- list of VSeqlet objects 
    """
    def __init__(self,
                 tsne_embedding=None,
                 aggregate_motif=None,
                 example_seqlets=[],
                 
    ):
        self.tsne_embedding=tsne_embedding
        self.aggregate_motif=aggregate_motif
        self.example_seqlets=example_seqlets 
        
class VMetaCluster(object):
    """
    Inputs:
    all are optional. 
    tsne_embedding -- VTsne object 
    tsne_embedding_denoised -- VTsne_denoised object 
    clusters -- list of VCluster objects 
    """

    def __init__(self,
                 tsne_embedding=None,
                 tsne_embedding_denoised=None,
                 clusters=[]):
        self.clusters=clusters
        self.tsne_embedding=tsne_embedding 
        self.tsne_embedding_denoised=tsne_embedding_denoised


class VAllMetaclusterHeatmap(object):
    """
    Heatmap of all metaclusters in the dataset. 
    
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle 
    
    cluster_id_to_mean: dictionary mapping a cluster id to the cluster mean 
    cluster_id_to_num_seqlets_in_cluster: a dictionary mapping a cluster id to the number of seqlets in the cluster 
    """
    def __init__(self,image=None,
                 cluster_id_to_mean={},
                 cluster_id_to_num_seqlets_in_cluster={}):
        self.image=load_image(image)
        self.cluster_id_to_mean=cluster_id_to_mean
        self.cluster_id_to_num_seqlets_in_cluster=cluster_id_to_num_seqlets_in_cluster        
        
class VPattern(object):
    """
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle 
    
    tracks -- list of VSnippet objects corresponding to the motif 
    """
    def __init__(self,original_pattern=None,
                 tracks=[]):
        self.original_pattern=original_pattern 
        self.tracks=tracks

        
class VAggregatedSeqlet(VPattern):
    """
    seqlets is a list of VSeqlet objects -- corresponds to an aggregate motif 
    """
    def __init__(self,original_aggregated_seqlet=None,
                 tracks=[]):
        super(VAggregatedSeqlet,self).__init__(original_pattern=original_aggregated_seqlet,tracks=tracks)
         
class VSeqlet(VPattern):
    """
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle 
    tracks: list of VSnippet objects 
    """
    def __init__(self,
                 original_seqlet=None,
                 tracks=[],
                 coordinates=None):
        super(VSeqlet,self).__init__(tracks)
        self.coordinates=coordinates
        self.original_seqlet=original_seqlet 
        
class VSnippet(object):
    """
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle     
    """
    
    def __init__(self,track_name=None,
                 fwd_image=None,
                 rev_image=None):
        self.track_name=track_name
        self.fwd_image=load_image(fwd_image)
        self.rev_image=load_image(rev_image)

    
class VHistogram(object):
    """
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle 
    
    thresh: threshold (double) 
    num_above_thresh: number motifs passing threshold (integer) 
    """
    
    def __init__(self,image=None,
                 thresh=None,
                 num_above_thresh=None,
                 label=None):
        self.image=load_image(image)
        self.thresh=thresh
        self.number_above_thresh=num_above_thresh
        self.label=label

        

class VTsne(object):
    """
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle     
    cluster_id_to_color_triplet is a dictionary 
    """
    def __init__(self,image=None,
                 cluster_id_to_color_triplet={}):
        self.image=load_image(image)
        self.cluster_id_to_color_triplet=cluster_id_to_color_triplet

        
class VTsne_denoised(VTsne):
    def __init__(self,image=None,
                 cluster_id_to_color_triplet={},
                 num_pre_filtered=0,
                 num_post_filtered=0):
        super(VTsne_denoised,self).__init__(image,
                                            cluster_id_to_color_triplet)
        self.num_pre_filtered=num_pre_filtered
        self.num_post_filtered=num_post_filtered
    
