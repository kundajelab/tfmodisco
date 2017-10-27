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
                 metaclusters=[] 
                 
                 
    ):
        self.metaclusters_heatmap=metaclusters_heatmap
        self.per_task_histograms=per_task_histograms
        self.metaclusters=metaclusters


        
class VCluster(object):
    """
    Inputs: 
    all are optional. 

    tsne_embedding -- VTsne object 
    tsne_embedding_denoised -- VTsne_denoised object 
    aggregate_motif -- VMotif object 
    example_seqlets -- list of VSeqlet objects 

    """    
    def __init__(self,
                 tsne_embedding=None,
                 tsne_embedding_denoised=None,
                 aggregate_motif=None,
                 example_seqlets=[],
                 
    ):

class VMetaCluster(VCluster):
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
                 clusters=[])
    VCluster.__init__(tsne_embedding,
                      tsne_embedding_denoised)
    self.clusters=clusters

    
class VAllMetaclusterHeatmap(object):
    """

    Histogram of all metaclusters in the dataset. 
    
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle 
    
    cluster_id_to_mean: dictionary mapping a cluster id to the cluster mean 
    cluster_id_to_num_seqlets_in_cluster: a dictionary mapping a cluster id to the number of seqlets in the cluster 
    """
    def __init__(self,image,cluster_id_to_mean,cluster_id_to_num_seqlets_in_cluster):
        self.image=load_image(image)
        self.cluster_id_to_mean=cluster_id_to_mean
        self.cluster_id_to_num_seqlets_in_cluster=cluster_id_to_num_seqlets_in_cluster
        

        

class VMotif(object):
    """
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle 
    
    motif_tracks -- list of VTrack objects corresponding to the motif 
    motif_seqlets -- list of VSeqlet objects corresponding to the motif 

    """
    def __init__(self,motif_tracks,motif_seqlets):
        self.motif_tracks=motif_tracks
        self.motif_seqlets=motif_seqlets
        
                 
class VSeqlet(object):
    """
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle 

    tracks: list of VTrack objects 

    """
    def __init__(self,image,tracks,coordinates):
        self.image=load_image(image) 
        self.tracks=tracks

class VTrack(object):
    """
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle 
    
    """
    def __init__(self,track_name,fwd_image,rev_image):
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
    def __init__(self,image, thresh,num_above_thresh):
        self.image=load_image(image) 
        self.thresh=thresh
        self.number_above_thresh=number_above_thresh
        
    

class VTsne(object):
    """
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle 
    
    cluster_id_to_color_triplet is a dictionary 

    """
    def __init__(self,image,cluster_id_to_color_triplet):
        self.image=load_image(image)
        self.cluster_id_to_color_triplet=cluster_id_to_color_triplet
        

class VTsne_denoised(VTsne):
    def __init__(self,image,cluster_id_to_color_triplet,num_pre_filtered,num_post_filtered):
        VTsne.__init__(self,image,cluster_id_to_color_triplet)
        self.num_pre_filtered=num_pre_filtered
        self.num_post_filtered=num_post_filtered
        
