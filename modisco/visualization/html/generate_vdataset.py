from os import listdir
from os.path import isfile, join
import random
import string
from html_class import *

def get_snippet(snippet_folder):
    return VSnippet(fwd_image=snippet_folder+"/fwd.png",
                    rev_image=snippet_folder+"/rev.png") 

def get_vseqlet(seqlet_folder):
    snippet_folders=[seqlet_folder+'/'+f for f in listdir(seqlet_folder) if (not f.startswith('.'))]
    tracks=[get_snippet(f) for f in snippet_folders]
    return VSeqlet(tracks=tracks)


def get_vaggregate_seqlet(vaggregate_seqlet_folder):
    track_folders=[vaggregate_seqlet_folder+'/'+f for f in listdir(vaggregate_seqlet_folder) if (not f.startswith('.'))]
    tracks=[get_snippet(f) for f in track_folders]
    return VAggregatedSeqlet(tracks=tracks)

def get_cluster(cluster_folder):
    aggregate_motif=get_vaggregate_seqlet(cluster_folder+'/'+'aggregate_motif')
    cluster_tsne=VTsne(image=cluster_folder+'/'+"tsne_highlighting_single_cluster.png")
    example_seqlets_dir=cluster_folder+'/'+"example_seqlets"
    example_seqlet_folders = [example_seqlets_dir+'/'+f for f in listdir(example_seqlets_dir) if (not f.startswith('.'))]
    example_seqlets=[get_vseqlet(f) for f in example_seqlet_folders]
    return VCluster(tsne_embedding=cluster_tsne,
                    aggregate_motif=aggregate_motif,
                    example_seqlets=example_seqlets)



def get_metacluster(metacluster_folder):
    tsne=VTsne(image=metacluster_folder+'/'+"tsne_embedding_all_clusters.png")
    tsne_denoised=VTsne_denoised(image=metacluster_folder+'/'+"tsne_embedding_noise_filtered.png")
    cluster_dir=metacluster_folder+'/'+'per_cluster_figures'
    cluster_files = [cluster_dir+'/'+f for f in listdir(cluster_dir) if (not f.startswith('.'))]
    clusters=[get_cluster(f) for f in cluster_files]
    return VMetaCluster(tsne_embedding=tsne,
                        tsne_embedding_denoised=tsne_denoised,
                        clusters=clusters)


#generates an example VDataset object from a user-supplied directory name
def generate_vdataset_from_folder(folder_name):
    metaclusters_heatmap=VAllMetaclusterHeatmap(image=folder_name+"/metaclusters_heatmap.png")
    
    per_task_histograms_folder=folder_name+"/per_task_histograms"
    per_task_histograms_files = [(per_task_histograms_folder+'/'+f) for f in listdir(per_task_histograms_folder) if isfile(join(per_task_histograms_folder, f))]
    per_task_histograms=[VHistogram(image=f,label=''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))) for f in per_task_histograms_files]

    per_metacluster_folder=folder_name+"/per_metacluster_figures"
    per_metacluster_files=[(per_metacluster_folder+'/'+f) for f in listdir(per_metacluster_folder) if (not f.startswith('.'))]
    metaclusters=[get_metacluster(f) for f in per_metacluster_files]

    return VDataset(metaclusters_heatmap=metaclusters_heatmap,\
                    per_task_histograms=per_task_histograms,\
                    metaclusters=metaclusters,
                    title="example")

def generate_vdataset_on_the_fly():
    pass


if __name__=="__main__":
    dataset=generate_vdataset_from_folder("example_figures_modisco/")
    import pdb

