from yattag import *
from encode_js_functions import *

def add_snippet(doc,tag,text,cur_snippet,snippet_index,modal_image_function_calls):
    with tag('h4'):
        if cur_snippet.track_name!=None:
            text("Track {snippet_name}".format(snippet_name=cur_snippet.track_name))
        else:
            text("Track {snippet_index}".format(snippet_index=snippet_index))
        with tag('h5'):
            text("Forward")
        modal_image_function_calls.append(add_modal_image(doc,tag,text,cur_snippet.fwd_image,len(modal_image_function_calls)))
        with tag('h5'):
            text("Reverse")
        modal_image_function_calls.append(add_modal_image(doc,tag,text,cur_snippet.rev_image,len(modal_image_function_calls)))
    return modal_image_function_calls

def add_seqlet(doc,tag,text,cur_seqlet,seqlet_index,modal_image_function_calls):
    #color-code odd & even seqlets to make them easier to distinguish visually
    if seqlet_index%2==0:
        cur_class="blue"
    else:
        cur_class="gray"
    with tag('div',klass=cur_class):
        with tag('h3'):
            text("Seqlet_{seqlet_index}".format(seqlet_index=seqlet_index))
        for snippet_index in range(len(cur_seqlet.tracks)):
            cur_snippet=cur_seqlet.tracks[snippet_index]
            modal_image_function_calls=add_snippet(doc,tag,text,cur_snippet,snippet_index,modal_image_function_calls)        
    return modal_image_function_calls

def add_aggregate_motif(doc,tag,text,aggregate_motif,modal_image_function_calls):
    with tag('h3'):
        text("Aggregate Motif")
    #iterate through the VSnippet objects that form the tracks
    for track_index in range(len(aggregate_motif.tracks)):
        cur_track=aggregate_motif.tracks[track_index]
        with tag('h4'):
            if cur_snippet.track_name!=None:
                text("Track {track_name}".format(track_name=aggregate_motif.track_name))
            else:
                text("Track {track_index}".format(track_index=track_index))
        with tag('h5'):
            text("Forward:")
        modal_image_function_calls.append(add_modal_image(doc,tag,text,cur_track.fwd_image,len(modal_image_function_calls)))
        with tag('h5'):
            text("Reverse:")
        modal_image_function_calls.append(add_modal_image(doc,tag,text,cur_track.rev_image,len(modal_image_function_calls)))
    return modal_image_function_calls

def add_cluster(doc,tag,text,cur_cluster,cluster_index,modal_image_function_calls):
    with tag('div',id='cluster_{cluster_index}'.format(cluster_index=cluster_index),
             klass="tabcontent"):
        modal_image_function_calls=add_aggregate_motif(doc,tag,text,cur_cluster.aggregate_motif,modal_image_function_calls)
        with tag('div',klass='column'):
            with tag('h3'):
                text("TSNE highlighting current cluster")
        modal_image_function_calls.append(add_modal_image(doc,tag,text,cur_cluster.tsne_embedding.image,cur_id=len(modal_image_function_calls)))

        with tag('button',klass='accordion'):
            text('Sample Seqlets')
        with tag('div',klass='panel'):
            for seqlet_index in range(len(cur_cluster.example_seqlets)):
                cur_seqlet=cur_cluster.example_seqlets[seqlet_index]
                modal_image_function_calls=add_seqlet(doc,tag,text,cur_seqlet,seqlet_index,modal_image_function_calls)
    return modal_image_function_calls 
        
def add_metacluster(doc,tag,text,cur_metacluster,metacluster_index,modal_image_function_calls):
    with tag('button',klass='accordion'):
        text("Metacluster {metacluster_index}".format(metacluster_index=metacluster_index))
    with tag('div',klass='panel'):
        with tag('div',klass='tsne'):
            #add a modal image for TSNE
            modal_image_function_calls.append(add_modal_image(doc,tag,text,cur_metacluster.tsne_embedding.image,cur_id=len(modal_image_function_calls)))
            with tag('p'):
                text('TSNE Embedding')
            #add a modal image for denoised TSNE
            modal_image_function_calls.append(add_modal_image(doc,tag,text,cur_metacluster.tsne_embedding_denoised.image,cur_id=len(modal_image_function_calls)))
            with tag('p'):
                text('Noise-Filtered TSNE Embedding')
                
        #add cluster tabs for the current metacluster 
        with tag('div',klass='tab'):
            for cluster_index in range(len(cur_metacluster.clusters)):
                unique_cluster_index=hash(str(metacluster_index)+str(cluster_index))
                with tag('button',klass='tablinks',onclick="openCluster(event,'cluster_{i}')".format(i=unique_cluster_index)):
                    text("cluster_{cluster_index}".format(cluster_index=cluster_index))
                    
        #add a div containing cluster-specific information for each cluster within the metacluster 
        for cluster_index in range(len(cur_metacluster.clusters)):
            cur_cluster=cur_metacluster.clusters[cluster_index]
            unique_cluster_index=hash(str(metacluster_index)+str(cluster_index))
            modal_image_function_calls=add_cluster(doc,tag,text,cur_cluster,unique_cluster_index,modal_image_function_calls)      
    return modal_image_function_calls 

def add_per_task_histograms(doc,tag,text,vdataset,modal_image_function_calls):
    with tag('ul'):                    
        for histogram_index in range(len(vdataset.per_task_histograms)):
            with tag('li'):
                cur_hist=vdataset.per_task_histograms[histogram_index]
                checkbox_label='check{histogram_index}'.format(histogram_index=histogram_index)
                with tag('label'):
                    with tag('input',
                             type='checkbox',
                             id=checkbox_label,
                             onchange='showHist("hist{histogram_index}")'.format(histogram_index=histogram_index)
                    ):
                        pass                    
                    with tag('span'):
                        text(cur_hist.label)
    with tag('div',klass='column'):
        for histogram_index in range(len(vdataset.per_task_histograms)):
            cur_hist=vdataset.per_task_histograms[histogram_index]
            hist_label='hist{histogram_index}'.format(histogram_index=histogram_index)
            with tag('div',klass='histogram',id=hist_label):
                modal_image_function_calls.append(add_modal_image(doc,tag,text,cur_hist.image,cur_id=len(modal_image_function_calls)))
                with tag('p'):
                    text(cur_hist.label)
    return modal_image_function_calls


def add_scripts(doc,tag,modal_image_function_calls):
    #accordion 
    with tag('script'):
        doc.asis(accordion())

    #modal functions
    with tag('script'):
        #define the modal function 
        doc.asis(def_modalFunction())
        #call the modal function on all the images in the HTML doc
        for modal_image_function_call in modal_image_function_calls:
            doc.asis(modal_image_function_call)
            
    #uncheck all checkboxes
    with tag('script'):
        doc.asis(uncheckAll())

    #select cluster tab
    with tag('script'):
        doc.asis(selectClusterTab())
    

def add_modal_image(doc,tag,text,image,cur_id):
    '''
    adds an image div and an associated modal div to the doc object 
    '''
    #get unique cur_id's for the modal elements 
    modalId="modal{cur_id}".format(cur_id=cur_id)
    imageId="image{cur_id}".format(cur_id=cur_id)
    modalImageId="modalImage{cur_id}".format(cur_id=cur_id)
    captionId="caption{cur_id}".format(cur_id=cur_id)
    closeId="close{cur_id}".format(cur_id=cur_id)
    #generate the modal elements associated with the image
    #add the div tag for the image 
    with tag('div',id=imageId):
        doc.asis(image)
    with tag('div',id=modalId,klass="modal"):
        with tag('span',id=closeId,klass="close"):
            doc.asis('&times;')
        with tag('img',klass='modal-content',id=modalImageId):
            pass
        with tag('div',id=captionId):
            pass

    #generate the modal function call for the image & return the associated string
    return call_modalFunction(modalId,imageId,modalImageId,captionId,closeId)

def generate_html_string(vdataset):
    '''
    generates main body of the html document.
    '''
    #store calls to modalFunction; these will be added at end of HTML doc in a script. 
    modal_image_function_calls=[];
    doc,tag,text=Doc().tagtext()
    doc.asis('<!DOCTYPE html>')
    doc.asis('<html lang=\'en\'>')
    doc.asis('<meta content="text/html;charset=utf-8" http-equiv="Content-Type">')
    doc.asis('<meta content="utf-8" http-equiv="encoding">')
    with tag('head'):
        with tag('title'):
            text(vdataset.title)
        doc.asis('<link type=\'text/css\' rel=\'stylesheet\' href="modisco.css">')
        with tag('script'):
            text(showHist())
    with tag('body'):
        with tag('div',klass='horizontal_panel'):
            with tag('div',klass='column'):
                with tag('h2'):
                    text('All Metaclusters Heatmap')                    
                #generate a modal image for the all metacluster heatmap.
                #add the modal function call to the list of all modal image function calls in the HTML doc 
                modal_image_function_calls.append(add_modal_image(doc,tag,text,vdataset.metaclusters_heatmap.image,cur_id=len(modal_image_function_calls)))
            with tag('div',klass='column'):
                with tag('h2'):
                    text("Task-specific histograms")
                #add the per-task histograms
                modal_image_function_calls=add_per_task_histograms(doc,tag,text,vdataset,modal_image_function_calls)
                
        #add each of the metaclusters                 
        for metacluster_index  in range(len(vdataset.metaclusters)):
            cur_metacluster=vdataset.metaclusters[metacluster_index]
            modal_image_function_calls=add_metacluster(doc,tag,text,cur_metacluster,metacluster_index,modal_image_function_calls)
            
        #add the scripts
        add_scripts(doc,tag,modal_image_function_calls)

    #get string from doc
    return doc.getvalue()
    
if __name__=="__main__":
    from generate_vdataset import *    
    vdataset=generate_vdataset_from_folder("example_figures_modisco/")
    test_out=generate_html_string(vdataset)
    outf=open('test_small.html','w')
    outf.write(test_out)
