from yattag import *
from encode_js_functions import *

def add_metacluster(doc,tag,metacluster):
    pass


def add_per_task_histograms(doc,tag,vdataset):
    with tag('ul'):                    
        for histogram_index in range(len(vdataset.per_task_histograms)):
            with tag('li'):
                cur_hist=vdataset.per_task_histograms[histogram_index]
                checkbox_label='check{histogram_index}'.format(histogram_index=histogram_index)
                with tag('label',for=checkbox_label):
                    tag('input',
                        type='checkbox',
                        id=checkbox_label,
                        onchange='showHist(\'{histogram_index}\')'.format(histogram_index=histogram_index)
                    )
                    with tag('span'):
                        text(cur_hist.label)
    with tag('div',class='column'):
        for histogram_index in range(len(vdataset.per_task_histograms)):
            cur_hist=vdataset.per_task_histograms[histogram_index]
            hist_label='hist{histogram_index}'.format(histogram_index=histogram_index)
            with tag('div',klass='histogram',id=hist_label):
                modal_image_function_calls.append(add_modal_image(doc,tag,cur_hist.image,id=len(modal_image_function_calls)))
                with tag('p'):
                    text(cur_hist.label)


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
    with tag('scrpit'):
        doc.asis(selectClusterTab())
    

def add_modal_image(doc,tag,image,id):
    '''
    adds an image div and an associated modal div to the doc object 
    '''
    #get unique id's for the modal elements 
    modalId="modal{id}".format(id=id)
    imageId="image{id}".format(id=id)
    modalImageId="modalImage{id}".format(id=id)
    captionId="caption{id}".format(id=id)

    #generate the modal elements associated with the image
    #add the div tag for the image 
    with doc.tag('div',id=imageId):
        text(image)
    with doc.tag('div',id=modalId,klass="modal"):
        with doc.tag('span',klass="close"):
            doc.asis('&times;')
        doc.stag('img',klass='modal-content',id=modalImageId)
        doc.stag('div',id=captionId)

    #generate the modal function call for the image & return the associated string
    return callModalFunction(modalId,imageId,modalImageId,captionId)

def generate_html_string(vdataset):
    '''
    generates main body of the html document.
    '''
    #store calls to modalFunction; these will be added at end of HTML doc in a script. 
    modal_image_function_calls=[];
    doc,tag,text=Doc().tagtext()
    doc.asis('<!DOCTYPE html>')
    doc.stag('html',lang='en')
    doc.stag('meta',content='text/html;charset=utf-8',http_equiv='Content-Type')
    doc.stag('meta',content='utf-8',http_equiv="encoding")    
    with tag('head'):
        with tag('title'):
            text(vdataset.title)
        doc.tag('link',type='text/css',rel='stylesheet',href='modisco.css')
        with tag('script'):
            text(showHist())
    with tag('body'):
        with tag('div',klass='horizontal_panel'):
            with tag('div',klass='column'):
                with tag('h2'):
                    text('All Metaclusters Heatmap')
                    
                #generate a modal image for the all metacluster heatmap.
                #add the modal function call to the list of all modal image function calls in the HTML doc 
                modal_image_function_calls.append(add_modal_image(doc,tag,vdataset.metaclusters_heatmap.image,id=len(modal_image_function_calls)))
            with tag('div',klass='column'):
                with tag('h2'):
                    text("Task-specific histograms")
                #add the per-task histograms
                add_per_task_histograms(doc,tag,vdataset)            
                        
        for metacluster in vdataset.metaclusters:
            add_metacluster(doc,tag,metacluster) 
        #add the scripts
        add_sripts(doc,tag,modal_image_function_calls)
            
    #indent the code 
    #return indent(doc.getvalue())
    return doc.getvalue()
    
if __name__=="__main__":
    import sys
    sys.path.append('tests')
    from generate_vdataset import *
    vdataset=generate_vdataset_from_folder("/home/annashch/modisco_private/modisco/visualization/html/example_figures_modisco/")
    test_out=generate_html_string(vdataset)
    outf=open('test_small.html','w')
    outf.write(test_out)
    
