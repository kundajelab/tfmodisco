import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import init_notebook_mode, plot_mpl
import base64
from html_classes import *
import sys
sys.path.append("../../")
from core import *
sys.path.append("../")
from viz_sequence import *
from matplotlibhelpers import * 

def load_image(image):
        """
    Determines whether 'image' is a string 
    or a matplotlib figure handle. 
    Loads the image accordingly
    Returns a string with a '<div>' entity for embedding into html. 
    """
    if type(image)==plt.Figure:
        image_div=plot_mpl(image,output_type='div')
    elif type(image)==str:
        encoded = base64.b64encode(open(image, "rb").read())
        image_tag='<img src="data:image/png;base64,'+encoded+'">'
        image_div='<div>'+image_tab+'</div>'
    else:
        raise(Exception("Invalid file format for image, must be either str or plt.Figure"))
    return image_div

#Converter methods for MODISCO classes
def convert_Snippet_to_VSnippet(snippet_instance,track_name=None):
        '''
        Converts and instance of the Snippet class into an instance of the VSnippet class for 
        HTML visualization 
        
        Optionally, provide a name for the track 
        '''
        #generate images
        fwd_image=plot_weights(snippet_instance.fwd)
        rev_image=plot_weights(snippet_instance.rev) 
        return VSnippet(track_name=track_name,
                        fwd_image=fwd_image,
                        rev_image=rev_image)

def create_VSnippet_list(pattern_instance):
        '''
        Creates a list of VSnippet objects for inclusion in tracks attribute of VPattern 
        and child classes of VPattern
        '''
        vsnippet_tracks=[] 
        for track_name in pattern_instance.tracks:
                cur_snippet=pattern_instance.tracks[track_name]
                cur_vsnippet=convert_Snippet_to_VSnippet(cur_snippet,track_name=track_name)
                vsnippet_tracks.append(cur_vsnippet)
        return vsnippet_tracks 
        
def convert_Pattern_to_VPattern(pattern_instance):
        '''
        Conversts an instance of the Pattern class to an instance of the VPattern class. 
        '''
        #create the VSnippet objects that compose the tracks
        vsnippet_tracks=create_vsnippet_list(pattern_instance)
        return VPattern(original_pattern=pattern_instance,
                        tracks=vsnippet_tracks)

def convert_Seqlet_to_VSeqlet(seqlet_instance):
        '''
        Converts an instance of the Seqlet class to an instance of the VSeqlet class. 
        '''
        #create the VSnippet objects that compose the tracks
        vsnippet_tracks=create_vsnippet_list(seqlet_instance)
        return VSeqlet(original_pattern=pattern_instance,
                        tracks=vsnippet_tracks)
        

def convert_AggregatedSeqlet_to_VAggregatedSeqlet(aggregated_seqlet_instance):
        '''
        Converts an instance of the AggregateSeqlet class to an instance of the VAggregatedSeqlet class. 
        '''
        pass

        
