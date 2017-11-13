import matplotlib.pyplot as plt
import numpy as np
from plotly.offline import init_notebook_mode, plot_mpl
import base64

def load_image(image,height=450,width=450):
    """
    Determines whether 'image' is a string 
    or a matplotlib figure handle. 
    Loads the image accordingly
    Returns a string with a '<div>' entity for embedding into html. 
    """
    if type(image)==plt.Figure:
        image_div=plot_mpl(image,output_type='div',height=height,width=width)
    elif type(image)==str:
        encoded = base64.b64encode(open(image, "rb").read())
        image_tag='<img src="data:image/png;base64,'+encoded+'",height='+str(height)+',width='+str(width)+'>'
        image_div='<div>'+image_tag+'</div>'
    else:
        raise(Exception("Invalid file format for image, must be either str or plt.Figure"))
    return image_div
