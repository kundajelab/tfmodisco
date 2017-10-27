import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import plotly.plotly as py


def load_image(image):
    """
    Determines whether 'image' is a string 
    or a matplotlib figure handle. 
    Loads the image accordingly
    Returns a string with a '<div>' entity for embedding into html. 
    """
    return image 



class MetaclusterHeatmap(object):
    """
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle 
    """
    def __init__(self,image):
        self.image=load_image(image) 



class VSeqlet(object):
    """
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle 

    """
    def __init__(self,image):
        self.image=load_image(image) 
    

class VTrack(object):
    """
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle 
    
    """
    def __init__(self,image):
        self.image=load_imaeg(image) 

    
class VHistogram(object):
    """
    Inputs:
    image can be either a string to a png image file 
    or a matplotlib figure handle 

    """
    def __init__(self,image):
        self.image=load_image(image) 

    

