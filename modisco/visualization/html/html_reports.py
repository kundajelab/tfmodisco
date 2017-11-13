#Generates HTML report for MODISCO analysis.
from html_class import *
from spawn_html import *

def generate_html_report(vdataset,oprefix):
        """
        Inputs: 
        vdataset is an instance of VDataset object 
        oprefix is a string indicating the output prefix for the html file to generate 
        """
    outf=open(oprefix+'.html','w')
    outf.write(generate_html_string(vdataset))
