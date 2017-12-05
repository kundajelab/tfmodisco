#Generates HTML report for MODISCO analysis.
from html_class import *
from spawn_html import *
from shutil import copyfile

def generate_html_report(vdataset,oprefix):
        """
        Inputs: 
        vdataset is an instance of VDataset object 
        oprefix is a string indicating the output prefix for the html file to generate 
        css_source is the filename of the modisco css sheet. By default it is modisco.css
        """
        outf=open(oprefix+'.html','w')
        outf.write(generate_html_string(vdataset))
        outf.close()
        
        #Copy the modisco css stylesheet to the output directory
        '''
        css_dest='modisco.css'
        if '/' in oprefix:
                css_dest='/'.join(vdataset.split('/')[0:-1])+'/'+css_source
        print("source:"+str(css_source))
        print("dest:"+str(css_dest))
        copyfile(css_source,css_dest)
        '''
