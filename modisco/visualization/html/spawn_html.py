from yattag import *
from encode_js_functions import *

def generate_html_string(vdataset):
    #generates main body of the html document.
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
                    
            
    
    #indent the code 
    return indent(doc.getvalue())

    
if __name__=="__main__":
    test_out=generate_html_doc("test",None)
    outf=open('test_small.html','w')
    outf.write(test_out)
    
