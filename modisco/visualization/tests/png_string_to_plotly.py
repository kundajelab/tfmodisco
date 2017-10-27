import base64
encoded = base64.b64encode(open("../example_figures_modisco/metaclusters_heatmap.png", "rb").read())
image_tag='<img src="data:image/png;base64,'+encoded+'">'
from yattag import Doc
doc,tag,text=Doc().tagtext()
doc.asis('<!DOCTYPE html>')
with tag('html'):
    with tag('body'):
        doc.asis(image_tag)
html_string=doc.getvalue()
outf=open('example.from.string.html','w')
outf.write(html_string)

