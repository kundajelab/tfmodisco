#demo code for converting a matplotlib fig to a plotly div for sticking into an html file 

from plotly.offline import init_notebook_mode, plot_mpl
import matplotlib.pyplot as plt

init_notebook_mode()

fig = plt.figure()
x = [10, 15, 20, 25, 30]
y = [100, 250, 200, 150, 300]
plt.plot(x, y, "o")



image_div=plot_mpl(fig, output_type='div')

#create a dummy file with yattag library
from yattag import Doc
doc,tag,text=Doc().tagtext()
doc.asis('<!DOCTYPE html>')
with tag('html'):
    with tag('body'):
        doc.asis(image_div)
html_string=doc.getvalue()
outf=open('example.html','w')
outf.write(html_string)


