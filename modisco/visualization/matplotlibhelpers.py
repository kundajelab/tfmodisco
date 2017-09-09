from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
import numpy as np


def scatter_plot(xycoords, clusters=None,
                 colors=None, figsize=(5,5), xlabel="", ylabel=""):
    """
        If clusters is not none, will assign colors using
            points evenly sampled from
            Blue -> Violet -> Red -> Yellow -> Green
    """
    plt.figure(figsize=figsize)
    if (clusters is None):
        plt.scatter(xycoords[:,0], xycoords[:,1])
    else:
        if (colors is None):
            max_label = np.max(clusters)
            colors = [frac_to_rainbow_colour(x/float(max_label+1))
                        if x > 0 else frac_to_rainbow_colour(0)
                        for x in range(max_label+1)]
            print("No colors supplied, so autogen'd as:\nIDX: R,G,B\n"+
                    "\n".join(str(x[0])+": "+(",".join("%0.03f"%y for y in x[1]))
                              for x in enumerate(colors)))
        plt.scatter(xycoords[:,0], xycoords[:,1],
                    c=[colors[x] for x in clusters])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def frac_to_rainbow_colour(frac):
    """
        frac is a number from 0 to 1. Map to
            a 3-tuple representing a rainbow colour.
        0 -> (0, 1, 0) #green
        1/6 -> (0, 0.5, 0.5) #cyan
        2/6 -> (0, 0, 1) #blue
        3/6 -> (0.5, 0, 0.5) #magenta 
        4/6 -> (1, 0, 0) #red
        5/6 -> (0.5, 0.5, 0) #yellow
        6/6 -> (0, 1, 0) #green again 
    """
    assert frac >= 0 and frac < 1
    interp = frac - int(frac/(1./6))*(1./6)
    if (frac < 1./6):
        #green to cyan
        to_return = (0, 1 - 0.5*interp, 0.5*interp) 
    elif (frac < 2./6):
        #cyan to blue
        to_return = (0, 0.5 - 0.5*interp, 0.5 + 0.5*interp)
    elif (frac < 3./6):
        #blue to magenta
        to_return = (0.5*interp, 0, 1 - 0.5*interp)
    elif (frac < 4./6):
        #magenta to red
        to_return = (0.5 + 0.5*interp, 0, 0.5 - 0.5*interp)
    elif (frac < 5./6):
        #red to yellow
        to_return = (1 - 0.5*interp, 0.5*interp, 0)
    else:
        #yellow to green
        to_return = (0.5 - 0.5*interp, 0.5 + 0.5*interp, 0)
    return to_return
