from __future__ import print_function, division, absolute_import
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import itertools


def scatter_plot(coords, clusters=None,
                 colors=None, figsize=(5,5),
                 xlabel="", ylabel="", zlabel="", **kwargs):
    """
        If clusters is not none, will assign colors using
            points evenly sampled from
            Blue -> Violet -> Red -> Yellow -> Green
    """
    assert coords.shape[1]==2 or coords.shape[1]==3
    fig = plt.figure(figsize=figsize)
    if (coords.shape[1]==2):
        ax = fig.add_subplot(111) 
    else:
        ax = fig.add_subplot(111, projection='3d')

    if (clusters is None):
        if (coords.shape[1]==2):
            ax.scatter(coords[:,0], coords[:,1], **kwargs)
        else:
            ax.scatter(coords[:,0], coords[:,1], **kwargs)
            
    else:
        if (colors is None):
            max_label = np.max(clusters)
            colors = [frac_to_rainbow_colour(x/float(max_label+1))
                        if x > 0 else frac_to_rainbow_colour(0)
                        for x in range(max_label+1)]
            print("No colors supplied, so autogen'd as:\nIDX: R,G,B\n"+
                    "\n".join(str(x[0])+": "+(",".join("%0.03f"%y for y in x[1]))
                              for x in enumerate(colors)))
        if (coords.shape[1]==2):
            ax.scatter(coords[:,0], coords[:,1],
                        c=[colors[x] for x in clusters], **kwargs)
        else:
            ax.scatter(coords[:,0], coords[:,1], coords[:,2],
                       c=[colors[x] for x in clusters], **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if (coords.shape[1]==3):
        ax.set_zlabel(zlabel)
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


def plot_heatmap(data, log_transform=False, zero_center=False,
                      cmap=plt.cm.coolwarm, figsize=(15,15)):
    fig, ax = plt.subplots(figsize=figsize)
    plot_heatmap_given_ax(ax, data , log_transform=log_transform,
                                zero_center=zero_center,
                                cmap=cmap)
    plt.show()
    return plt


def plot_heatmap_given_ax(ax, data, log_transform=False,
                          zero_center=False, cmap=plt.cm.coolwarm):
    if log_transform:
        data = np.log(np.abs(data)+1)*np.sign(data)
    if (zero_center):
        data = data*((data<0)/(1 if np.min(data)==0
                else np.abs(np.min(data))) + (data>0)/np.max(data))
    ax.pcolor(data, cmap=cmap)
    return ax


def plot_cluster_heatmap(data, clustering_func, **kwargs):
    cluster_indices=clustering_func(data) 
    data = reorganize_rows_by_clusters(data, cluster_indices)
    plot_heatmap(data=data, **kwargs) 


def reorganize_rows_by_clusters(rows, cluster_indices):
    unique_clusters = sorted(set(cluster_indices))
    cluster_idx_to_row_indices =\
        OrderedDict([(idx, []) for idx in unique_clusters])
    for row_idx, cluster_idx in zip(range(len(rows)), cluster_indices):
        cluster_idx_to_row_indices[cluster_idx].append(row_idx)
    new_indices = list(itertools.chain(*cluster_idx_to_row_indices.values()))
    return rows[new_indices]
