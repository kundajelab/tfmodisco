from matplotlib.widgets import RectangleSelector
from matplotlib.path import Path
import matplotlib.pyplot as plt
from . import viz_sequence
from .. import affinitymat
import sklearn.manifold
import numpy as np


def l1_norm_features(features_mat):
    return features_mat/np.sum(np.abs(features_mat), axis=1)[:,None]


def compute_pairwise_continjacc_simmat(pattern, track_names_and_signs):
    flattened_contrib_scores_vector = np.array([
        np.sum([seqlet[track_name].fwd.flatten()*sign
                for track_name,sign in track_names_and_signs], axis=0)
        for seqlet in pattern.seqlets])
    normed_flattened_contrib_scores_vector =\
        l1_norm_features(flattened_contrib_scores_vector)
    sim_mat = np.zeros((len(pattern.seqlets), len(pattern.seqlets)))
    for i in range(len(pattern.seqlets)):
        sim_mat[i] = affinitymat.core.contin_jaccard_vec_mat_sim(
            a_row=normed_flattened_contrib_scores_vector[i],
            mat=normed_flattened_contrib_scores_vector)
    return sim_mat


def get_tsne_embedding(pattern, track_names_and_signs, perplexity,
                       seed=1234):
    print("Computing pairwise similarities")
    pairwise_simmat = compute_pairwise_continjacc_simmat(
                    pattern=pattern,
                    track_names_and_signs=track_names_and_signs)
    print("Computing tsne embedding")
    tsne_embedding = (sklearn.manifold.TSNE(metric="precomputed",
                                            verbose=0,
                                            perplexity=perplexity,
                                            random_state=seed)
                      .fit_transform(1/(np.maximum(pairwise_simmat, 1e-7) )))
    #1/(pairwise_simmat) mapps the affinities to distances.
    print("Computed embedding")
    return tsne_embedding


def make_interactive_plot(pattern, track_names_and_signs,
                          figsize=(10,7), height_ratios=[2,1,1],
                          perplexity=10):

    tsne_embedding = get_tsne_embedding(
        pattern=pattern,
        track_names_and_signs=track_names_and_signs,
        perplexity=perplexity)

    fig, ax = plt.subplots(nrows=3, ncols=1,
                           gridspec_kw={'height_ratios': height_ratios},
                           figsize=figsize)

    pts = ax[0].scatter(tsne_embedding[:, 0], tsne_embedding[:, 1])
    selector = SelectFromCollection(ax[0], pts)

    def accept(event):
        selected_indices = selector.ind
        all_seqlets = pattern.seqlets
        ax[0].set_title("Number of points selected: "
                        +str(len(selected_indices)))
        ax[1].clear()
        ax[2].clear()
        mean_contrib = np.mean(np.array([
            all_seqlets[idx][track_name].fwd*sign
            for idx in selected_indices
            for (track_name, sign) in track_names_and_signs]), axis=0)
        mean_onehot = np.mean(np.array([
            all_seqlets[idx]["sequence"].fwd
            for idx in selected_indices]), axis=0)
        viz_sequence.plot_weights_given_ax(ax=ax[1], array=mean_contrib,
                                           height_padding_factor=0.2,
                                           length_padding=1.0,
                                           subticks_frequency=2, highlight={})
        viz_sequence.plot_weights_given_ax(ax=ax[2], array=mean_onehot,
                                           height_padding_factor=0.2,
                                           length_padding=1.0,
                                           subticks_frequency=2, highlight={})
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_release_event", accept)
    plt.show()


class SelectFromCollection(object):
    """Select points from a matplotlib collection using `RectangleSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = np.array(collection.get_offsets())
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        assert len(self.fc)==1
        self.orig_fc = np.array(self.fc[0])
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.selector = RectangleSelector(ax,
                         onselect=self.onselect, useblit=False)
        self.ind = []

    def onselect(self, eclick, erelease):
        lowerx = min(eclick.xdata, erelease.xdata)
        upperx = max(eclick.xdata, erelease.xdata)
        lowery = min(eclick.ydata, erelease.ydata)
        uppery = max(eclick.ydata, erelease.ydata)
        self.ind = np.nonzero((self.xys[:,0] >= lowerx)
                              *(self.xys[:,0] <= upperx)
                              *(self.xys[:,1] >= lowery)
                              *(self.xys[:,1] <= uppery))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[:, 0:3] = self.orig_fc[None,:3]
        #Red color for selection
        self.fc[self.ind, -1] = 1
        self.fc[self.ind, 0] = 1
        self.fc[self.ind, 1] = 0
        self.fc[self.ind, 2] = 0
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.selector.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
