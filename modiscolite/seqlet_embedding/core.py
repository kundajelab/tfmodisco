from __future__ import division, print_function


class AbstractSeqletsToOnedEmbedderFactory(object):

    #Return an instance of AbstractSeqletsToOnedEmbedder
    def __call__(onehot_track_name, toscore_track_names_and_signs):
        raise NotImplementedError()


class AbstractSeqletsToOnedEmbedder(object):

    def __call__(seqlets):
        raise NotImplementedError()
