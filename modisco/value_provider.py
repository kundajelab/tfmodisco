from __future__ import division, print_function, absolute_import
import numpy as np


class AbstractValueProvider(object):

    def __call__(self, seqlet):
        raise NotImplementedError()

    @classmethod
    def from_hdf5(cls, grp):
        the_class = eval(grp.attrs["class"])
        return the_class.from_hdf5(grp) 


class CoorScoreValueProvider(AbstractValueProvider):

    def __call__(self, seqlet):
        return seqlet.coor.score 

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__

    @classmethod
    def from_hdf5(cls, grp):
        return cls()


class TransformCentralWindowValueProvider(AbstractValueProvider):

    def __init__(self, track_name, central_window, val_transformer):
        if isinstance(track_name, str):
            self.track_name = track_name
        else: 
            self.track_name = track_name.decode('utf-8')
        self.central_window = central_window
        self.val_transformer = val_transformer

    def __call__(self, seqlet):
        val = self.get_val(seqlet=seqlet)
        return self.val_transformer(val=val)

    def get_val(self, seqlet):
        flank_to_ignore = int(0.5*(len(seqlet)-self.central_window))
        track_values = seqlet[self.track_name]\
                        .fwd[flank_to_ignore:-flank_to_ignore]
        return np.sum(track_values)

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.attrs["track_name"] = self.track_name
        grp.attrs["central_window"] = self.central_window
        self.value_transformer.save_hdf5(grp.create_group("val_transformer")) 

    @classmethod
    def from_hdf5(cls, grp):
        if isinstance(grp.attrs["track_name"], str):
            track_name = grp.attrs["track_name"]
        else:
            track_name = grp.attrs["track_name"].decode('utf-8')
        central_window = grp.attrs["central_window"] 
        val_transformer = AbstractValTransformer.from_hdf5(
                             grp["val_provider"]) 
        return cls(track_name=track_name,
                   central_window=central_window,
                   val_transformer=val_transformer)


class AbstractValTransformer(object):

    def __call__(self, val):
        raise NotImplementedError()

    @classmethod
    def from_hdf5(cls, grp):
        the_class = eval(grp.attrs["class"])
        return the_class.from_hdf5(grp) 


class PercentileValTransformer(AbstractValTransformer):

    def __init__(self, distribution):
        self.distribution = np.array(sorted(distribution))

    @classmethod
    def from_hdf5(cls, grp):
        distribution = np.array(grp["distribution"][:])
        return cls(distribution=distribution) 

    def save_hdf5(self, grp):
        grp.create_dataset("distribution", data=self.distribution)

    def __call__(self, val):
        return np.searchsorted(a=self.distribution,
                               v=val)/float(len(self.distribution))


class LaplaceCdfValTransformer(AbstractValTransformer):
                
    def __init__(self, neg_b, pos_b, mu):
        self.neg_b = neg_b
        self.pos_b = pos_b
        self.mu = mu

    def __call__(self, val):
        val -= self.mu
        if (val < 0):
            return -(1-np.exp(val/self.neg_b))
        else:
            return (1-np.exp(-val/self.pos_b))

    @classmethod
    def from_hdf5(cls, grp):
        neg_b = grp.attrs["neg_b"]
        pos_b = grp.attrs["pos_b"]
        mu = grp.attrs["mu"]
        return cls(neg_b=neg_b,
                   pos_b=pos_b,
                   mu=mu) 

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.attrs["neg_b"] = self.neg_b
        grp.attrs["pos_b"] = self.pos_b
        grp.attrs["mu"] = self.mu
