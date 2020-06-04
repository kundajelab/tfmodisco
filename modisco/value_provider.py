from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.stats


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
                        .fwd[flank_to_ignore:(len(seqlet)-flank_to_ignore)]
        return np.sum(track_values)

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.attrs["track_name"] = self.track_name
        grp.attrs["central_window"] = self.central_window
        self.val_transformer.save_hdf5(grp.create_group("val_transformer")) 

    @classmethod
    def from_hdf5(cls, grp):
        if isinstance(grp.attrs["track_name"], str):
            track_name = grp.attrs["track_name"]
        else:
            track_name = grp.attrs["track_name"].decode('utf-8')
        central_window = grp.attrs["central_window"] 
        val_transformer = AbstractValTransformer.from_hdf5(
                             grp["val_transformer"]) 
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


#2x Cdf - 0.5
class Gamma2xCdfMHalfValTransformer(AbstractValTransformer):

    def __init__(self, a):
        self.a = a

    @classmethod
    def from_hdf5(cls, grp):
        a = grp.attrs["a"]
        return cls(a=a) 

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.attrs["a"] = self.a

    def __call__(self, val):
        #I'm doing it this way to maintain continuity in the scores
        signed_onempval = (scipy.stats.gamma.cdf(x=val, a=self.a)-0.5)*2 
        return signed_onempval


class AbsPercentileValTransformer(AbstractValTransformer):

    def __init__(self, distribution):
        self.distribution = np.array(sorted(np.abs(distribution)))

    @classmethod
    def from_hdf5(cls, grp):
        distribution = np.array(grp["distribution"][:])
        return cls(distribution=distribution) 

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.create_dataset("distribution", data=self.distribution)

    def __call__(self, val):
        return np.sign(val)*np.searchsorted(
                 a=self.distribution,
                 v=abs(val))/float(len(self.distribution))


class SignedPercentileValTransformer(AbstractValTransformer):

    def __init__(self, distribution):
        self.distribution = np.array(distribution)
        self.pos_dist = np.array(sorted(
            [x for x in self.distribution if x > 0]))
        self.abs_neg_dist = np.array(sorted(
            [abs(x) for x in self.distribution if x < 0]))

    @classmethod
    def from_hdf5(cls, grp):
        distribution = np.array(grp["distribution"][:])
        return cls(distribution=distribution) 

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.create_dataset("distribution", data=self.distribution)

    def __call__(self, val):
        if (val == 0):
            return 0
        elif (val > 0):
            #add 1E-7 for complicated numerical stability issues 
            # basically need robustness when dealing with ties
            return  np.searchsorted(
                     a=self.pos_dist, v=(val+1E-7))/float(len(self.pos_dist))
        else:
            #add 1E-7 for complicated numerical stability issues 
            # basically need robustness when dealing with ties
            return  np.searchsorted(
                     a=self.abs_neg_dist, v=(abs(val)+1E-7))/float(
                        len(self.abs_neg_dist))
