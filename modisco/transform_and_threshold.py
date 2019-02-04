from __future__ import division, print_function, absolute_import
import numpy as np


#TnT = Transform and Threshold
class AbstractTnTResults(object):

    def __init__(self, neg_threshold,
                       transformed_neg_threshold,
                       pos_threshold,
                       transformed_pos_threshold):
        #both 'transformed_neg_threshold' and 'transformed_pos_threshold'
        # should be positive, i.e. they should be relative to the
        # transformed distribution used to set the threshold, e.g. a
        # cdf value
        self.neg_threshold = neg_threshold
        assert transformed_neg_threshold >= 0.0
        self.transformed_neg_threshold = transformed_neg_threshold
        self.pos_threshold = pos_threshold
        self.transformed_pos_threshold = transformed_pos_threshold
        assert transformed_pos_threshold >= 0.0

    def get_seqlet_value_provider(track_name, central_window):
        
        def seqlet_value_provider(seqlet):
            flank_to_ignore = int(0.5*(len(seqlet)-central_window))
            track_values = seqlet[self.track_name]\
                            .fwd[flank_to_ignore:-flank_to_ignore]
            val = np.sum(track_values)
            return self.transform_val(val)

        return seqlet_value_provider

    def transform_val(self, val):
        raise NotImplementedError()

    @classmethod
    def from_hdf5(cls, grp):
        the_class = eval(grp.attrs['class'])
        return the_class.from_hdf5(grp) 


class LaplaceTnTResults(AbstractTnTResults):

    def __init__(self, neg_threshold,
                       transformed_neg_threshold,
                       neg_b,
                       pos_threshold,
                       transformed_pos_threshold,
                       pos_b, mu):
        super(AbstractTnTResults, self).__init__(
            neg_threshold=neg_threshold,
            transformed_neg_threshold=transformed_neg_threshold,
            pos_threshold=pos_threshold,
            transformed_pos_threshold=transformed_pos_threshold)
        self.neg_b = neg_b
        self.pos_b = pos_b
        self.mu = mu

    def transform_val(self, val):
        val -= self.mu
        if (val < 0):
            return -(1-np.exp(val/self.neg_b))
        else:
            return (1-np.exp(-val/self.pos_b))

    @classmethod
    def from_hdf5(cls, grp):
        mu = grp.attrs['mu'] 
        neg_threshold = grp.attrs['neg_threshold']
        transformed_neg_threshold = grp.attrs['transformed_neg_threshold']
        neg_b = grp.attrs['neg_b']
        pos_threshold = grp.attrs['pos_threshold']
        transformed_pos_threshold = grp.attrs['transformed_pos_threshold']
        pos_b = grp.attrs['pos_b']
        return cls(neg_threshold=neg_threshold,
                   transformed_neg_threshold=transformed_neg_threshold,
                   neg_b=neg_b,
                   pos_threshold=pos_threshold,
                   transformed_pos_threshold=transformed_pos_threshold,
                   pos_b=pos_b,
                   mu=mu)

    def save_hdf5(self, grp):
        grp.attrs['class'] = type(self).__name__
        grp.attrs['mu'] = self.mu
        grp.attrs['neg_threshold'] = self.neg_threshold
        grp.attrs['transformed_neg_threshold'] = self.transformed_neg_threshold
        grp.attrs['neg_b'] = self.neg_b 
        grp.attrs['pos_threshold'] = self.pos_threshold
        grp.attrs['transformed_pos_threshold'] = self.transformed_pos_threshold
        grp.attrs['pos_b'] = self.pos_b


class EmpiricalNullTnTResults(AbstractTnTResults):
    def __init__(self, neg_threshold, transformed_neg_threshold,
                       empirical_null_neg,
                       pos_threshold, transformed_neg_threshold,
                       empirical_null_pos):
        super(AbstractTnTResults, self).__init__(
            neg_threshold=neg_threshold,
            transformed_neg_threshold=transformed_neg_threshold,
            pos_threshold=pos_threshold,
            transformed_pos_threshold=transformed_pos_threshold)
        self.empirical_null_neg = empirical_null_neg
        self.empirical_null_pos = empirical_null_pos

    @classmethod
    def from_hdf5(cls, grp):
        raise NotImplementedError()

    def save_hdf5(self, grp):
        raise NotImplementedError()


class AbstractTnTFunction(object):

    def __call__(self, values, null_dist):
        raise NotImplementedError()

    @classmethod
    def from_hdf5(cls, grp):
        the_class = eval(grp.attrs["class"]) 
        return the_class.from_hdf5(grp) 


class FdrThreshFromEmpiricalNull(AbstractTnTFunction):

    def __init__(self, target_fdr, verbose):
        self.target_fdr = target_fdr
        self.verbose = verbose

    def __call__(self, values, null_dist):
        #values and null_dist are both vectors
        #sort values:
        #sort positive and negative values separately
        #return both a positive and negative threshold
        #plot the results if verbose is True
        #TODO: finish implementing 


class LaplaceTnTFunction(AbstractTnTFunction):
    count = 0
    def __init__(self, target_fdr, min_windows, verbose):
        assert (target_fdr > 0.0 and target_fdr < 1.0)
        self.target_fdr = target_fdr
        self.verbose = verbose
        self.min_windows = min_windows

    @classmethod
    def from_hdf5(cls, grp):
        target_fdr = grp.attrs["target_fdr"]
        min_windows = grp.attrs["min_windows"]
        verbose = grp.attrs["verbose"]
        return cls(target_fdr=target_fdr,
                   min_windows=min_windows, verbose=verbose)

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.attrs["target_fdr"] = self.target_fdr
        grp.attrs["min_windows"] = self.min_windows
        grp.attrs["verbose"] = self.verbose 

    def __call__(self, values, null_dist=None):

        # first estimate mu, using two level histogram to get to 1e-6
        hist1, bin_edges1 = np.histogram(values, bins=1000)
        peak1 = np.argmax(hist1)
        l_edge = bin_edges1[peak1]
        r_edge = bin_edges1[peak1+1]
        top_values = values[ (l_edge < values) & (values < r_edge) ]

        hist2, bin_edges2 = np.histogram(top_values, bins=1000)
        peak2 = np.argmax(hist2)
        l_edge = bin_edges2[peak2]
        r_edge = bin_edges2[peak2+1]
        mu = (l_edge + r_edge) / 2
        print("peak(mu)=", mu)

        pos_values = np.array(sorted(values[values > mu] - mu))
        neg_values = np.array(sorted(values[values < mu] - mu, key=lambda x: -x))

        #We assume that the null is governed by a laplace, because
        #that's what I (Av Shrikumar) have personally observed
        #But we calculate a different laplace distribution for
        # positive and negative values, in case they are
        # distributed slightly differently
        #estimate b using the percentile
        #for x below 0:
        #cdf = 0.5*exp(x/b)
        #b = x/(log(cdf/0.5))
        neg_laplace_b = np.percentile(neg_values, 95)/(np.log(0.95))
        pos_laplace_b = (-np.percentile(pos_values, 5))/(np.log(0.95))

        #for the pos and neg, compute the expected number above a
        #particular threshold based on the total number of examples,
        #and use this to estimate the fdr
        #for pos_null_above, we estimate the total num of examples
        #as 2*len(pos_values)
        pos_fdrs = (len(pos_values)*(np.exp(-pos_values/pos_laplace_b)))/(
                    len(pos_values)-np.arange(len(pos_values)))
        pos_fdrs = np.minimum(pos_fdrs, 1.0)
        neg_fdrs = (len(neg_values)*(np.exp(neg_values/neg_laplace_b)))/(
                    len(neg_values)-np.arange(len(neg_values)))
        neg_fdrs = np.minimum(neg_fdrs, 1.0)

        pos_fdrs_passing_thresh = [x for x in zip(pos_values, pos_fdrs)
                                   if x[1] <= self.target_fdr]
        neg_fdrs_passing_thresh = [x for x in zip(neg_values, neg_fdrs)
                                   if x[1] <= self.target_fdr]
        if (len(pos_fdrs_passing_thresh) > 0):
            pos_threshold, pos_thresh_fdr = pos_fdrs_passing_thresh[0]
        else:
            pos_threshold, pos_thresh_fdr = pos_values[-1], pos_fdrs[-1]
            pos_threshold += 0.0000001
        if (len(neg_fdrs_passing_thresh) > 0):
            neg_threshold, neg_thresh_fdr = neg_fdrs_passing_thresh[0]
            neg_threshold = neg_threshold - 0.0000001
        else:
            neg_threshold, neg_thresh_fdr = neg_values[-1], neg_fdrs[-1]

        if (self.min_windows is not None):
            num_pos_passing = np.sum(pos_values > pos_threshold)
            num_neg_passing = np.sum(neg_values < neg_threshold)
            if (num_pos_passing + num_neg_passing < self.min_windows):
                #manually adjust the threshold
                shifted_values = values - mu
                values_sorted_by_abs = sorted(np.abs(shifted_values), key=lambda x: -x)
                abs_threshold = values_sorted_by_abs[self.min_windows-1]
                if (self.verbose):
                    print("Manually adjusting thresholds to get desired num seqlets")
                pos_threshold = abs_threshold
                neg_threshold = -abs_threshold
        
        pos_threshold_cdf = 1-np.exp(-pos_threshold/pos_laplace_b)
        neg_threshold_cdf = 1-np.exp(neg_threshold/neg_laplace_b)
        #neg_threshold = np.log((1-self.threshold_cdf)*2)*neg_laplace_b
        #pos_threshold = -np.log((1-self.threshold_cdf)*2)*pos_laplace_b
        
        neg_threshold += mu
        pos_threshold += mu
        neg_threshold = min(neg_threshold, 0)
        pos_threshold = max(pos_threshold, 0)
        
        

        #plot the result
        if (self.verbose):
            print("Mu: %e +/- %e" % (mu, (r_edge-l_edge)/2))
            print("Lablace_b:",neg_laplace_b,"and",pos_laplace_b)
            print("Thresholds:",neg_threshold,"and",pos_threshold)
            print("#fdrs pass:",len(neg_fdrs_passing_thresh),"and", len(pos_fdrs_passing_thresh))
            print("CDFs:",neg_threshold_cdf,"and",pos_threshold_cdf)
            print("Est. FDRs:",neg_thresh_fdr,"and",pos_thresh_fdr)
            neg_linspace = np.linspace(np.min(values), mu, 100)
            pos_linspace = np.linspace(mu, np.max(values), 100)
            neg_laplace_vals = (1/(2*neg_laplace_b))*np.exp(
                            -np.abs(neg_linspace-mu)/neg_laplace_b)
            pos_laplace_vals = (1/(2*pos_laplace_b))*np.exp(
                            -np.abs(pos_linspace-mu)/pos_laplace_b)
            from matplotlib import pyplot as plt
            plt.figure()
            hist, _, _ = plt.hist(values, bins=100)
            plt.plot(neg_linspace,
                     neg_laplace_vals/(
                      np.max(neg_laplace_vals))*np.max(hist))
            plt.plot(pos_linspace,
                     pos_laplace_vals/(
                      np.max(pos_laplace_vals))*np.max(hist))
            plt.plot([neg_threshold, neg_threshold],
                     [0, np.max(hist)])
            plt.plot([pos_threshold, pos_threshold],
                     [0, np.max(hist)])
            if plt.isinteractive():
                plt.show()
            else:
                import os, errno
                try:
                    os.makedirs("figures")
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                fname = "figures/laplace_" + str(LaplaceThreshold.count) + ".png"
                plt.savefig(fname)
                print("saving plot to " + fname)
                LaplaceThreshold.count += 1

        return LaplaceTnTResults(
                neg_threshold=neg_threshold,
                transformed_neg_threshold=neg_threshold_cdf,
                neg_b=neg_laplace_b,
                pos_threshold=pos_threshold,
                transformed_pos_threshold=pos_threshold_cdf,
                pos_b=pos_laplace_b,
                mu=mu)

