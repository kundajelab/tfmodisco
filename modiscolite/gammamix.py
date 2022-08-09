##
#Python implementation of mixture of gamma distributions
# Based on the mixtools R package implementation:
# https://github.com/cran/mixtools/blob/master/R/gammamixEM.R
# Notebook at https://github.com/kundajelab/tfmodisco/blob/master/examples/
#             mixture_of_gammas/Mixture%20of%20Gamma%20Distributions.ipynb
##

from __future__ import division, print_function
import numpy as np
from collections import namedtuple
from scipy.stats import gamma
from scipy.optimize import minimize
from scipy.special import digamma
import sys


GammaMixParams = namedtuple("MixParams", ["mix_prop", "alpha", "invbeta", "k"])
GammaMixResult = namedtuple("GammaMixResult", ["params",
                                               "ll", "iteration",
                                               "expected_membership"]) 


def gammamix_init(x, mix_prop=None, alpha=None, invbeta=None, k=2):
    n = len(x)
    if (mix_prop is None):
        mix_prop = np.random.random((k,)) 
        mix_prop = mix_prop/np.sum(mix_prop)
    else:
        k = len(mix_prop)

    if (k==1):
        x_bar = np.array([np.mean(x)])
        x2_bar = np.array([np.mean(np.square(x))])
    else:
        #sort the values
        x_sort = sorted(x) 
        #figure out how many go in each mixing
        #component based on the current mixing
        #parameters
        ind = np.floor(n*np.cumsum(mix_prop)).astype("int")
        #collect the values corresponding to each
        #component to compute the initial alpha and beta
        x_part = []
        x_part.append(x_sort[0:ind[0]])
        for j in range(1,k):
            x_part.append(x_sort[ind[j-1]:ind[j]])
        x_bar = np.array([np.mean(y) for y in x_part])
        x2_bar = np.array([np.mean(np.square(y)) for y in x_part])

    if (alpha is None):
        alpha = np.square(x_bar)/(x2_bar - np.square(x_bar))

    if (invbeta is None):
        invbeta = x_bar/(x2_bar - np.square(x_bar))

    return GammaMixParams(mix_prop=mix_prop,
                          alpha=alpha,
                          invbeta=invbeta, k=k)


def gamma_component_pdfs(x, theta, k):
    component_pdfs = []
    alpha = theta[0:k]
    invbeta = theta[k:2*k]
    for j in range(k):
        component_pdfs.append(gamma.pdf(x=x, a=alpha[j], scale=invbeta[j])) 
    component_pdfs = np.array(component_pdfs)
    return component_pdfs


def log_deriv_gamma_component_pdfs(x, theta, k):
    log_deriv_alpha_component_pdfs = []
    log_deriv_invbeta_component_pdfs = []
    alpha = theta[0:k]
    invbeta = theta[k:2*k]
    for j in range(k):
        log_deriv_invbeta_component_pdfs.append(
            (x/(invbeta[j]**2) - alpha[j]/invbeta[j]))
        log_deriv_alpha_component_pdfs.append(
            (np.log(x) - np.log(invbeta[j]) - digamma(alpha[j])))
    return (np.array(log_deriv_invbeta_component_pdfs),
            np.array(log_deriv_alpha_component_pdfs))


def gamma_ll_func_to_optimize(theta, x, expected_membership, mix_prop, k):
    component_pdfs = gamma_component_pdfs(x=x,
                                          theta=theta, k=k)
    if (np.isnan(np.sum(component_pdfs))):
        assert False
    #prevent nan errors for np.log
    component_pdfs = component_pdfs+((component_pdfs == 0)*1e-32)
    #log likelihood
    ll =  -np.sum(expected_membership*np.log(
                  mix_prop[:,None]*component_pdfs))
    #log deriv gamma component pdfs
    (log_deriv_invbeta_component_pdfs,
     log_deriv_alpha_component_pdfs) =\
     log_deriv_gamma_component_pdfs(x=x, theta=theta, k=k) 

    log_derivs = np.array(
        list(-np.sum(
             expected_membership
             *log_deriv_alpha_component_pdfs, axis=1))+
        list(-np.sum(
          expected_membership
          *log_deriv_invbeta_component_pdfs, axis=1)))
    
    return ll, log_derivs
                                                          

#based on https://github.com/cran/mixtools/blob/master/R/gammamixEM.R
def gammamix_em(x, mix_prop=None, alpha=None, invbeta=None,
                k=2, epsilon=0.001, maxit=1000,
                maxrestarts=20, progress_update=20, verb=False):

    #initialization
    x = np.array(x) 
    mix_prop, alpha, invbeta, k =\
        gammamix_init(x=x, mix_prop=mix_prop, alpha=alpha,
                      invbeta=invbeta, k=k) 
    if (verb):
        print("initial vals:",mix_prop, alpha, invbeta, k) 
        sys.stdout.flush()
    theta = np.concatenate([alpha, invbeta],axis=0)
    
    iteration = 0
    mr = 0
    diff = epsilon + 1
    n = len(x)

    
    old_obs_ll = np.sum(np.log(np.sum(
                    mix_prop[:,None]*gamma_component_pdfs(
                        x=x,
                        theta=theta, k=k), axis=0))) 

    ll = [old_obs_ll]

    best_result = None
    best_obs_ll = old_obs_ll

    while ((np.abs(diff) > epsilon) and (iteration < maxit)):
        #dens1 = mix_prop[:,None]*gamma_component_pdfs(
        #                             x=x,
        #                             theta=theta, k=k)
        dens1 = mix_prop[:,None]*gamma_component_pdfs(
                                     x=x,
                                     theta=theta, k=k)
        expected_membership = dens1/np.sum(dens1, axis=0)[None,:] 
        mix_prop_hat = np.mean(expected_membership, axis=1)
        minimization_result = minimize(
            fun=gamma_ll_func_to_optimize,
            x0=theta,
            bounds=[(1e-7,None) for t in theta],
            args=(x, expected_membership, mix_prop, k),
            jac=True) 
        if (minimization_result.success==False):
            print(minimization_result)
            print("Choosing new starting values")
            if (mr==maxrestarts):
                raise RuntimeError("Try a different number of components?") 
            mr += 1 
            mix_prop, alpha, invbeta, k = gammamix_init(x=x, k=k) 
            theta = np.concatenate([alpha, invbeta],axis=0)
            iteration = 0
            diff = epsilon + 1
            old_obs_ll = np.sum(np.log(np.sum(
                            mix_prop[:,None]*gamma_component_pdfs(
                                x=x,
                                theta=theta, k=k), axis=0))) 
            ll = [old_obs_ll]
        else:
            theta_hat = minimization_result.x 
            alpha_hat = theta_hat[0:k]
            invbeta_hat = theta_hat[k:2*k]


            new_obs_ll = np.sum(np.log(np.sum(
                          mix_prop_hat[:,None]*gamma_component_pdfs(
                            x=x,
                            theta=theta_hat, k=k),axis=0))) 
            diff = new_obs_ll - old_obs_ll
            old_obs_ll = new_obs_ll
            ll.append(old_obs_ll)

            mix_prop = mix_prop_hat
            theta = theta_hat
            alpha = alpha_hat
            invbeta = invbeta_hat
            iteration = iteration + 1

            if (old_obs_ll >= best_obs_ll):
                best_result = GammaMixResult(
                    params=GammaMixParams(mix_prop=mix_prop,
                                          alpha=alpha, invbeta=invbeta, k=k),
                    ll=ll,
                    iteration=iteration,
                    expected_membership=expected_membership)
                best_obs_ll = old_obs_ll
                #if verb:
                #    print("New best!") 
                #    print(GammaMixParams(mix_prop=mix_prop,
                #                          alpha=alpha,
                #                          invbeta=invbeta, k=k))

            if verb:
                if (iteration%progress_update == 0):
                    print("iteration =", iteration,
                          "log-lik diff =", diff,
                          " log-lik =", new_obs_ll) 
                    sys.stdout.flush()

    if (iteration == maxit):
        print("WARNING! NOT CONVERGENT!")
    print("Number of iterations=", iteration)

    return best_result 


