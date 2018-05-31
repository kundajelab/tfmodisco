from __future__ import division, print_function
import numpy as np
from collections import namedtuple
from scipy.stats import gamma
from scipy.optimize import minimize


GammaMixParams = namedtuple("MixParams", ["mix_prop", "alpha", "beta", "k"])


def gammamix_init(x, mix_prop=None, alpha=None, beta=None, k=2):
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
        x_part = np.array(x_part)
        x_bar = np.mean(x_part, axis=1) 
        x2_bar = np.mean(np.square(x_part), axis=1)

    if (alpha is None):
        alpha = np.square(x_bar)/(x2_bar - np.square(x_bar))

    if (beta is None):
        beta = (x2_bar - np.square(x_bar))/x_bar

    return GammaMixParams(mix_prop=mix_prop,
                          alpha=alpha,
                          beta=beta, k=k)


def gamma_component_pdfs(mix_prop, theta, k):
    component_pdfs = []
    alpha = theta[0:k]
    beta = theta[k:2*k]
    scale = 1.0/beta 
    for j in range(k):
        component_pdfs.append(gamma.pdf(x=x, a=alpha[j], scale=1.0/beta[j])) 
    component_pdfs = np.array(component_pdfs)
    component_pdfs = mix_prop[:,None]*component_pdfs
    return component_pdfs
    

def gamma_ll_func_to_optimize(theta, expected_membership, mix_prop, k):
    return -np.sum(expected_membership*np.log(
                    gamma_component_pdfs(mix_prop=mix_prop,
                                         theta=theta, k=k)))
                                                          

#based on https://github.com/cran/mixtools/blob/master/R/gammamixEM.R
def gammamix_em(x, mix_prop=None, alpha=None, beta=None,
                k=2, epsilon=1e-08, maxit=1000, maxrestarts=20, verb=False):

    #initialization
    x = np.array(x) 
    mix_prop, alpha, beta, k =\
        gammamix_init(x=x, mix_prop=mix_prop, alpha=alpha, beta=beta, k=k) 
    theta = np.concatenate([alpha, beta],axis=0)
    
    iteration = 0
    mr = 0
    diff = epsilon + 1
    n = len(x)

    old_obs_ll = np.sum(np.log(np.sum(
                    gamma_component_pdfs(
                        mix_prop=mix_prop,
                        theta=theta, k=k), axis=0))) 

    ll = [old_obs_ll]

    while ((diff > epsilon) and (iteration < maxit)):
        dens1 = gamma_component_pdfs(mix_prop=mix_prop,
                                     theta=theta, k=k)
        expected_membership = dens1/np.sum(dens1, axis=0)[None,:] 
        mix_prop_hat = np.mean(expected_membership, axis=1)
        minimization_result = minimize(
            fun=gamma_ll_func_to_optimize,
            x0=theta,
            args=(expected_membership, mix_prop, k),
            jac=False) 
        if (minimization_result.success==False):
            print("Choosing new starting values")
            if (mr==maxrestarts):
                raise RuntimeError("Try a different number of components?") 
            mr += 1 
            mix_prop, alpha, beta, k = gammamix_init(x=x, k=k) 
            theta = np.concatenate([alpha, beta],axis=0)
            iteration = 0
            diff = epsilon + 1
            old_obs_ll = np.sum(np.log(np.sum(
                            gamma_component_pdfs(
                                mix_prop=mix_prop,
                                theta=theta, k=k), axis=0))) 
            ll = [old_obs_ll]
        else:
            theta_hat = minimization_result.x 
            alpha_hat = theta_hat[0:k]
            beta_hat = theta_hat[k:2*k]


            new_obs_ll = np.sum(np.log(np.sum(gamma_density(
                            mix_prop=mix_prop_hat,
                            theta_hat=theta_hat, k=k)))) 
            diff = new_obs_ll - old_obs_ll
            old_obs_ll = new_obs_ll
            ll.append(old_obs_ll)

            mix_prop = mix_prop_hat
            theta = theta_hat
            alpha = alpha_hat
            beta = beta_hat
            iteration = iteration + 1
            if verb:
                print("iteration =", iteration,
                      "log-lik diff =", diff,
                      " log-lik =", new_obs_ll) 

    if (iteration == maxit):
        print("WARNING! NOT CONVERGENT!")
    print("Number of iterations=", iteration)
    theta = np.concatenate([alpha, beta], axis=0)

    return (GammaMixParams(mix_prop=mix_prop,
                           alpha=alpha, beta=beta, k=k),
            new_obs_ll, expected_membership)


