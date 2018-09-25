"""Implements various similarity metrics in a sliding-window fashion

Main function: sliding_similarity

General remarks:
- qa: query array (pattern) of shape (query_seqlen, channels) used for scanning
- ta: target array which gets scanned by qa of shape (..., target_seqlen, channels)

by Ziga Avsec
"""
from __future__ import division, print_function, absolute_import
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from scipy.signal import correlate


def rolling_window(a, window_width):
    """Create a new array suitable for rolling window operation

    Adopted from http://www.rigtorp.se/2011/01/01/rolling-statistics-numpy.html,
    also discussed in this PR: https://github.com/numpy/numpy/issues/7753

    Args:
      a: input array of shape (..., positions)
      window_width: width of the window to scan

    Returns:
      array of shape (..., positions - window_width + 1, window_width)
    """
    shape = a.shape[:-1] + (a.shape[-1] - window_width + 1, window_width)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def sliding_continousjaccard(qa, ta):
    """Score the region with contionous jaccard
    Args:
      qa: query array (pattern) of shape (query_seqlen, channels) used for scanning
      ta: target array which gets scanned by qa of shape (..., target_seqlen, channels)

    Returns:
      a tuple: (jaccard score of the normalized array, L1 magnitude of the scanned window)
        both are of shape (..., target_seqlen - query_seqlen + 1)
    """
    qa = qa.swapaxes(-2, -1)  #
    ta = ta.swapaxes(-2, -1)

    assert ta.shape[-1] >= qa.shape[-1]  # target needs to be longer than the query

    window_len = qa.shape[-1]
    # out_len = qa.shape[-1] - window_len + 1

    ta_strided = rolling_window(ta, window_len).swapaxes(-2, -1)

    # compute the normalization factor
    qa_L1_norm = np.sum(np.abs(qa))
    ta_L1_norm = np.sum(np.abs(ta_strided), axis=(-3, -2))
    per_pos_scale_factor = qa_L1_norm / (ta_L1_norm + (0.0000001 * (ta_L1_norm == 0)))

    ta_strided_normalized = ta_strided * per_pos_scale_factor[..., np.newaxis, np.newaxis, :]

    qa_strided = qa[..., np.newaxis]

    ta_strided_normalized_abs = np.abs(ta_strided_normalized)
    qa_strided_abs = np.abs(qa_strided)
    union = np.sum(np.maximum(ta_strided_normalized_abs, qa_strided_abs), axis=(-3, -2))
    # union = np.sum(np.maximum(np.abs(ta_strided_normalized), np.abs(qa_strided)), axis=(-3, -2))
    intersection = np.sum(np.minimum(ta_strided_normalized_abs, qa_strided_abs) *
                          np.sign(ta_strided_normalized) * np.sign(qa_strided), axis=(-3, -2))
    return intersection / union, ta_L1_norm


def parallel_sliding_continousjaccard(qa, ta, n_jobs=10, verbose=True):
    """Parallel version of sliding_continousjaccard
    """
    r = np.stack(Parallel(n_jobs)(delayed(sliding_continousjaccard)(qa, ta[i])
                                  for i in tqdm(range(len(ta)), disable=not verbose)))
    return r[:, 0], r[:, 1]

# ------------------------------------------------
# PWM scanning


def sliding_dotproduct(qa, ta):
    """'convolution' implemented in numpy with valid padding
    """
    return correlate(ta, qa[np.newaxis], mode='valid')[..., 0]


def parallel_sliding_dotproduct(qa, ta, n_jobs=10, verbose=True):
    """Parallel version of sliding_dotproduct
    """
    return np.stack(Parallel(n_jobs)(delayed(sliding_dotproduct)(qa, ta[i][np.newaxis])
                                     for i in tqdm(range(len(ta)), disable=not verbose)))[:, 0]


def sliding_similarity(qa, ta, metric='continousjaccard', n_jobs=10, verbose=True):
    """
    Args:
      qa (np.array): query array (pattern) of shape (query_seqlen, channels) used for scanning
      ta (np.array): target array which gets scanned by qa of shape (..., target_seqlen, channels)
      metric (str): similarity metric to use. Can be either from continousjaccard, dotproduct.
        dotproduct implements 'convolution' in numpy

    Returns:
      single array for dotproduct or a tuple of two arrays for continousjaccard (match and magnitude)
    """
    if metric == 'continousjaccard':
        return parallel_sliding_continousjaccard(qa, ta, n_jobs, verbose)
    elif metric == 'dotproduct':
        return parallel_sliding_dotproduct(qa, ta, n_jobs, verbose)
    else:
        raise ValueError("metric needs to be from: 'continousjaccard', 'dotproduct'")


# --------------------------------------------
# Example on how to implement pwm scanning using
#
# def pssm_scan(pwm, seqs, background_probs=[0.27, 0.23, 0.23, 0.27], n_jobs=10, verbose=True):
#     """
#     """
#     def pwm2pssm(arr, background_probs):
#         """Convert pwm array to pssm array
#         pwm means that rows sum to one
#         """
#         arr = arr / arr.sum(1, keepdims=True)
#         b = np.array(background_probs)[np.newaxis]
#         return np.log(arr / b).astype(arr.dtype)
#     pssm = pwm2pssm(pwm, background_probs)
#     return sliding_metric(pssm, seqs, 'dotproduct', n_jobs, verbose)
