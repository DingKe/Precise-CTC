# coding:utf-8
# Calculating path probability based on Alex Grave's forward-backward CTC algorithm, numba version.
# Author    :  David Leon (Dawei Leng)
# Created   :   9, 28, 2015
# Revised   :   1,  5, 2016
# Reference :  [1] Alex Graves, etc., Connectionist temporal classification: labelling unsegmented sequence data with
#                  recurrent neural networks, ICML, 2006
#              [2] Maas Andrew, etc., https://github.com/amaas/stanford-ctc/blob/master/ctc_fast/ctc-loss/ctc_fast.pyx
# Comment   :
#              1) according to real field experiments, this numba version is even a little faster than the cython version
#              2) acceleration benefiting from numba is ~100 times over pure python version
#              3) pre-allocate memory for list can bring ~10 times acceleration for pure python version
# ----------------------------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
__version__ = '1.10'
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def ctc_path_probability(scorematrix, queryseq, blank=-1):
    """
    Compute path probability based on CTC algorithm, only forward pass is used.
    scorematrix : (C+1, T) probability score matrix, usually the output of a softmax layer.
    queryseq    : query sequence, (L, 1)
    Returns negative log likelihood, i.e., smaller value is better
    """
    L = queryseq.shape[0]                                                 # Length of label sequence
    L2 = 2 * L + 1                                                        # Length of label sequence padded with blanks
    T = scorematrix.shape[1]                                              # Length of utterance (time)
    alphas = np.zeros((L2, T))
    # Initialize alphas and forward pass
    alphas[0, 0] = scorematrix[blank, 0]
    alphas[1, 0] = scorematrix[queryseq[0], 0]
    c = np.sum(alphas[:, 0])
    alphas[:, 0] = alphas[:, 0] / c
    LLForward = np.log(c)
    for t in range(1, T):
        start = max(0, L2 - 2 * (T - t))
        end = min(2 * t + 2, L2)
        for s in range(start, end):
            l = int((s - 1) / 2)
            # blank
            if s % 2 == 0:
                if s == 0:
                    alphas[s, t] = alphas[s, t - 1] * scorematrix[blank, t]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * scorematrix[blank, t]
            # same label twice
            else:
                if s == 1 or queryseq[l] == queryseq[l - 1]:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * scorematrix[queryseq[l], t]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) * scorematrix[
                        queryseq[l], t]
        # normalize at current time (prevent underflow)
        c = np.sum(alphas[start:end, t])
        alphas[start:end, t] /= c
        LLForward += np.log(c)
    return -LLForward, alphas

