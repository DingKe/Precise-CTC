# coding:utf-8
# Calculating path probability based on Alex Grave's forward-backward CTC algorithm, cython version.
# Author    :  David Leon (Dawei Leng)
# Created   :  11, 12, 2015
# Revised   :   1,  5, 2016
# Usage     :  python setup_ctc_cython.py build_ext --inplace
# Reference :  [1] Alex Graves, etc., Connectionist temporal classification: labelling unsegmented sequence data with
#                  recurrent neural networks, ICML, 2006
#              [2] Maas Andrew, etc., https://github.com/amaas/stanford-ctc/blob/master/ctc_fast/ctc-loss/ctc_fast.pyx
# ------------------------------------------------------------------------------------------------

from libc cimport math
import cython
import numpy as np
cimport numpy as np
np.seterr(divide='raise',invalid='raise')

# Turn off bounds checking, negative indexing
@cython.boundscheck(False)
@cython.wraparound(False)

def ctc_path_probability(double[:,:] scorematrix not None, int[:] queryseq not None, int blank=0):
    """
    Compute path probability based on CTC algorithm, only forward pass is used.
    scorematrix : (C+1, T) probability score matrix, usually the output of a softmax layer.
    queryseq    : query sequence, (L, 1)
    Returns negative log likelihood, i.e., smaller value is better
    """
    cdef unsigned int L = queryseq.shape[0]                               # Length of label sequence
    cdef unsigned int numphones = scorematrix.shape[0]                    # Number of labels
    cdef unsigned int L2 = 2 * L + 1                                      # Length of label sequence padded with blanks
    cdef unsigned int T = scorematrix.shape[1]                            # Length of utterance (time)
    cdef double[:,:] alphas = np.zeros((L2, T), dtype=np.double)
    cdef unsigned int start, end, t, s, l
    cdef double c, LLFoward

    if blank < 0:
        blank += numphones
    try:
        # Initialize alphas and forward pass
        alphas[0, 0] = scorematrix[blank, 0]
        alphas[1, 0] = scorematrix[queryseq[0], 0]
        c = alphas[0, 0] + alphas[1, 0]
        alphas[0, 0] = alphas[0, 0] / c
        alphas[1, 0] = alphas[1, 0] / c
        LLFoward = math.log(c)
        for t in xrange(1, T):
            start = max(0, L2 - 2 * (T - t))
            end = min(2 * t + 2, L2)
            for s in xrange(start, L2):
                l = (s - 1) / 2
                # blank
                if s % 2 == 0:
                    if s == 0:
                        alphas[s, t] = alphas[s, t - 1] * scorematrix[blank, t]
                    else:
                        alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * scorematrix[blank, t]
                # same label twice
                elif s == 1 or queryseq[l] == queryseq[l - 1]:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1]) * scorematrix[queryseq[l], t]
                else:
                    alphas[s, t] = (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) * scorematrix[queryseq[l], t]

            # normalize at current time (prevent underflow)
            c = 0.0
            for s in xrange(start, end):
                c += alphas[s, t]
            c = max(1e-15, c)
            for s in xrange(start, end):
                alphas[s, t] = alphas[s, t] / c
            LLFoward += math.log(c)
    except (FloatingPointError, ZeroDivisionError) as e:
        print e.message
        return -1

    return -LLFoward
