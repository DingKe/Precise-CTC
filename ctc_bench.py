# coding:utf-8
# Test bench for different CTC implementations
# Author    :  David Leon (Dawei Leng)
# Created   :   1,  7, 2016
# Revised   :   1,  7, 2016
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'
from ctc_cython import ctc_path_probability as ctc_cython
from ctc_numba import ctc_path_probability as ctc_numba
from ctc_theano import CTC, ctc_path_probability as ctc_theano_nonbatch
import numpy as np, theano, time
import theano.tensor as tensor
floatX = theano.config.floatX

if __name__ == '__main__':
    np.random.seed(33)
    C = 10
    L = 50
    T = 200
    x1, x2, x3, x4, x5 = tensor.imatrix(name='queryseq'), tensor.tensor3(dtype=floatX, name='scorematrix'), \
                         tensor.fmatrix(name='queryseq_mask'), tensor.fmatrix(name='scorematrix_mask'), \
                         tensor.iscalar(name='blank_symbol')

    print('compile CTC.cost() ...', end='')
    result = CTC.cost(x1, x2, x3, x4, x5)
    f1 = theano.function([x1, x2, x3, x4, x5], result)
    print(' done')

    print('compile ctc_theano_nonbatch function ...', end='')
    Tx , Ty, Tz = tensor.dmatrix(), tensor.ivector(), tensor.iscalar()
    Tr = ctc_theano_nonbatch(Tx, Ty, Tz)
    f2 = theano.function([Tx, Ty, Tz], Tr)
    print(' done')

    scorematrix = np.random.randn(C+1, T)
    scorematrix -= np.max(scorematrix, axis=0)
    scorematrix = np.exp(scorematrix)
    scorematrix /= np.sum(scorematrix, axis=0)
    blank = C
    B = 1
    for _ in range(10):

        seq = np.floor(np.random.rand(L) * C).astype(np.int32)

        time0 = time.time()
        NLL_cython = ctc_cython(scorematrix=scorematrix, queryseq=np.array(seq, dtype=np.int32), blank=blank)

        time1 = time.time()
        NLL_numba, alphas = ctc_numba(scorematrix, seq, blank=blank)

        time2 = time.time()
        NLL_theano_nonbatch = f2(scorematrix, seq, blank)

        time3 = time.time()
        y=seq.reshape([L, B])
        yhat = scorematrix.T.reshape([T, C+1, B])
        NLL_theano_batch = f1(y, yhat, np.ones_like(y, dtype=np.float32), np.ones([T,1], dtype=np.float32), blank)
        time4 = time.time()

        print("NLL_cython = %f, NLL_numba = %f, NLL_theano_nonbatch = %f, NLL_theano_batch = %f" %
              (NLL_cython, NLL_numba, NLL_theano_nonbatch[0], NLL_theano_batch))
        print("time_cython = %0.4f, time_numba = %0.4f, time_theano_nonbatch = %0.4f, time_theano_batch = %0.4f" %
              (time1-time0, time2-time1, time3-time2, time4-time3))
        print()
