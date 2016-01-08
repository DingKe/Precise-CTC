# coding=utf-8
#================================================================================================================
# Useful theano routines
# Dependency :
# Author     : David Leon (Dawei Leng)
# Created    :  8, 17, 2015
# Revised    :  1,  8, 2016
# All rights reserved
#================================================================================================================
import numpy as np
import theano.tensor as tensor
import theano
from theano.ifelse import ifelse
__author__ = 'dawei.leng'
__version__ = '1.22'


#------------- Edit distance functions, 3 versions included (pure-python, numpy-based, theano-based)  ------------#

# Theano version of Levenshtein's edit distance function, basically equal with the following editdist_np(),
# except that the zero-length conditional return logic is not implemented here.
# This theano version is testified faster than the numpy version
# s, t     : theano vector, 's' for source and 't' for target
# return   : scalar, dtype = theano.config.floatX
# Example:
#    > s = tensor.vector()
#    > t = tensor.vector()
#    > d = editdist(s,t)
#    > f = theano.function([s,t],d)
def editdist(s, t):
    def update(x, previous_row):
        current_row = previous_row + 1
        current_row = tensor.set_subtensor(current_row[1:], tensor.minimum(current_row[1:], tensor.add(previous_row[:-1], tensor.neq(target,x))))
        current_row = tensor.set_subtensor(current_row[1:], tensor.minimum(current_row[1:], current_row[0:-1] + 1))
        return current_row
    source, target = ifelse(tensor.lt(s.shape[0], t.shape[0]), (t, s), (s, t))
    previous_row = tensor.arange(target.size + 1, dtype=theano.config.floatX)
    result, updates = theano.scan(fn = update, sequences=source, outputs_info=previous_row, name='editdist')
    return result[-1,-1]

# numpy version
# from [https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python, the 5th version]
def editdist_np(source, target):
    if len(source) < len(target):
        return editdist_np(target, source)
    if len(target) == 0:
        return len(source)

    previous_row = np.arange(target.size + 1)
    for s in source:
        current_row = previous_row + 1
        current_row[1:] = np.minimum(current_row[1:], np.add(previous_row[:-1], target != s))
        current_row[1:] = np.minimum(current_row[1:], current_row[0:-1] + 1)
        previous_row = current_row

    return previous_row[-1]

# Pure python version
# from [https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python, the 6th version]
def editdist_py(s, t):
        """
        From Wikipedia article; Iterative with two matrix rows.
        """
        if s == t: return 0
        elif len(s) == 0: return len(t)
        elif len(t) == 0: return len(s)
        v0 = [None] * (len(t) + 1)
        v1 = [None] * (len(t) + 1)
        for i in range(len(v0)):
            v0[i] = i
        for i in range(len(s)):
            v1[0] = i + 1
            for j in range(len(t)):
                cost = 0 if s[i] == t[j] else 1
                v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
            for j in range(len(v0)):
                v0[j] = v1[j]

        return v1[len(t)]

#------------- these following two functions can be used for RNN transcription (remove duplicate, remove blank)  ------------#
def remove_value(x, value):
    """
    Remove certain valued elements from a vector
    x: vector (must); value: scalar
    return a vector with all elements = 'value' removed
    """
    return (x - value).nonzero_values() + value

def remove_adjdup(x):
    """
    Remove adjacent duplicate items of a vector
    x: vector
    return a vector with adjacent duplicate items removed, for example [1,2,2,2,3,3,4] -> [1,2,3,4]
    """
    def update(x, nondup, idx):
        nondup = tensor.switch(tensor.eq(nondup[idx], x), nondup, tensor.set_subtensor(nondup[idx + 1], x))  # tensor.switch is much faster than ifelse
        idx = tensor.switch(tensor.eq(nondup[idx], x), idx, idx + 1)
        return nondup, idx
    nondup = x
    idx = tensor.as_tensor_variable(0)
    idx = tensor.cast(idx, 'int32')
    result, updates = theano.scan(fn = update, sequences=x, outputs_info=[nondup, idx], name='remove_adjdup')
    nondup = result[0][-1]
    idx = result[1][-1]
    return nondup[0:idx+1]


def sequence_pad(X, maxlen=None, truncating='post', padding='post', value=0.):
    """
    Function for padding sequence (list of numpy arrays)
    :param X: list of numpy arrays. The arrays must have the same shape except the first dimension.
    :param maxlen: the allowed maximum of the first dimension of X's arrays. Any array longer than maxlen is truncated to maxlen
    :param truncating: = 'pre'/'post', indicating whether the truncation happens at either the beginning or the end of the array (default)
    :param padding: = 'pre'/'post',indicating whether the padding happens at either the beginning or the end of the array (default)
    :param value: scalar, the padding value, default = 0.0
    :return: the padded sequence Xout (now a tensor with shape (Narrays, N1stdim, N2nddim, ...)
    :return: the corresponding mask, same shape with Xout
    """
    lengths = [s.shape[0] for s in X]    # 'sequences' must be list, 's' must be numpy array, len(s) return the first dimension of s
    Nsamples = len(X)
    if maxlen is None:
        maxlen = tensor.max(tensor.stack(*lengths))

    Xout = (tensor.ones(shape = [Nsamples, maxlen] + X[0].shape[2:]) * value).astype(X[0].dtype)
    Mask = tensor.zeros_like(Xout)

    def update(x, idx):
        if truncating == 'pre':
            trunc = x[-maxlen:]
        elif truncating == 'post':
            trunc = x[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if padding == 'post':
            Xout[idx, :len(trunc)] = trunc
            Mask[idx, :len(trunc)] = 1
        elif padding == 'pre':
            Xout[idx, -len(trunc):] = trunc
            Mask[idx, -len(trunc):] = 1
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    result, updates = theano.scan(fn=update, sequences=[X, tensor.arange(Nsamples)])
    return Xout, Mask

def array_split(array):
    """
    Split an array into list of subarrays along the 1st dimension
    :param array:
    :return:
    """
    def update(x):
        return x
    result, updates = theano.scan(fn=update, sequences=array, name='array_split')
    return result

def log_add(lna, lnb):
    """
    Compute the ln(a+b) given {lna,lnb}
    :param
    :return: ln(a+b)
    """
    max_ = tensor.maximum(lna, lnb)
    result = (max_ + tensor.log1p(tensor.exp(lna + lnb - 2 * max_)))   #log1p(x) = log(1+x)
    return tensor.switch(tensor.isnan(result), max_, result)


if __name__ == '__main__':
    from ctc_cython import ctc_path_probability as ctc_cython
    from ctc_theano import ctc_path_probability as ctc_theano_nonbatch
    from ctc_numba import ctc_path_probability as ctc_numba
    import pickle, time
    theano.config.mode = 'FAST_RUN'
    Tx = tensor.dmatrix()
    Ty = tensor.ivector()
    Tz = tensor.iscalar()
    time0 = time.time()
    print('compiling function ...', end='')
    Tr = ctc_theano_nonbatch(Tx, Ty, Tz)
    f = theano.function([Tx, Ty, Tz], Tr, profile=False, mode='FAST_RUN')
    time1 = time.time()
    print('done, time cost = %0.2fs' % (time1-time0))


    for i in range(50):
        C, T = 10, 200
        scorematrix = np.random.randn(C+1, T)
        scorematrix -= np.max(scorematrix, axis=0)
        scorematrix = np.exp(scorematrix)
        scorematrix /= np.sum(scorematrix, axis=0)

        queryseq = np.array([0, 1, 2, 1, 1])
        blank = -1

        # with open('myctcdebug2.pkl', 'rb') as fl:
        #     scorematrix, queryseq, blank = pickle.load(fl)
        #     fl.close()
        # print(scorematrix.shape)
        # print(queryseq.shape)
        # print(blank)


        time0 = time.time()
        result0 = ctc_numba(scorematrix, queryseq, blank)
        NLL0 = result0[0]
        time1 = time.time()
        result1 = f(scorematrix, queryseq, blank)
        NLL1 = result1[0]
        time2 = time.time()
        print('NLL0 = %f, NLL1 = %f' % (NLL0, NLL1))
        print('time0 = %0.2f, time1 = %0.2f' %(time1-time0, time2-time1))
    # f.profile.print_summary()
    # print(result1[1][-1])
    # print('result1.shape = ', result1[1][-1].shape)
    # print(result0[1])
    # print('result0.shape = ', result0[1].shape)
