# coding:utf-8
__author__ = 'dawei.leng'
__version__ = '1.20'
"""
------------------------------------------------------------------------------------------------------------------------
 Another CTC implemented in theano.
 The CTC.cost() function returns the average NLL over a batch samples given query sequences and score matrices.
 This implementation features:
    1) using time step rescaling instead of log scale computation, giving accurate path probability.
    2) batch / mask supported.
    3) speed comparable with (~35% slower than) the numba implementation which is the fastest by now.

 A longer explanation why I "reinvent the wheel":

    CTC plays a key role in LSTM-RNN training, with its power we can be liberated from the cumbersome segmentation /
 alignment task. By the time of this publication, there're already plenty of theano implementations of CTC all over the
 web. However, during my offline handwriting recognition research work with RNN, I sadly found that with these open-sourced
 theano implementations, none of them was able to compute the right path probability p(l|x) [1], though claimed successful
 RNN training's been done. This is really a pain in the ass. I've to get off the chair and dig into the origin of CTC
 algorithm to find out what went wrong.
    It took me days to read the papers, understand the algorithm and try to re-implement it on my own. Finally the culprit
 is caught. The problem rise from how the numerical normalization is done. The CTC algorithm calculates with probability
 values, which are (much) less than 1.0. This will incur underflow along the dynamic programming recursion. In [2] it's
 recommended by Alex Graves to do the calculation in log scale by
                                        ln(a + b) = lna + ln(1 + exp(lnb-lna))
 Adversely, this log scale calculation can occasionally cause numerical overflow, and this's why the above mentioned CTC
 implementations failed to compute the right path probability. The solution is to use time step rescaling method as in [3]
 instead of log scale calculation. The forward / backward variable will be rescaled at each time step of the DP recursion
 to prevent numerical underflow. My experiments have verified the effectiveness of this method.

 One somewhat confusing fact I have to mention is that in Section 7.3.1 of [2], Alex Graves stated "Note that rescaling
 the variables at every timestep is less robust, and can fail for very long sequences". Meanwhile contradictory results
 got from experiments I conducted.

 I'd like to acknowledge the authors of [4 ~ 6], their work and discussions with them are really of great help for developing
 this CTC theano implementation.

 Created   :  12, 10, 2015
 Revised   :   1,  5, 2016
 Reference :  [1] Alex Graves, etc., Connectionist temporal classification: labelling unsegmented sequence data with
                  recurrent neural networks, ICML, 2006
              [2] Alex Graves, Supervised sequence labelling with recurrent neural networks, 2014
              [3] Lawrence R. Rabiner, A tutorial on hidden Markov models and selected applications in speech recognition,
                  Proceedings of the IEEE, 1989
              [4] Maas Andrew, etc., https://github.com/amaas/stanford-ctc/blob/master/ctc_fast/ctc-loss/ctc_fast.pyx
              [5] Mohammad Pezeshki, https://github.com/mohammadpz/CTC-Connectionist-Temporal-Classification/blob/master/ctc_cost.py
              [6] Shawn Tan, https://github.com/shawntan/rnn-experiment/blob/master/CTC.ipynb
------------------------------------------------------------------------------------------------------------------------
"""
import theano
from theano import tensor
from theano.ifelse import ifelse
floatX = theano.config.floatX

class CTC(object):
    """
    Compute CTC cost, using time normalization instead of log scale computation.
    Batch supported.
    To compute the batch cost, use .cost() function below.
    Speed slower than the numba & cython version (~6min vs ~3.9min on word_correction_CTC experiment), much faster than
    the following non-batch version ctc_path_probability().
    B: BATCH_SIZE
    L: query sequence length (maximum length of a batch)
    C: class number
    T: time length (maximum time length of a batch)
    """
    @classmethod
    def cost(self, queryseq, scorematrix, queryseq_mask, scorematrix_mask, blank_symbol):
        """
        Compute CTC cost, using only the forward pass
        :param queryseq: (L, B)
        :param scorematrix: (T, C+1, B)
        :param queryseq_mask: (L, B)
        :param scorematrix_mask: (T, B)
        :param blank_symbol: scalar
        :return: negative log likelihood averaged over a batch
        """
        queryseq_padded, queryseq_mask_padded = self._pad_blanks(queryseq, blank_symbol, queryseq_mask)
        results = self.path_probability(queryseq_padded, scorematrix, queryseq_mask_padded, scorematrix_mask, blank_symbol)
        NLL = -results[1][-1]                                             # negative log likelihood
        NLL_avg = tensor.mean(NLL)                                        # batch averaged NLL, used as cost
        return NLL_avg

    @classmethod
    def path_probability(self, queryseq_padded, scorematrix, queryseq_mask_padded, scorematrix_mask, blank_symbol):
        """
        Compute p(l|x) using only the forward variable
        :param queryseq_padded: (2L+1, B)
        :param scorematrix: (T, C+1, B)
        :param queryseq_mask_padded: (2L+1, B)
        :param scorematrix_mask: (T, B)
        :param blank_symbol: = C
        :return:
        """
        pred_y = self._class_batch_to_labeling_batch(queryseq_padded, scorematrix, scorematrix_mask)  # (T, 2L+1, B), reshaped scorematrix

        r2, r3 = self._recurrence_relation(queryseq_padded, queryseq_mask_padded, blank_symbol)       # r2 (2L+1, 2L+1), r3 (2L+1, 2L+1, B)

        def step(p_curr, p_prev, LLForward, countdown):                                               # p_curr (2L+1, B), p_prev (B, 2L+1), LLForward (B, 1)
                                                                                                      # p_curr = one column of scorematrix
            dotproduct = (p_prev + tensor.dot(p_prev, r2) +                                           # tensor.dot(p_prev, r2) = alpha(t-1, u-1)
                          (p_prev.dimshuffle(1, 'x', 0) * r3).sum(axis=0).T)                          # = alpha(t-1, u-2) conditionally
            p_curr = p_curr.T * dotproduct * queryseq_mask_padded.T                                   # (B, 2L+1) * (B, 2L+1) * (B, 2L+1) = (B, 2L+1)
            start = tensor.max([0, queryseq_padded.shape[0] - 2 * countdown])
            mask = tensor.concatenate([tensor.zeros([queryseq_padded.shape[1], start]),
                                       tensor.ones([queryseq_padded.shape[1], queryseq_padded.shape[0] - start])], axis=1)
            p_curr *= mask
            c_batch = p_curr.sum(axis=1, keepdims=True)                                               # (B, 1)
            p_curr /= c_batch
            LLForward += tensor.log(c_batch)
            countdown -= 1
            return p_curr, LLForward, countdown                                                       # (B, 2L+1), (B, 1), scalar

        results, _ = theano.scan(
                step,
                sequences=[pred_y],                                                                   # scan only work on the first dimension
                outputs_info=[tensor.eye(queryseq_padded.shape[0])[0] * tensor.ones(queryseq_padded.T.shape),
                              tensor.unbroadcast(tensor.zeros([queryseq_padded.shape[1], 1]), 1), scorematrix.shape[0]])
        return results

    @staticmethod
    def _pad_blanks(queryseq, blank_symbol, queryseq_mask=None):
        """
        Pad queryseq and corresponding queryseq_mask with blank symbol
        :param queryseq  (L, B)
        :param queryseq_mask (L, B)
        :param blank_symbol  scalar
        :return queryseq_padded, queryseq_mask_padded, both with shape (2L+1, B)
        """
        # for queryseq
        queryseq_extended = queryseq.dimshuffle(1, 0, 'x')                              # (L, B) -> (B, L, 1)
        blanks = tensor.zeros_like(queryseq_extended) + blank_symbol                    # (B, L, 1)
        concat = tensor.concatenate([queryseq_extended, blanks], axis=2)                # concat.shape = (B, L, 2)
        res = concat.reshape((concat.shape[0], concat.shape[1] * concat.shape[2])).T    # res.shape = (2L, B), the reshape will cause the last 2 dimensions interlace
        begining_blanks = tensor.zeros((1, res.shape[1])) + blank_symbol                # (1, B)
        queryseq_padded = tensor.concatenate([begining_blanks, res], axis=0)            # (1+2L, B)
        # for queryseq_mask
        if queryseq_mask is not None:
            queryseq_mask_extended = queryseq_mask.dimshuffle(1, 0, 'x')                          # (L, B) -> (B, L, 1)
            concat = tensor.concatenate([queryseq_mask_extended, queryseq_mask_extended], axis=2) # concat.shape = (B, L, 2)
            res = concat.reshape((concat.shape[0], concat.shape[1] * concat.shape[2])).T
            begining_blanks = tensor.ones((1, res.shape[1]), dtype=floatX)
            queryseq_mask_padded = tensor.concatenate([begining_blanks, res], axis=0)
        else:
            queryseq_mask_padded = None
        return queryseq_padded, queryseq_mask_padded

    @staticmethod
    def _class_batch_to_labeling_batch(queryseq_padded, scorematrix, scorematrix_mask=None):
        """
        Convert dimension 'class' of scorematrix to 'label'
        :param queryseq_padded: (2L+1, B)
        :param scorematrix: (T, C+1, B)
        :param scorematrix_mask: (T, B)
        :return: (T, 2L+1, B)
        """
        scorematrix = scorematrix * scorematrix_mask.dimshuffle(0, 'x', 1)                   # (T, C+1, B) * (T, 1, B)
        batch_size = scorematrix.shape[2]  # = B
        res = scorematrix[:, queryseq_padded.astype('int32'), tensor.arange(batch_size)]     # (T, 2L+1, B), indexing each row of scorematrix with queryseq_padded
        return res

    @staticmethod
    def _recurrence_relation(queryseq_padded, queryseq_mask_padded, blank_symbol):
        """
        Generate structured matrix r2 & r3 for dynamic programming recurrence
        :param queryseq_padded: (2L+1, B)
        :param queryseq_mask_padded: (2L+1, B)
        :param blank_symbol: = C
        :return: r2 (2L+1, 2L+1), r3 (2L+1, 2L+1, B)
        """
        L2 = queryseq_padded.shape[0]                                                        # = 2L+1
        blanks = tensor.zeros((2, queryseq_padded.shape[1])) + blank_symbol                  # (2, B)
        ybb = tensor.concatenate((queryseq_padded, blanks), axis=0).T                        # (2L+3, B) -> (B, 2L+3)
        sec_diag = tensor.neq(ybb[:, :-2], ybb[:, 2:]) * tensor.eq(ybb[:, 1:-1], blank_symbol) * queryseq_mask_padded.T  # (B, 2L+1)
        r2 = tensor.eye(L2, k=1)                                                             # upper diagonal matrix (2L+1, 2L+1)
        r3 = tensor.eye(L2, k=2).dimshuffle(0, 1, 'x') * sec_diag.dimshuffle(1, 'x', 0)      # (2L+1, 2L+1, B)
        return r2, r3

def ctc_path_probability(scorematrix, queryseq, blank):
    """
    Compute path probability based on CTC algorithm, only forward pass is used.
    Batch not supported, for batch version, refer to the CTC class above
    Speed much slower than the numba & cython version (51.5min vs ~3.9min on word_correction_CTC experiment)
    :param scorematrix: (T, C+1)
    :param queryseq:    (L, 1)
    :param blank:       scalar, blank symbol
    :return: (NLL, alphas), NLL > 0 (smaller is better, = -log(p(l|x)); alphas is the forward variable)
    """

    def update_s(s, alphas, scorematrix, queryseq, blank, t):
        l = (s - 1) // 2
        alphas = ifelse(tensor.eq(s % 2, 0),
                        ifelse(tensor.eq(s, 0),
                               tensor.set_subtensor(alphas[s, t], alphas[s, t - 1] * scorematrix[blank, t]),
                               tensor.set_subtensor(alphas[s, t],
                                                    (alphas[s, t - 1] + alphas[s - 1, t - 1]) * scorematrix[blank, t]),
                               name='for_blank_symbol'),
                        ifelse(tensor.or_(tensor.eq(s, 1), tensor.eq(queryseq[l], queryseq[l - 1])),
                               tensor.set_subtensor(alphas[s, t],
                                                    (alphas[s, t - 1] + alphas[s - 1, t - 1]) * scorematrix[
                                                        queryseq[l], t]),
                               tensor.set_subtensor(alphas[s, t],
                                                    (alphas[s, t - 1] + alphas[s - 1, t - 1] + alphas[s - 2, t - 1]) *
                                                    scorematrix[queryseq[l], t]),
                               name='for_same_label_twice'))
        return alphas

    def update_t(t, LLForward, alphas, scorematrix, queryseq, blank, T, L2):
        start = tensor.max([0, L2 - 2 * (T - t)])
        end = tensor.min([2 * t + 2, L2])
        s = tensor.arange(start, end)
        results, _ = theano.scan(fn=update_s, sequences=[s], non_sequences=[scorematrix, queryseq, blank, t],
                                 outputs_info=[alphas], name='scan_along_s')
        alphas = results[-1]
        c = tensor.sum(alphas[start:end, t])
        c = tensor.max([1e-15, c])
        alphas = tensor.set_subtensor(alphas[start:end, t], alphas[start:end, t] / c)
        LLForward += tensor.log(c)
        return LLForward, alphas

    L = queryseq.shape[0]                                                 # Length of label sequence
    L2 = 2 * L + 1                                                        # Length of label sequence padded with blanks
    T = scorematrix.shape[1]                                              # time length
    alphas = tensor.zeros((L2, T))
    # Initialize alphas and forward pass
    alphas = tensor.set_subtensor(alphas[[0, 1], 0], scorematrix[[blank, queryseq[0]], 0])
    c = tensor.sum(alphas[:, 0])
    alphas = tensor.set_subtensor(alphas[:, 0], alphas[:, 0] / c)
    LLForward = tensor.log(c)
    t = tensor.arange(1, T)
    results, _ = theano.scan(fn=update_t, sequences=[t], non_sequences=[scorematrix, queryseq, blank, T, L2],
                             outputs_info=[LLForward, alphas], name='scan_along_t')
    NLL, alphas = ifelse(tensor.gt(T, 1), (-results[0][-1], results[1][-1]), (-LLForward, alphas))
    return NLL, alphas

