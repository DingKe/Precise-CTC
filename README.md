# CTC-implementations
CTC in different implementations

&nbsp;&nbsp; Another CTC implemented in theano.
&nbsp;&nbsp; The CTC.cost() function returns the average NLL over a batch samples given query sequences and score matrices.
&nbsp;&nbsp; This implementation features:
&nbsp;&nbsp;    1) using time step rescaling instead of log scale computation, giving accurate path probability.
&nbsp;&nbsp;    2) batch / mask supported.
&nbsp;&nbsp;    3) speed comparable with (~35% slower than) the numba implementation which is the fastest by now.
&nbsp;&nbsp;
&nbsp;&nbsp; A longer explanation why I "reinvent the wheel":
&nbsp;&nbsp;
&nbsp;&nbsp;    CTC plays a key role in LSTM-RNN training, with its power we can be liberated from the cumbersome segmentation /
&nbsp;&nbsp; alignment task. By the time of this publication, there're already plenty of theano implementations of CTC all over the
&nbsp;&nbsp; web. However, during my offline handwriting recognition research work with RNN, I sadly found that with these open-sourced
&nbsp;&nbsp; theano implementations, none of them was able to compute the right path probability p(l|x) [1], though claimed successful
&nbsp;&nbsp; RNN training's been done. This is really a pain in the ass. I've to get off the chair and dig into the origin of CTC
&nbsp;&nbsp; algorithm to find out what went wrong.
&nbsp;&nbsp;    It took me days to read the papers, understand the algorithm and try to re-implement it on my own. Finally the culprit
&nbsp;&nbsp; is caught. The problem rise from how the numerical normalization is done. The CTC algorithm calculates with probability
&nbsp;&nbsp; values, which are (much) less than 1.0. This will incur underflow along the dynamic programming recursion. In [2] it's
&nbsp;&nbsp; recommended by Alex Graves [2] to do the calculation in log scale by
&nbsp;&nbsp;                                        ln(a + b) = lna + ln(1 + exp(lnb-lna))
&nbsp;&nbsp; Adversely, this log scale calculation can occasionally cause numerical overflow, and this's why the above mentioned CTC
&nbsp;&nbsp; implementations failed to compute the right path probability. The solution is to use time step rescaling method as in [3]
&nbsp;&nbsp; instead of log scale calculation. The forward / backward variable will be rescaled at each time step of the DP recursion
&nbsp;&nbsp; to prevent numerical underflow. My experiments have verified the effectiveness of this method.
&nbsp;&nbsp;
&nbsp;&nbsp; One somewhat confusing fact I have to mention is that in Section 7.3.1 of [2], Alex Graves stated "Note that rescaling
&nbsp;&nbsp; the variables at every timestep is less robust, and can fail for very long sequences". Meanwhile contradictory results
&nbsp;&nbsp; got from experiments I conducted.
&nbsp;&nbsp;
&nbsp;&nbsp; I'd like to acknowledge the authors of [4 ~ 6], their work and discussions with them are really of great help for developing
&nbsp;&nbsp; this CTC theano implementation.
&nbsp;&nbsp;
&nbsp;&nbsp; Created   :  12, 10, 2015
&nbsp;&nbsp; Revised   :   1,  5, 2016
&nbsp;&nbsp; Reference :  [1] Alex Graves, etc., Connectionist temporal classification: labelling unsegmented sequence data with
&nbsp;&nbsp;                  recurrent neural networks, ICML, 2006
&nbsp;&nbsp;              [2] Alex Graves, Supervised sequence labelling with recurrent neural networks, 2014
&nbsp;&nbsp;              [3] Lawrence R. Rabiner, A tutorial on hidden Markov models and selected applications in speech recognition,
&nbsp;&nbsp;                  Proceedings of the IEEE, 1989
&nbsp;&nbsp;              [4] Maas Andrew, etc., https://github.com/amaas/stanford-ctc/blob/master/ctc_fast/ctc-loss/ctc_fast.pyx
&nbsp;&nbsp;              [5] Mohammad Pezeshki, https://github.com/mohammadpz/CTC-Connectionist-Temporal-Classification/blob/master/ctc_cost.py
&nbsp;&nbsp;              [6] Shawn Tan, https://github.com/shawntan/rnn-experiment/blob/master/CTC.ipynb
