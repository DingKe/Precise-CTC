# CTC-implementations
CTC in different implementations, including cython, numba/python and theano versions.  

All the implementations use time step rescaling instead of log scale calculation, giving accurate path probability p(l|x).
The theano implementation includes a batch version and a non-batch version. The batch version can be used for LSTM-RNN training.

### A longer explanation why I "reinvent the wheel":

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

Reference :  
              [1] Alex Graves, etc., Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks, ICML, 2006  
              [2] Alex Graves, Supervised sequence labelling with recurrent neural networks, 2014  
              [3] Lawrence R. Rabiner, A tutorial on hidden Markov models and selected applications in speech recognition, Proceedings of the IEEE, 1989  
              [4] Maas Andrew, etc., https://github.com/amaas/stanford-ctc/blob/master/ctc_fast/ctc-loss/ctc_fast.pyx  
              [5] Mohammad Pezeshki, https://github.com/mohammadpz/CTC-Connectionist-Temporal-Classification/blob/master/ctc_cost.py  
              [6] Shawn Tan, https://github.com/shawntan/rnn-experiment/blob/master/CTC.ipynb  
              
