# coding:utf-8
# LSTM-CTC simulation experiment using augmented mnist data
# Created   :   1, 21, 2016
# Revised   :   1, 29, 2016
# Author    :  David Leon (Dawei Leng)
# All rights reserved
#------------------------------------------------------------------------------------------------


from keras.models import Sequential
from keras.layers.recurrent import *
from keras.optimizers import SGD
from keras.layers.core import TimeDistributedDense, Dropout

import numpy as np
import gzip, pickle, theano
from ctc import best_path_decode
from ctc_numba import ctc_path_probability as ctc_numba
from NN_auxiliary import pad_sequence_into_array
from mytheano_utils import editdist_py as editdist

#-------------- concatenate images to simulate character sequence --------------#
def mnist_concatenate_image(Xin, yin, minNcharPerseq=3, maxNcharPerseq=6):
    """
    Concatenate single char images to simulate character sequence image
    :param Xin: X_train or X_test, with shape (Nsample, Ncolumn, Nrow)
    :param yin: y_train or y_test, with shape (Nsample, )
    :param minNcharPerseq: minimum char number per sequence
    :param maxNcharPerseq: maximum char number per sequence
    :return: X_aug (B, T, D), X_aug_mask (B, T), y_aug (B, L), y_aug_mask (B, L)
    """
    Nsample = Xin.shape[0]
    RandomWalk = np.random.choice(Nsample, Nsample * maxNcharPerseq, replace = True)
    X_aug_list = []
    y_aug_list = []
    idx = 0
    for i in range(Nsample):
        seqlen = np.random.randint(minNcharPerseq, maxNcharPerseq+1)
        x = Xin[RandomWalk[idx], :, :].T
        y = yin[RandomWalk[idx]] * np.ones([seqlen], dtype=yin.dtype)
        idx += 1
        for j in range(seqlen-1):
            x = np.concatenate([x, Xin[RandomWalk[idx], :, :].T], axis=0)
            y[j+1] = yin[RandomWalk[idx]]
            idx += 1

        X_aug_list.append(x)
        y_aug_list.append(y)

    X_aug, X_aug_mask = pad_sequence_into_array(X_aug_list)
    y_aug, y_aug_mask = pad_sequence_into_array(y_aug_list)
    return X_aug, X_aug_mask, y_aug, y_aug_mask


if __name__ == '__main__':
    np.random.seed(1337) # for reproducibility
    theano.config.floatX='float64'
    with gzip.open('mnist_float64.gpkl', 'rb') as f:
        X_train, y_train, X_test, y_test = pickle.load(f)
        f.close()

    minNcharPerseq, maxNcharPerseq= 6, 6
    print('Concatenating images')
    X_train2, X_train2_mask, y_train2, y_train2_mask = mnist_concatenate_image(X_train, y_train, minNcharPerseq, maxNcharPerseq)
    X_test2, X_test2_mask, y_test2, y_test2_mask = mnist_concatenate_image(X_test, y_test, minNcharPerseq, maxNcharPerseq)

    print('X_train2 shape:', X_train2.shape)                    # (B, T, D)
    print('y_train2 shape:', y_train2.shape)                    # (B, L)


    Nclass = 10
    B, T, D = X_train2.shape   # D = 28
    L = y_train2.shape[1]

    model = Sequential()
    model.add(LSTM(input_dim=D, output_dim=100,
                   return_sequences=True, activation='tanh'))   # output (B, T, C+1)
    # model.add(Dropout(p=0.3))
    model.add(TimeDistributedDense(input_dim=100, output_dim=Nclass + 1, activation='softmax'))


    print("model compiling")
    # sgd = SGD(lr=1e-3, decay=0.0, momentum=0.8, nesterov=False)
    # model.compile(loss='ctc_cost_for_train', optimizer='Adadelta', class_mode='ctc', y_truth=y_train2, theano_mode='FAST_RUN', mask=True)
    # model.compile(loss='ctc_cost_for_train', optimizer='Adadelta', class_mode='ctc', theano_mode='FAST_RUN', mask=True)
    model.compile(loss='ctc_cost_precise', optimizer='Adadelta', class_mode='ctc', theano_mode='FAST_RUN', mask=True)

    print("model training")
    batch=1
    for j in range(100):
        TE = 0.0
        TD = 0.0
        LOSS = 0.0
        n = 0
        TE2, TD2 = 0.0, 0.0
        batches = range(0, B, batch)
        shuffle = np.random.choice(batches, size=len(batches), replace=False)
        for i in shuffle:
            ctcloss, cer, te, td, resultseq, resultseq_mask, scorematrixT = model.train_on_batch(
                    X=X_train2[i:i+batch,:,:].reshape([batch,T,D]),
                    y=y_train2[i:i+batch,:].reshape([batch, L]),
                    X_mask=X_train2_mask[i:i+batch,:].reshape([batch, T]),
                    y_mask=y_train2_mask[i:i+batch,:].reshape([batch, L]),
                    accuracy=True)
            TE += te
            TD += td
            LOSS += ctcloss*batch
            n += batch
            aveloss = LOSS / n
            print('epoch = %d, batch = %d, ave_loss = %0.4f, ave_CER = %0.4f, progress = %0.1f' % (j, i, aveloss, TE/TD, n/B))
            # print('batch loss = %0.4f, CER = %0.4f' % (ctcloss, cer))
            # print()


            b, t, cp = scorematrixT.shape
            scorematrix = scorematrixT.T.reshape(cp, t)
            resultseq2 = best_path_decode(scorematrix)
            rl = np.sum(resultseq_mask)
            queryseq = y_train2[i,:].astype(np.int32)
            ctcloss2 = ctc_numba(scorematrix, queryseq)[0]
            te2 = editdist(resultseq2, list(queryseq))
            td2 = len(queryseq)
            TE2 += te2
            TD2 += td2


            print('batch loss = %0.4f, ctc_numba = %0.4f, batch_CER = %0.4f, batch_CER2 = %0.4f' % (ctcloss, ctcloss2, cer, te2/td2))
            print('ave_CER = %0.4f, ave_CER2 = %0.4f' % (TE/TD, TE2/TD2))
            print()

            resultseq1 = list((resultseq[0:rl].T).astype(np.int32)[0])
            if resultseq1 != resultseq2:
                print('result seq not same')
                print('resultseq1 =', resultseq1)
                print('resultseq2 =', resultseq2)
                print('groundtruth=', queryseq)