# coding:utf-8
# Parse the run log of mnist_ctc experiment and do the visualization
# Created   :   1, 28, 2016
# Revised   :   1, 29, 2016
# Author    :  David Leon (Dawei Leng)
# All rights reserved
#------------------------------------------------------------------------------------------------
__author__ = 'dawei.leng'

from matplotlib import pyplot as plt

def parse_runlog(file=None, encoding='ascii'):
    BatchLoss = []
    NumbaLoss = []
    BatchCER1 = []
    BatchCER2 = []
    ave_CER1 = []
    ave_CER2 = []
    with open(file, mode='rt', encoding=encoding) as f:
        for line in f:
            if line.startswith('batch loss ='):
                strs = line.split(',')
                values = []
                for str in strs:
                    substrs = str.split('=')
                    values.append(float(substrs[1]))
                BatchLoss.append(values[0])
                NumbaLoss.append(values[1])
                BatchCER1.append(values[2])
                BatchCER2.append(values[3])
            if line.startswith('ave_CER ='):
                strs = line.split(',')
                values = []
                for str in strs:
                    substrs = str.split('=')
                    values.append(float(substrs[1]))
                ave_CER1.append(values[0])
                ave_CER2.append(values[1])

    return BatchLoss, NumbaLoss, BatchCER1, BatchCER2, ave_CER1, ave_CER2


if __name__ == '__main__':

    BatchLoss, NumbaLoss, BatchCER1, BatchCER2, ave_CER1, ave_CER2 = parse_runlog(r"mnist_ctc_v4_fortrain.log")
    fig = plt.figure()
    plt.plot(ave_CER2)
    print(min(ave_CER2))

    BatchLoss, NumbaLoss, BatchCER1, BatchCER2, ave_CER1, ave_CER2 = parse_runlog(r"mnist_ctc_v4_precise.log")
    plt.plot(ave_CER2,'r')
    plt.title('Average CER curves')
    plt.legend(['CTC_for_train.cost()', 'CTC_precise.cost()'])
    plt.ylim([0,1])
    plt.xlabel('batch (1 sample per batch)')
    plt.ylabel('Average CER')
    plt.box(on=True)
    plt.grid()
    print(min(ave_CER2))
    plt.show()