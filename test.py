import numpy as np
import pandas as pd
import math
from scipy.stats import logistic


if __name__ == '__main__':
    # m = {}
    #
    # a = [['a','b'],['f']]
    # m[0] = a
    # b = ['c']
    # m[0].append(b)
    # m.append(a)
    # print(m)
    # -------------------#
    # #-------------------#
    # all dataset
    # f = open('TaFeng/dataset.txt','a')
    # fa = open('TaFeng/test.txt','r')
    # fb = open('TaFeng/train.txt','r')
    # fc = open('TaFeng/validation.txt','r')
    #
    # a = fa.readline()
    # b = fb.readline()
    # c = fc.readline()
    # while a:
    #     f.write(a)
    #     a = fa.readline()
    #
    # while b:
    #     f.write(b)
    #     b = fb.readline()
    #
    # while c:
    #    f.write(c)
    #    c = fc.readline()
    #
    # fa.close()
    # fb.close()
    # fc.close()
    # f.close()
    #------------------
    # train dataset
    # f = open('TaFeng/train.txt','a')
    # fa = open('TaFeng/D01.txt','r')
    # a = fa.readline()
    # while a:
    #     f.write(a)
    #     a = fa.readline()
    # fa.close()
    # f.close()
    # ------------#
    # file = open('TaFeng/train_textFormat1.csv')
    # m = np.array([])
    # print(m)
    # suma = []
    # for i in range(10):
    #     a = file.readline().strip().split(',')
    #     b = list(map(eval, a))
    #     # print(b)
    #     d = []
    #     d = np.vstack((d,b))
    #     print(d)
    # --------------------------------------------------
    # file = open('TaFeng/train_textFormat.csv')
    # dataset = list()
    # label = list()
    # # print(result)
    # for i in range(100):
    #     a = file.readline().strip().split(',')
    #     label.append(int(a[0]))
    #     # b = list(map(eval, a[1:-1]))
    #     b = map(eval, a[1:-1])
    #     dataset.append(b)
    #
    # dataset = np.array(dataset)
    #
    # print(dataset)
    #
    # print(label)
    # ---------------------------------------------------



    # dataset = [
    #         [0,7,11],
    #         [0,4,8],
    #         [0,3,7],
    #         [0,5,6],
    #         [0,10,11],
    #         [1,7,11],
    #         [2,7,11],
    #         [5,7,11],
    #         [1,2]
    #     ]
    # dataset.append([3,4])
    # print(type(dataset))
    # dataset = np.array(dataset)
    # print(type(dataset))

    # a = list()
    # b = [1,2,3]
    # c = [4,5,6]
    # a.append(b)
    # a.append(c)
    # print(a)
    # b = logistic.cdf(-711)

    # a = np.exp(711)
    # intermediate = 1.0/a

    # a = math.exp(-711.945091687)
    #
    b = np.random.random((10, 6))
    print(b[1][3])