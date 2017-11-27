# accomplish the FM
# train_dataset is train_textFormat.txt
# validation_dataset is validation_textFormat.txt
# test_dataset is test_textFormat.txt

import numpy as np
from scipy.stats import logistic
# import BFM_Load_DataSet


# 先实现最简单的版本


# 1、加载数据：第一列当作Y，第二列为user，第三列为target，后面为basket
# 首先实现user与target之间的FM


class FM():
    # Parameters
    # ----------
    # w0: double
    # w: np.ndarray[DOUBLE, ndim = 1, mode = 'c']
    # v: np.ndarray[DOUBLE, ndim = 8, mode = 'c']
    # num_user: int
    # num_item: int
    # n_iter: int
    # min_target: double
    # max_target: double
    # learning_rate: int


    def __init__(self, num_user, num_item, n_iter, k_dim, learning_rate, alpha0, alpha1, alpha2, dataset, label):
        self.w0 = np.random.random()
        # self.w = np.zeros(num_user + num_item)
        # self.w = np.zeros(num_user + num_item + num_item)
        self.w = np.random.random(num_user + num_item + num_item)
        # self.v = np.zeros([num_user + num_item, num_user + num_item])
        # self.v = np.zeros([num_user + num_item + num_item, k_dim])
        self.v = np.random.random((num_user + num_item + num_item, k_dim))
        self.num_user = num_user
        self.num_item = num_item
        self.n_iter = n_iter
        self.k_dim = k_dim
        self.learning_rate = learning_rate
        self.alpha0 = alpha0  # regularization parameter
        self.alpha1 = alpha1  # regularization parameter
        self.alpha2 = alpha2  # regularization parameter
        self.dataset = dataset
        self.label = label

    def train(self):

        dataset = self.dataset
        label = self.label

        w = self.w
        v = self.v
        k_dim = self.k_dim
        step = self.n_iter
        learning_rate = self.learning_rate
        alpha0 = self.alpha0
        alpha1 = self.alpha1
        alpha2 = self.alpha2
        # 对存在关系的潜在因子随机化
        # for row in range(len(dataset)):
        #     for column in range(len(dataset[row])):
        #         for f in range(k_dim):
        #             i = dataset[row][column]
        #             if label[row] == 1:
        #                 w[i] = np.random.random()
        #                 v[i][f] = np.random.random()

        # 随机初始化

        # 迭代开始
        for step in range(step):
            # 根据求导情况，需要求解vT中每一列的和
            # 每次迭代，重新计算Vjk
            Vjk = {}
            for k in range(k_dim):
                sum_result = 0
                for row in range(len(dataset)):
                    for column in range(len(dataset[row])):  # 只选择存在关系的
                        j = dataset[row][column]  # v矩阵中的行数
                        sum_result += v[j][k]
                Vjk[k] = sum_result
            # 根据对Vi,f进行求导，结果包含y的表达式，所以先求每个i下面的y值
            # y 的求解根据公式 （1.1）
            for row in range(len(dataset)):
                k0 = self.w0
                k1 = 0
                k2 = 0
                for f in range(k_dim):
                    sum_one = 0
                    sum_two = 0
                    for column in range(len(dataset[row])):
                        i = dataset[row][column]
                        k1 = k1 + w[i]
                        sum_one += v[i][f]
                        sum_two += pow(v[i][f], 2)
                    result = (pow(sum_one, 2) - sum_one) * 0.5
                    k2 = k2 + result
                y = k0 + k1 + k2
                # 根据公式（1.2）,对w0、wi，Vi,f的梯度进行求解
                intermediate = (logistic.cdf(y * label[row]) - 1) * label[row]
                # 对每个i的因子，对与其相关的f个 v[i][f] 计算梯度并更新进行更新
                for column in range(len(dataset[row])):
                    i = dataset[row][column]
                    for f in range(k_dim):
                        # 梯度计算结果好像有问题
                        gradient_k2 = intermediate * (Vjk[f] - v[i][f])/2.0 + 2 * alpha2 * v[i][f]
                        # 11.26 by hucheng
                        v[i][f] = v[i][f] - learning_rate * gradient_k2
                    gradient_k0 = intermediate + 2 * alpha0 * k0
                    gradient_k1 = intermediate + 2 * alpha1 * w[i]
                    k0 -= learning_rate * gradient_k0
                    w[i] -= learning_rate * gradient_k1

            # 计算OPT_BFM(T)
            print("TRAIN OVER")

            if(step%5==0):
                opt_bfm = 0
                for row in range(len(dataset)):
                    k1 = 0
                    k2 = 0
                    reg = 0
                    reg1 = 0
                    reg2 = 0
                    for f in range(k_dim):
                        sum_one = 0
                        sum_two = 0
                        for column in range(len(dataset[row])):
                            i = dataset[row][column]
                            sum_one += v[i][f]
                            sum_two += pow(v[i][f], 2)
                        result = (pow(sum_one, 2) - sum_two) * 0.5
                        k2 = k2 + result
                        reg2 += alpha2 * v[i][f] ** 2
                    reg1 = w[i] ** 2
                    y = k0 + k1 + k2

                    log_result = np.log(logistic.cdf(y * label[row]))
                    # print(log_result)
                    reg = reg1 + reg2
                    opt_bfm -= log_result
                    opt_bfm += reg

                opt_bfm += k0 ** 2
                print(opt_bfm)

                print(step, ' has finished,')

        return k0, w, v


if __name__ == '__main__':
    # dataset : np.ndarray
    # 1 16760:1 36578:1 56343:1 69968:1 65583:1 56522:1 56500:1 57871:1
    # 保存为这样1 16760 36578 56343 69968 65583 56522 56500 57871 list形式
    # 假设数据集如此
    # 1,0,0,0,0,0,0,1,0,0,0,1   [0，7，11]
    # 0,1,0,0,1,0,0,0,0,1,0,0   [1，4，9]
    # 0,0,1,0,0,1,0,0,0,0,1,0   [2，5，10]
    # 0,0,0,1,0,0,1,0,1,0,0,0   [3，6，8]
    # ----------------------------------
    # [1,1,1,1]

    # dataset = [
    #         [0,7,11],
    #         [1,4,9],
    #         [2,5,10],
    #         [3,6,8]
    #     ]
    # label = [1,1,1,1]
    # -------------------------------------------
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
    # print(type(dataset))
    #
    # label = [1,1,1,1,1,1,-1,1,1]
    # -------------------------------------------

    file = open('TaFeng/train_textFormat.csv')
    dataset = list()
    label = list()
    line = file.readline()
    i = 1
    while line:
        line = line.strip().split(',')
        dataLine = file.readline().strip().split(',')

        label.append(int(dataLine[0].strip()))
        data = list(map(eval, dataLine[1:-1]))
        dataset.append(data)
        line = file.readline()

        if i == 10000:
            break
        else:
            i += 1
    dataset = np.array(dataset)

    # user_dict,item_dict = BFM_Load_DataSet.createDict()
    # num_user = len(user_dict)
    # num_item = len(item_dict)
    num_user = 32266
    num_item = 23812
    # num_user, num_item, n_iter, k_dim, learning_rate, alpha0, alpha1, alpha2, dataset, label
    fm = FM(num_user, num_item, 50, 8, 0.0001, 0.01, 0.01, 0.01, dataset, label)
    k0, w, v = fm.train()

    k1 = 0
    k2 = 0
    #
    validation_optive = [[271, 37157]]
    validation_negtive = [[5103, 40268]]
    # # validation = [[0,7]]
    validation = np.array(validation_negtive)
    for row in range(len(validation)):
        for f in range(8):
            sum_one = 0
            sum_two = 0
            for column in range(len(validation[row])):
                i = validation[row][column]
                k1 = k1 + w[i]
                sum_one += v[i][f]
                sum_two += pow(v[i][f], 2)

            result = (pow(sum_one, 2) - sum_one) * 0.5
            k2 = k2 + result
        y = k0 + k1 + k2
    a = logistic.cdf(y)
    # a = pow(1 + math.exp(y  * (-1)), -1)
    print(y)
    print(a)

    print(k0)
    print(len(w))
    print(len(v))
