# accomplish the FM
# train_dataset is train_textFormat.txt
# validation_dataset is validation_textFormat.txt
# test_dataset is test_textFormat.txt

import numpy as np
from scipy.stats import logistic


# 1、加载数据：第一列当作Y（label），第二列为user，第三列为target，后面为basket
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

    def __init__(self, num_user, num_item, n_iter, k_dim, learning_rate, lamada, dataset, label):
        self.w0 = np.random.random()
        self.w = np.random.random(num_user + num_item + num_item)
        self.v = np.random.random((num_user + num_item + num_item, k_dim))
        self.num_user = num_user
        self.num_item = num_item
        self.n_iter = n_iter
        self.k_dim = k_dim
        self.learning_rate = learning_rate
        self.lamada = lamada  # regularization parameter
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
        lamada = self.lamada

        # 迭代开始
        for step in range(step):
            # 根据求导情况，需要求解vT中每一列的和
            # 每次迭代，重新计算Vjk
            Vjk = np.zeros((len(dataset), k_dim))
            for k in range(k_dim):
                sum_result = 0
                for row in range(len(dataset)):
                    for column in range(len(dataset[row])):  # 只选择存在关系的
                        j = dataset[row][column]  # v矩阵中的行数
                        sum_result += v[j][k]
                    Vjk[row][k] = sum_result
                    sum_result = 0
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
                        # 计算梯度，然后更新三个参数
                        # 此处还要减去user与basket-item之间的组合关系
                        gradient_k2 = intermediate * (Vjk[row][f] - v[i][f]) + 2 * lamada * v[i][f]
                        # 11.26 by hucheng
                        v[i][f] = v[i][f] - learning_rate * gradient_k2
                    gradient_k0 = intermediate + 2 * lamada * k0
                    gradient_k1 = intermediate + 2 * lamada * w[i]
                    k0 -= learning_rate * gradient_k0
                    w[i] -= learning_rate * gradient_k1
        return k0, w, v


if __name__ == '__main__':

    file = open('TaFeng/train_textFormat.csv')
    dataset = list()
    label = list()
    line = file.readline()
    i = 1
    while line.strip() != '':
        line = line.strip().split(',')
        dataLine = file.readline().strip().split(',')
        label.append(int(dataLine[0].strip()))
        data = list(map(eval, dataLine[1:-1]))
        dataset.append(data)
        line = file.readline()

        #  为来训练时间断，先用10000行数据进行训练
        #     if i == 10000:
        #         break
        #     else:
        #         i += 1
    dataset = np.array(dataset)

    # user_dict,item_dict = BFM_Load_DataSet.createDict()
    # num_user = len(user_dict)
    # num_item = len(item_dict)
    num_user = 32266
    num_item = 23812
    n_iter = 50
    k_dim = 8
    # num_user, num_item, n_iter, k_dim, learning_rate, alpha0, alpha1, alpha2, dataset, label
    fm = FM(num_user, num_item, n_iter, k_dim, 0.08, 0.01, dataset, label)
    k0, w, v = fm.train()

    k1 = 0
    k2 = 0

    validation_optive = [[271, 37157]]
    validation_negtive = [[5103, 40268]]
    # # validation = [[0,7]]
    validation = np.array(validation_optive)
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
    print('y=', y)
    # a 应该要无限接近为1
    print('a=', a)

    # print(k0)
    # print(len(w))
    # print(len(v))
