#!/usr/bin/env python3
# coding=utf-8

"""
简单RBF算法：解决异或问题
"""
import numpy as np
from math import exp

learning_rate = 0.1


def RBF_algorithm(X, Y):
    m, d = X.shape
    q = 10

    def radial_basis(x, c, beta):
        temp = np.sum((x - c) ** 2)
        return exp(-1 * beta * temp)

    # 随机初始化隐层神经元中心 C
    C = np.random.random_sample((q, d))
    # 随机初始化隐层神经元与输出神经元的权值
    W = np.random.random_sample((q, 1))
    # 随机初始化样本与隐层神经元的中心距离的缩放系数
    beta = np.random.random_sample((q, 1))

    iters = 0
    while True:
        iters += 1
        cur_err = 0.0
        labels = []

        dW = np.zeros((q, 1))
        dBeta = np.zeros((q, 1))
        for k in range(m):
            Rho = np.zeros((q, 1))
            for i in range(q):
                Rho[i] = radial_basis(X[k], C[i, :], beta[i])
            phi = np.sum(W.transpose().dot(Rho))
            labels.append(phi)
            # 计算累积误差
            cur_err += 1 / 2 * float(phi - Y[k]) ** 2
            for i in range(q):
                temp = (phi - Y[k]) * Rho[i]
                dW[i] -= temp
                dBeta[i] += temp * W[i] * np.sum((X[k] - C[i:]) ** 2)

        # 更新权重 W 和缩放系数 beta
        W += learning_rate * dW
        beta += learning_rate * dBeta

        # 达到停止条件
        if cur_err / m < 0.0001:
            print('Iters: ', iters * m)
            print(labels)
            break


if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    RBF_algorithm(X, Y)
