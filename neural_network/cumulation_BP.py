#!/usr/bin/env python3
# coding=utf-8

"""
累积BP（error BackPropagation algorithm）算法：西瓜数据集3.0
"""
import numpy as np
import pandas as pd
from math import exp

csv_file = '../decision_tree/watermelon3.0.csv'
learning_rate = 0.5


def load_data():
    df = pd.read_csv(csv_file, encoding="utf-8")
    dataLst = df.values[:, -3:-1].tolist()
    labelLst = df['label'].values[:].tolist()
    return dataLst, labelLst


def ABP_algorithm(dataLst, labelLst):
    """ error BackPropagation algorithm """
    m = len(dataLst)
    d = len(dataLst[0])  # 输入神经元个数
    q, l = d + 1, 1  # 隐层神经元个数，输出神经元个数

    # 在（0，1）范围内随机初始化网络中的所有连接权和阈值
    W = np.random.random_sample((q, l))
    V = np.random.random_sample((d, q))
    theta = np.random.random_sample((l, 1))
    gamma = np.random.random_sample((q, 1))

    def sigmoid(x):
        return 1.0 / (1 + exp(-1 * x))

    X, Y = np.array(dataLst), np.array(labelLst)
    iters = 0
    while True:
        iters += 1
        cur_err = 0.0

        b = np.zeros((m, q))
        y = np.zeros((m, l))
        for k in range(m):
            # 计算隐层神经元的输出
            for h in range(q):
                b[k, h] = sigmoid(np.dot(V[:, h], X[k]) - gamma[h])
            # 计算输出神经元的输出
            for j in range(l):
                y[k, j] = sigmoid(np.dot(W[:, j], b[k]) - theta[j])
            # 计算累计误差
            cur_err += float(np.sum((y[k] - Y[k]) ** 2)) / 2

        dW = np.zeros((q, l))
        dV = np.zeros((d, q))
        dTheta = np.zeros((l, 1))
        dGamma = np.zeros((q, 1))
        for k in range(m):
            # 计算输出层神经元的梯度项
            g = np.zeros((l, 1))
            for j in range(l):
                g[j] = y[k, j] * (1 - y[k, j]) * (Y[k] - y[k, j])
            # 计算隐层神经元的梯度项
            e = np.zeros((q, 1))
            for h in range(q):
                e[h] = b[k, h] * (1 - b[k, h]) * np.dot(W[h, :], g)
            for j in range(l):
                for h in range(q):
                    dW[h, j] += g[j] * b[k, h]
            for h in range(q):
                for i in range(d):
                    dV[i, h] += e[h] * X[k, i]
            dTheta -= g
            dGamma -= e
        # 更新参数
        W += learning_rate * dW
        V += learning_rate * dV
        theta += learning_rate * dTheta
        gamma += learning_rate * dGamma

        # 达到停止条件
        if cur_err / m < 0.001:
            print('Iters: ', iters)
            for i in range(m):
                print('%.2f  %.2f' % (y[i], labelLst[i]))
            break


if __name__ == '__main__':
    dataLst, labelLst = load_data()
    ABP_algorithm(dataLst, labelLst)
