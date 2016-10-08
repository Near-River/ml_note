#!/usr/bin/env python3
# coding=utf-8

"""
标准BP（error BackPropagation algorithm）算法：西瓜数据集3.0
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


def BP_algorithm(dataLst, labelLst):
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

    iters = 0
    while True:
        iters += 1
        cur_err = 0.0
        labels = []
        for k in range(m):
            X_k = np.array(dataLst[k]).transpose()
            Y_k = np.array([labelLst[k]]).transpose()
            # 计算隐层神经元的输出
            alpha = np.zeros((q, 1))
            B = np.zeros((q, 1))
            for h in range(q):
                alpha[h] = np.dot(V[:, h], X_k)
                B[h] = sigmoid(alpha[h] - gamma[h])
            # 计算输出神经元的输出
            beta = np.zeros((l, 1))
            Y = np.zeros((l, 1))
            for j in range(l):
                beta[j] = np.dot(W[:, j], B)
                Y[j] = sigmoid(beta[j] - theta[j])
            # 计算均方误差
            cur_err += float(np.sum((Y_k - Y) ** 2)) / 2
            # 计算输出层神经元的梯度项
            g = np.zeros((l, 1))
            for j in range(l):
                g[j] = Y[j] * (1 - Y[j]) * (Y_k[j] - Y[j])
            # 计算隐层神经元的梯度项
            e = np.zeros((q, 1))
            for h in range(q):
                e[h] = B[h] * (1 - B[h]) * np.dot(W[h, :], g)
            # 更新连接权 w_hj & v_ih
            for j in range(l):
                for h in range(q):
                    W[h, j] += learning_rate * g[j] * B[h]
            for h in range(q):
                for i in range(d):
                    V[i, h] += learning_rate * e[h] * X_k[i]
            # 更新阈值 theta_j & gamma_h
            for j in range(l):
                theta[j] -= learning_rate * g[j]
            for h in range(q):
                gamma[h] -= learning_rate * e[h]
            labels.append(float(Y))
        # 达到停止条件
        if cur_err / m < 0.001:
            print('Iters: ', iters * m)
            for i in range(m):
                print('%.2f  %.2f' % (labels[i], labelLst[i]))
            break


if __name__ == '__main__':
    dataLst, labelLst = load_data()
    BP_algorithm(dataLst, labelLst)
