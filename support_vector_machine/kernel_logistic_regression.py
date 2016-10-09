#!/usr/bin/env python3
# coding=utf-8

"""
核对率回归：西瓜数据集
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp, log

csv_file = '../linear_model/watermelon.csv'
sigma = 0.5  # 高斯核的带宽


def gauss_kernel(X_i, X_j):
    return exp(-1 * np.sum((X_i - X_j) ** 2) / (2 * sigma * sigma))


def load_data():
    df = pd.read_csv(csv_file, encoding="utf-8")
    dataMat = df[['Density', 'Sugariness']].values[:]
    labelMat = df['Label'].values[:]
    return dataMat, labelMat


def model_training(X, Y):
    """ Maximal likelihood method: W* = arg min l(W) """
    m, n = X.shape
    K = np.mat(np.zeros((m, m)))  # 高斯核矩阵
    for i in range(m):
        for j in range(m):
            K[i, j] = gauss_kernel(X[i, :], X[j, :])
    X = np.mat(np.ones((m, m + 1)))
    Y = np.mat(Y).T
    X[:, :-1] = K
    W = np.mat(np.zeros((m + 1, 1)))

    m, n = X.shape
    old_l, cur_l = 0, 0  # 记录上次计算的 l 值， 当前的 l 值
    iters = 0  # 计算迭代次数
    while iters < 400:
        # calculate l
        cur_l = 0
        WX = np.mat(np.zeros((m, 1)))
        for i in range(m):
            WX[i] = X[i] * W
            cur_l += float(-1 * Y[i] * WX[i] + log(1 + exp(WX[i])))
        if abs(cur_l - old_l) < 0.001: break
        # update W and save l
        old_l = cur_l
        dl_1, dl_2 = np.mat(np.zeros((n, 1))), np.mat(np.zeros((n, n)))  # l 对 W 的一阶求导， l 对 W 的二阶求导
        p1 = np.mat(np.zeros((m, 1)))  # P(y_i=1 | W)
        for i in range(m):
            p1[i] = 1 - 1.0 / (1 + exp(WX[i]))
            dl_1 -= X[i].T * (Y[i] - p1[i])
            dl_2 += X[i].T * X[i] * float(p1[i] * (1 - p1[i]))
        W -= dl_2.I * dl_1
        iters += 1
    print('Iters: ', iters)


if __name__ == '__main__':
    dataMat, labelMat = load_data()
    model_training(dataMat, labelMat)
