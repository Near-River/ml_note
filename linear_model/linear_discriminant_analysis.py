#!/usr/bin/env python3
# coding=utf-8

"""
线性判别分析(LDA)：西瓜数据集3.0ɑ
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file = 'watermelon.csv'


def model_training():
    df = pd.read_csv(csv_file, encoding="utf-8")
    df0, df1 = df[df.Label == 0], df[df.Label == 1]
    m0, m1 = df0.shape[0], df1.shape[0]
    X0 = np.mat(df0[['Density', 'Sugariness']].values[:])
    X1 = np.mat(df1[['Density', 'Sugariness']].values[:])
    # 计算均值向量
    mean0 = np.mat(np.average(X0, axis=0)).T
    mean1 = np.mat(np.average(X1, axis=0)).T
    # 计算协方差矩阵
    covmatrix0, covmatrix1 = np.mat(np.zeros((2, 2))), np.mat(np.zeros((2, 2)))
    for i in range(m0):
        covmatrix0 += (X0[i].T - mean0) * (X0[i] - mean0.T)
    for i in range(m1):
        covmatrix1 += (X1[i].T - mean1) * (X1[i] - mean1.T)
    # 奇异值分解
    U, S, VT = np.linalg.svd(covmatrix0 + covmatrix1)
    m, n = U.shape[0], VT.shape[0]
    sigma = np.zeros((m, n))
    for i in range(min(m, n)):
        sigma[i][i] = S[i]
    U, S, V = np.mat(U), np.mat(sigma), np.mat(VT.transpose())
    W = V * S.I * U.T * (mean0 - mean1)
    # W = (covmatrix0 + covmatrix1).I * (mean0 - mean1)
    # plot
    xcord1 = (X1[:, 0].T.tolist())[0]
    ycord1 = (X1[:, 1].T.tolist())[0]
    xcord2 = (X0[:, 0].T.tolist())[0]
    ycord2 = (X0[:, 1].T.tolist())[0]
    plt.figure(1)
    ax = plt.subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-0.2, 1.2, 0.1)
    w0, w1 = float(W[0]), float(W[1])
    y = -1 * ((w0 * x) / w1)
    plt.sca(ax)
    plt.plot(x, y)
    plt.xlabel('Density')
    plt.ylabel('Sugariness')
    plt.title('LDA')

    plt.show()


if __name__ == '__main__':
    model_training()
