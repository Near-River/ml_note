#!/usr/bin/env python3
# coding=utf-8

"""
对率回归：西瓜数据集3.0ɑ
"""
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp, log


def build_csv_file(filename):
    raw_data = []
    with open('watermelon.data', 'r', encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
        raw_data.append((lines[1].strip().split(' '))[1:])
        for line in lines[2:]:
            (no, density, sugariness, type) = line.strip().split(' ')
            type = 1 if type == 'good' else 0
            raw_data.append((density, sugariness, type))

    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, dialect='excel')
        writer.writerows(raw_data)


def load_data(csv_file):
    df = pd.read_csv(csv_file, encoding="utf-8")
    m, n = np.shape(df)  # #samples & #attributes
    df['Norm'] = np.ones((m, 1))
    dataMat = np.mat(df[['Density', 'Sugariness', 'Norm']].values[:])
    labelMat = np.mat(df['Label'].values[:]).T
    # print(np.shape(dataMat), np.shape(labelMat))
    W = np.mat(np.zeros((n, 1)))
    return dataMat, labelMat, W


def model_training(X, Y, W):
    """
    Maximal likelihood method: W* = arg min l(W)
    """
    m, n = np.shape(X)
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


def model_view(X, Y, W):
    m, n = np.shape(X)
    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    for i in range(m):
        if Y[i] == 1:
            xcord1.append(X[i, 0])
            ycord1.append(X[i, 1])
        else:
            xcord2.append(X[i, 0])
            ycord2.append(X[i, 1])
    plt.figure(1)
    ax = plt.subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(0.2, 0.8, 0.1)
    w0, w1, w2 = float(W[0]), float(W[1]), float(W[2])
    y = -1 * ((w0 * x + w2) / w1)
    plt.sca(ax)
    plt.plot(x, y)
    plt.xlabel('Density')
    plt.ylabel('Sugariness')
    plt.title('Newton logistic regression')
    plt.show()


if __name__ == '__main__':
    csv_file = 'watermelon.csv'
    build_csv_file(csv_file)
    X, Y, W = load_data(csv_file)
    model_training(X, Y, W)
    model_view(X, Y, W)
