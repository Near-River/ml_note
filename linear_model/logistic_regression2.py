#!/usr/bin/env python3
# coding=utf-8

"""
对率回归：选择两个UCI数据集，比较10折交叉验证法和留一法所估计出的对率回归的错误率。
    iris 数据集
"""
import numpy as np
import pandas as pd
from math import exp, log

"""
dataset: iris
class:
-- Iris Setosa
-- Iris Versicolour
-- Iris Virginica
"""
csv_file = 'iris.csv'


def validate():
    df = pd.read_csv(csv_file, encoding="utf-8")
    # 简单起见，选取 Iris Setosa 和 Iris Versicolour 这两类数据集做评估方法的比较
    m, n = np.shape(df)
    df['norm'] = np.ones((m, 1))
    X = np.mat(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'norm']].values[:100])
    labels = []
    for label in df['label'].values[:100]:
        labels.append(0) if label == 'Iris-setosa' else labels.append(1)
    Y = np.mat(labels).T
    # 10折交差验证
    errs_count = 0  # 记录错误分类的总数
    k = 100
    for j in range(10):
        W = np.mat(np.zeros((n, 1)))
        old_l, cur_l = 0, 0  # 记录上次计算的 l 值， 当前的 l 值
        iters = 0  # 计算迭代次数
        while iters < 400:
            # calculate l
            cur_l = 0
            WX = np.mat(np.zeros((k, 1)))
            for i in range(k):
                if 5 * j <= i <= (5 * j + 4) or (50 + 5 * j) <= i <= (54 + 5 * j): continue
                WX[i] = X[i] * W
                cur_l += float(-1 * Y[i] * WX[i] + log(1 + exp(WX[i])))
            if abs(cur_l - old_l) < 0.001: break
            # update W and save l
            old_l = cur_l
            dl_1, dl_2 = np.mat(np.zeros((n, 1))), np.mat(np.zeros((n, n)))  # l 对 W 的一阶求导， l 对 W 的二阶求导
            p1 = np.mat(np.zeros((k, 1)))  # P(y_i=1 | W)
            for i in range(k):
                if 5 * j <= i <= (5 * j + 4) or (50 + 5 * j) <= i <= (54 + 5 * j): continue
                p1[i] = 1 - 1.0 / (1 + exp(WX[i]))
                dl_1 -= X[i].T * (Y[i] - p1[i])
                dl_2 += X[i].T * X[i] * float(p1[i] * (1 - p1[i]))
            W -= dl_2.I * dl_1
            iters += 1
        # print('Iters: ', iters)
        # 使用测试集验证模型性能
        for i in range(5 * j, 5 * j + 5):
            p1 = 1 - 1.0 / (1 + exp(X[i] * W))
            if p1 >= 0.5: errs_count += 1
        for i in range(50 + 5 * j, 55 + 5 * j):
            p1 = 1 - 1.0 / (1 + exp(X[i] * W))
            if p1 < 0.5: errs_count += 1
    print('Error Rate: %f%%' % ((errs_count / 10) * 100.0))

    # 留一法
    errs_count = 0  # 记录错误分类的总数
    for j in range(k):
        W = np.mat(np.zeros((n, 1)))
        old_l, cur_l = 0, 0  # 记录上次计算的 l 值， 当前的 l 值
        iters = 0  # 计算迭代次数
        while iters < 400:
            # calculate l
            cur_l = 0
            WX = np.mat(np.zeros((k, 1)))
            for i in range(k):
                if i == j: continue
                WX[i] = X[i] * W
                cur_l += float(-1 * Y[i] * WX[i] + log(1 + exp(WX[i])))
            if abs(cur_l - old_l) < 0.001: break
            # update W and save l
            old_l = cur_l
            dl_1, dl_2 = np.mat(np.zeros((n, 1))), np.mat(np.zeros((n, n)))  # l 对 W 的一阶求导， l 对 W 的二阶求导
            p1 = np.mat(np.zeros((k, 1)))  # P(y_i=1 | W)
            for i in range(k):
                if i == j: continue
                p1[i] = 1 - 1.0 / (1 + exp(WX[i]))
                dl_1 -= X[i].T * (Y[i] - p1[i])
                dl_2 += X[i].T * X[i] * float(p1[i] * (1 - p1[i]))
            W -= dl_2.I * dl_1
            iters += 1
        # 验证留下的唯一样例的正确性
        p1 = 1 - 1.0 / (1 + exp(X[j] * W))
        if j < 50:
            if p1 >= 0.5: errs_count += 1
        else:
            if p1 < 0.5: errs_count += 1
    print('Error Rate: %f%%' % (1.0 * errs_count))


if __name__ == '__main__':
    validate()
