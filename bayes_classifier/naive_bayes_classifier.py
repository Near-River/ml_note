#!/usr/bin/env python3
# coding=utf-8

"""
朴素贝叶斯分类器（使用拉普拉斯修正）：西瓜数据集3.0
"""
import numpy as np
import pandas as pd
from math import sqrt, exp, pow, pi

csv_file = '../decision_tree/watermelon3.0.csv'


def load_data():
    df = pd.read_csv(csv_file, encoding="utf-8")
    dataset = df.values[:, 1:]
    return dataset


def normal_distribution_func(u, sigma):
    def calc_probability(x):
        return 1 / (sqrt(2 * pi) * sigma) * exp(-0.5 * pow(x - u, 2) / pow(sigma, 2))

    return calc_probability


def build_bayes_classifier(dataset):
    m, n = dataset.shape
    cls = list(set([sample[-1] for sample in dataset]))
    N = len(cls)  # number of the categories
    subdatasetMap = {}
    for i in range(m):
        c = dataset[i][-1]
        if c in subdatasetMap:
            subdatasetMap[c].append(dataset[i])
        else:
            subdatasetMap[c] = [dataset[i]]

    Prob = {}  # 类先验概率查询表
    for c_i in cls:
        m_i = len(subdatasetMap[c_i])
        Prob[c_i] = (m_i + 1) / (m + N)
    Prob2 = {}  # 类条件概率查询表
    for i in range(n - 1):
        if type(dataset[0][i]).__name__ not in ('int', 'float'):
            uniqAttrVals = list(set([sample[i] for sample in dataset]))
            N_i = len(uniqAttrVals)
            for attrVal in uniqAttrVals:
                for c in cls:
                    count = 0  # 训练集中属于 c 类且在属性 x_i 上取值为 attrVal 的样本个数
                    for sample in subdatasetMap[c]:
                        if sample[i] == attrVal: count += 1
                    m_c = len(subdatasetMap[c])
                    Prob2[(attrVal, c)] = (count + 1) / (m_c + N_i)
        else:  # continuous feature
            for c in cls:
                m_c = len(subdatasetMap[c])
                u, sigma = 0.0, 0.0
                valsLst = []
                for sample in subdatasetMap[c]:
                    valsLst.append(sample[i])
                    u += sample[i]
                u /= m_c
                for val in valsLst:
                    sigma += (val - u) ** 2
                sigma = sqrt(sigma / m_c)
                Prob2[(i, c)] = normal_distribution_func(u, sigma)
    return Prob, Prob2


def evaluation(Prob, Prob2, sample):
    L = {}
    for c in Prob.keys():
        L[c] = Prob[c]
    for i in range(len(sample)):
        attrVal = sample[i]
        if type(attrVal).__name__ not in ('int', 'float'):
            for c in L.keys():
                L[c] *= Prob2[(attrVal, c)]
        else:
            for c in L.keys():
                prob_density_func = Prob2[(i, c)]
                L[c] *= prob_density_func(attrVal)
    max_prob, opti_c = 0.0, None
    for c in L.keys():
        if L[c] > max_prob:
            max_prob = L[c]
            opti_c = c
    print('Category: ', opti_c)


if __name__ == '__main__':
    dataset = load_data()
    Prob, Prob2 = build_bayes_classifier(dataset)
    sample = dataset[0][:-1]
    evaluation(Prob, Prob2, sample)
