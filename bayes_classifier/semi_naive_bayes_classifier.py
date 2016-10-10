#!/usr/bin/env python3
# coding=utf-8

"""
半朴素贝叶斯分类器（使用拉普拉斯修正）：西瓜数据集3.0
策略：AODE（Averaged One-Dependent Estimator）
"""
import numpy as np
import pandas as pd
from math import sqrt, exp, pow, pi

csv_file = '../decision_tree/watermelon3.0.csv'
threshold = 8


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
    subdatasetMap = {}
    for i in range(m):
        c = dataset[i][-1]
        if c in subdatasetMap:
            subdatasetMap[c].append(dataset[i])
        else:
            subdatasetMap[c] = [dataset[i]]
    AttrValsNumMap = {}
    AttrKindsNumMap = {}
    discreteAttrVals = []
    continuousAttrNo = []
    for i in range(n - 1):
        if type(dataset[0][i]).__name__ not in ('int', 'float'):
            uniqAttrVals = list(set([sample[i] for sample in dataset]))
            AttrKindsNumMap[i] = len(uniqAttrVals)
            for val in uniqAttrVals:
                discreteAttrVals.append((i, val))
                AttrValsNumMap[val] = 0
                for sample in dataset:
                    if sample[i] == val: AttrValsNumMap[val] += 1
        else:
            continuousAttrNo.append(i)
    Prob = {}  # P[x_i, c]
    Prob2 = {}  # P[x_j | c, x_i]
    for c in cls:
        for no, x_i in discreteAttrVals:
            if AttrValsNumMap[x_i] >= threshold:
                count = 0
                subdataset = []
                for sample in subdatasetMap[c]:
                    if sample[no] == x_i:
                        subdataset.append(sample)
                        count += 1
                Prob[(x_i, c)] = (count + 1) / (m + AttrKindsNumMap[no])

                for no2, x_j in discreteAttrVals:
                    count2 = 0
                    for sample in subdataset:
                        if sample[no2] == x_j: count2 += 1
                    Prob2[(x_j, c, x_i)] = (1 + count2) / (count + AttrKindsNumMap[no2])

                for no3 in continuousAttrNo:
                    u, sigma = 0.0, 0.0
                    valsLst = []
                    for sample in subdataset:
                        valsLst.append(sample[no3])
                        u += sample[no3]
                    u /= len(subdataset)
                    for val in valsLst:
                        sigma += (val - u) ** 2
                    sigma = sqrt(sigma / len(subdataset))
                    Prob2[(no3, c, x_i)] = normal_distribution_func(u, sigma)

    return cls, Prob, Prob2, AttrValsNumMap


def evaluation(cls, Prob, Prob2, AttrValsNumMap, sample):
    d = len(sample)
    L = {}
    for c in cls: L[c] = 0.0
    for i in range(d):
        x_i = sample[i]
        if type(x_i).__name__ not in ('int', 'float') and AttrValsNumMap[x_i] >= threshold:
            for c in L.keys():
                prob = Prob[(x_i, c)]
                for j in range(d):
                    x_j = sample[j]
                    if type(x_j).__name__ not in ('int', 'float'):
                        prob *= Prob2[(x_j, c, x_i)]
                    else:
                        prob_density_func = Prob2[(j, c, x_i)]
                        prob *= prob_density_func(x_j)
                L[c] += prob

    max_prob, opti_c = 0.0, None
    for c in L.keys():
        print(c, L[c])
        if L[c] > max_prob:
            max_prob = L[c]
            opti_c = c
    print('Category: ', opti_c)


if __name__ == '__main__':
    dataset = load_data()
    cls, Prob, Prob2, AttrValsNumMap = build_bayes_classifier(dataset)
    sample = dataset[1][:-1]
    evaluation(cls, Prob, Prob2, AttrValsNumMap, sample)
