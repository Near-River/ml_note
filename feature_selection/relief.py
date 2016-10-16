#!/usr/bin/env python3
# coding=utf-8

"""
Relief：Relevant Features
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt, pow

csv_file = '../decision_tree/watermelon3.0.csv'


def load_data():
    df = pd.read_csv(csv_file, encoding="utf-8")
    dataset = df.values[:, 1:]
    return dataset


def relief(dataset, k):
    m, n = dataset.shape
    d = n - 1
    posdataset = []
    negdataset = []
    for i in range(m):
        c = dataset[i][-1]
        if c == 1:
            posdataset.append(dataset[i])
        else:
            negdataset.append(dataset[i])
    Dist = np.zeros((m, m))  # distance matrix
    for i in range(m):
        x_i = dataset[i]
        for j in range(m):
            x_j = dataset[j]
            dist = 0.0
            for t in range(d):
                if type(x_i[t]).__name__ in ('int', 'float'):
                    dist += pow(x_i[t] - x_j[t], 2)
                else:
                    val1, val2 = x_i[t], x_j[t]
                    count1 = count2 = 0
                    for _sample in dataset:
                        if _sample[t] == val1: count1 += 1
                        if _sample[t] == val2: count2 += 1
                    subcount1 = subcount2 = 0
                    for _sample in posdataset:
                        if _sample[t] == val1: subcount1 += 1
                        if _sample[t] == val2: subcount2 += 1
                    dist += pow(subcount1 / count1 - subcount2 / count2, 2)
                    subcount1 = subcount2 = 0
                    for _sample in negdataset:
                        if _sample[t] == val1: subcount1 += 1
                        if _sample[t] == val2: subcount2 += 1
                    dist += pow(subcount1 / count1 - subcount2 / count2, 2)
            Dist[i, j] = sqrt(dist)

    relevant_statistic_lst = []  # record the relevant statistical variables

    for j in range(d):
        delta_j = 0.0
        for i in range(m):
            x_i = dataset[i]
            c = x_i[-1]
            x_i_nh = None  # near hit
            x_i_nm = None  # near miss
            pos_dist = neg_dist = 1000.0
            for t in range(m):
                sample = dataset[t]
                if i == t: continue
                if Dist[i, t] < pos_dist and sample[-1] == c:
                    x_i_nh = sample
                    pos_dist = Dist[i, t]
                if Dist[i, t] < neg_dist and sample[-1] != c:
                    x_i_nm = sample
                    neg_dist = Dist[i, t]
            if type(x_i[j]).__name__ in ('int', 'float'):
                delta_j += pow(x_i_nm[j] - x_i[j], 2) - pow(x_i_nh[j] - x_i[j], 2)
            else:
                diff1 = 1 if x_i[j] != x_i_nh[j] else 0
                diff2 = 1 if x_i[j] != x_i_nm[j] else 0
                delta_j += diff2 - diff1
        relevant_statistic_lst.append((j, delta_j / m))

    relevant_statistic_lst.sort(key=lambda d: d[1], reverse=True)
    for i in range(k):
        print(relevant_statistic_lst[i])


if __name__ == '__main__':
    dataset = load_data()
    relief(dataset, k=3)
