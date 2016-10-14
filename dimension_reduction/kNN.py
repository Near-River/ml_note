#!/usr/bin/env python3
# coding=utf-8

"""
k近邻学习：k-Nearest Neighbor
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


def k_NN(X, sample, k=3):
    # pick out the nearest k samples from test sample in samples set X
    m, n = X.shape
    d = n - 1
    cls = list(set([sample[-1] for sample in X]))
    subdatasetMap = {}
    for i in range(m):
        c = X[i][-1]
        if c in subdatasetMap:
            subdatasetMap[c].append(X[i])
        else:
            subdatasetMap[c] = [X[i]]
    dists = []
    for i in range(m):
        dist = 0
        x_i = X[i, :]
        for j in range(d):
            attrVal = sample[j]
            if type(attrVal).__name__ not in ('int', 'float'):
                count1 = count2 = 0
                for _sample in X:
                    if _sample[j] == attrVal: count1 += 1
                    if _sample[j] == x_i[j]: count2 += 1
                for c in cls:
                    subcount1 = subcount2 = 0
                    subdataset = subdatasetMap[c]
                    for _sample in subdataset:
                        if _sample[j] == attrVal: subcount1 += 1
                        if _sample[j] == x_i[j]: subcount2 += 1
                    dist += pow(subcount1 / count1 - subcount2 / count2, 2)
            else:
                dist += pow(attrVal - x_i[j], 2)
        dists.append((i, dist))

    dists.sort(key=lambda dist: dist[1])
    neighbors = []
    # voting strategy
    voting = {}
    for c in cls: voting[c] = 0
    for i in range(k):
        no = dists[i][0]
        neighbors.append(no)
        c = X[no, -1]
        voting[c] += 1
    winner, votes = cls[0], voting[cls[0]]
    for i in range(1, len(cls)):
        if voting[cls[i]] > votes:
            winner = cls[i]
            votes = voting[cls[i]]
    print('The category of the sample is:', winner)
    return neighbors


def knn_plot(X, sample, neighbors):
    plt.figure(1)
    ax = plt.subplot(111)
    colors = ['red', 'yellow', 'blue']
    marks = ['s', 'o']

    xcord1, ycord1 = [], []
    xcord2, ycord2 = [], []
    for i in range(X.shape[0]):
        _sample = X[i]
        if i in neighbors:
            xcord2.append(_sample[-3])
            ycord2.append(_sample[-2])
        else:
            xcord1.append(_sample[-3])
            ycord1.append(_sample[-2])
    ax.scatter(xcord1, ycord1, s=30, c=colors[0], marker=marks[0])
    ax.scatter(xcord2, ycord2, s=30, c=colors[1], marker=marks[0])
    ax.scatter([sample[-3]], [sample[-2]], s=30, c=colors[2], marker=marks[1])
    plt.xlabel('Density')
    plt.ylabel('Sugar_ratio')
    plt.title('k-NN')
    plt.show()


if __name__ == '__main__':
    dataset = load_data()
    sample = dataset[0]
    X = dataset[1:]
    neighbors = k_NN(X, sample, k=3)
    knn_plot(X, sample, neighbors)
