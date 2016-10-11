#!/usr/bin/env python3
# coding=utf-8

"""
原型聚类算法：K-means
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

xlsx_file = 'watermelon4.0.xlsx'


def load_data():
    df = pd.read_excel(xlsx_file)
    return df.values


def k_means(X, k=3):
    m = X.shape[0]
    clusters = {}
    cluster_tag = [0 for _ in range(m)]
    mean_vectors = {}
    used = set()
    for i in range(k):  # randomly initialize the mean_vector for clusters
        while True:
            rand = random.randint(0, m - 1)
            if rand in used: continue
            used.add(rand)
            mean_vectors[i + 1] = X[rand]
            break
    # print(mean_vectors)
    iters = 0
    while True:
        iters += 1
        count = 0
        clusters.clear()
        for i in range(k): clusters[i + 1] = []
        for j in range(m):
            x_j = X[j, :]
            min_dist = 10000
            for i in range(1, k + 1):
                dist = np.sum((x_j - mean_vectors[i]) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    cluster_tag[j] = i
            clusters[cluster_tag[j]].append(x_j)
        for i in range(1, k + 1):
            temp = np.array([0.0, 0.0])
            for x in clusters[i]: temp += x
            mean_vector = 1 / len(clusters[i]) * temp
            if not np.equal(mean_vectors[i], mean_vector).all():
                mean_vectors[i] = mean_vector
            else:
                count += 1
        if count == k: break  # reach the stop condition
    # print('Iters:', iters)
    return clusters


def cluster_plot(clusters):
    plt.figure(1)
    ax = plt.subplot(111)
    i = 0
    colors = ['red', 'yellow', 'blue']
    marks = ['s', 'o', 'x']
    for cluster in clusters.values():
        xcord, ycord = [], []
        for x in cluster:
            xcord.append(x[0])
            ycord.append(x[1])
        ax.scatter(xcord, ycord, s=30, c=colors[i], marker=marks[i])
        i += 1
    plt.xlabel('Density')
    plt.ylabel('Sugariness')
    plt.title('K-means clustering')
    plt.show()


if __name__ == '__main__':
    X = load_data()
    clusters = k_means(X, k=3)
    cluster_plot(clusters)
