#!/usr/bin/env python3
# coding=utf-8

"""
训练一个 SVR 模型：数据集（西瓜数据集3.0）
    以‘密度’为输入，‘含糖率’为输出
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

csv_file = '../decision_tree/watermelon3.0.csv'


def load_data():
    df = pd.read_csv(csv_file, encoding="utf-8")
    X = np.array([[i] for i in df.values[:, -3]])
    y = np.array(df.values[:, -2])
    return X, y


def operation(X, y):
    svr_rbf = svm.SVR(kernel='rbf', C=1e3, gamma=10)
    X_rbf = np.arange(min(X), max(X), 0.01)
    y_rbf = svr_rbf.fit(X, y).predict([[i] for i in X_rbf])
    # Plot the training points
    plt.scatter(X, y, c='red', label='raw_data')
    plt.hold('on')
    # Plot the regression model
    plt.plot(X_rbf, y_rbf, color='navy', lw=2, label='RBF model')
    plt.xlabel('Density')
    plt.ylabel('Sugar_rate')
    plt.title('RBF_SVR')
    plt.show()


if __name__ == '__main__':
    X, y = load_data()
    operation(X, y)
