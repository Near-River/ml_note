#!/usr/bin/env python3
# coding=utf-8

"""
使用线性核和高斯核分别训练一个 SVM 模型：数据集（西瓜数据集3.0）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm

csv_file = '../decision_tree/watermelon3.0.csv'


def load_data():
    df = pd.read_csv(csv_file, encoding="utf-8")
    X = np.array(df.values[:, -3:-1])
    labels = df['label'].values[:].tolist()
    for i in range(len(labels)):
        if labels[i] == 0: labels[i] = -1
    y = np.array(labels)
    return X, y


def operation(X, y):
    # clf = svm.SVC(kernel='linear')
    # clf.fit(X, y)
    # print(clf.support_vectors_)
    # print(clf.predict([[0.719, 0.103]]))

    lin_svc = svm.LinearSVC().fit(X, y)
    rbf_svc = svm.SVC(kernel='rbf', C=1e3, gamma=0.5).fit(X, y)
    # create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # title for the plots
    titles = ['LinearSVC (linear kernel)', 'SVC with RBF kernel']
    c = []
    for i in y:
        c.append('red') if i == 1 else c.append('blue')
    for i, clf in enumerate((lin_svc, rbf_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max] x [y_min, y_max].
        plt.subplot(1, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=c, cmap=plt.cm.Paired)
        plt.xlabel('Density')
        plt.ylabel('Sugar_rate')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])
    plt.show()


if __name__ == '__main__':
    X, y = load_data()
    operation(X, y)
