#!/usr/bin/env python3
# coding=utf-8

"""
ID3决策树(基于信息熵进行划分选择的决策树算法)
西瓜数据集 3.0
属性：色泽   根蒂 	 敲声 	  纹理     脐部 	触感   密度 	 含糖率 	    标签
      color  root  knocks  texture  navel  touch  density  sugar_ratio  label
"""
import pandas as pd
from math import log2
from decision_tree.plot_tree_tools import *


class ID3_Algorithm(object):
    def __init__(self, csv_file):
        self.loadData(csv_file)

    def loadData(self, csv_file):
        df = pd.read_csv(csv_file, encoding="utf-8")
        self.dataset = df.values[:, 1:].tolist()
        self.labels = df.columns.values[1:-1].tolist()

    def treeFactory(self):
        self.AttrsMap = {}
        for i in range(len(self.labels)):
            label = self.labels[i]
            if type(self.dataset[0][i]).__name__ not in ('float', 'int'):
                attrValues = [sample[i] for sample in self.dataset]
                self.AttrsMap[label] = set(attrValues)

        return self.generateTree(self.dataset[:], self.labels[:])

    def generateTree(self, dataset, labels):
        # case one: all the samples of dataset belong to the same classification C
        classLst = [sample[-1] for sample in dataset]
        if classLst.count(classLst[0]) == len(classLst): return classLst[0]
        # case two: attributes set A of dataset is nil OR all samples of dataset have the same value in A
        if len(dataset[0]) == 1 or self.isSameOnAllAttrs(dataset):
            return self.majorClsCnt(classLst)  # return the class with the largest number of samples in dataset

        # select the optimal partition attribute
        optiAttr = self.selectOptimalAttr(dataset, labels)
        optiAttrLabel = labels[optiAttr]

        # recursively generate the decision tree
        tree = {optiAttrLabel: {}}
        attrValues = [sample[optiAttr] for sample in dataset]
        uniqueVals = set(attrValues)
        del labels[optiAttr]
        for value in uniqueVals:
            sublabels = labels[:]
            subdataset = self.splitDataSet(dataset, optiAttr, value)
            tree[optiAttrLabel][value] = self.generateTree(subdataset, sublabels)

        if optiAttrLabel.find('<=') != -1:  # the optimal partition attribute is a continuous feature
            if 0 not in uniqueVals: tree[optiAttrLabel][0] = self.majorClsCnt(classLst)
            if 1 not in uniqueVals: tree[optiAttrLabel][1] = self.majorClsCnt(classLst)
        else:  # the optimal partition attribute is a discrete feature
            for value in self.AttrsMap[optiAttrLabel]:
                if value not in uniqueVals:
                    tree[optiAttrLabel][value] = self.majorClsCnt(classLst)
        return tree

    def majorClsCnt(self, classLst):
        """ all the features have been divided, we need voting class category for the samples.
                subject to cls with the largest number of samples in dataset
        """
        clsCount = {}
        for cls in classLst:
            if cls not in clsCount: clsCount[cls] = 0
            clsCount[cls] += 1
        # return (list(clsCount.keys()))[(list(clsCount.values())).index(max(clsCount.values()))]
        maxCount = max(clsCount.values())
        for cls in clsCount.keys():
            if clsCount[cls] == maxCount: return cls

    def isSameOnAllAttrs(self, dataset):
        """ check whether all samples of dataset have the same value in A """
        for i in range(len(dataset[0]) - 1):
            attrValues = [sample[i] for sample in dataset]
            if len(set(attrValues)) > 1: return False
        return True

    def calcEntropy(self, dataset):
        """ Calculate the information entropy. """
        samplesCnt = len(dataset)
        clsCount = {}
        for sample in dataset:
            cls = sample[-1]
            if cls not in clsCount: clsCount[cls] = 0
            clsCount[cls] += 1
        entropy = 0.0
        for count in clsCount.values():
            prob = float(count) / samplesCnt
            entropy -= (prob * log2(prob))
        return entropy

    def splitDataSet(self, dataset, attrNo, value):
        """ Divide the dataset for the discrete variables, and take out all the samples whose value equals value. """
        subdataset = []
        for sample in dataset:
            if sample[attrNo] == value:
                reducedsample = sample[:attrNo]
                reducedsample.extend(sample[attrNo + 1:])
                subdataset.append(reducedsample)
        return subdataset

    def splitContinuousDataSet(self, dataset, attrNo, value, direction):
        """ Divide the dataset for the continuous data
                direction = negative(0) | positive(1)
        """
        subdataset = []
        for sample in dataset:
            if direction == 0:  # contrary to the conditions: <= value
                if sample[attrNo] > value:
                    reducedsample = sample[:attrNo]
                    reducedsample.extend(sample[attrNo + 1:])
                    subdataset.append(reducedsample)
            else:
                if sample[attrNo] <= value:
                    reducedsample = sample[:attrNo]
                    reducedsample.extend(sample[attrNo + 1:])
                    subdataset.append(reducedsample)
        return subdataset

    def selectOptimalAttr(self, dataset, labels):
        """ The optimal attribute partition selection.
                denote attribute set A (A={a1, a2, ... , am})
        """
        attrsNum = len(dataset[0]) - 1  # attributes number
        # calculate the information entropy
        entropy = self.calcEntropy(dataset)
        bestInfoGain = -1000.0  # the best information gain respect to attribute ai(ai belongs to A)
        optiAttr = -1  # the No of the optimal partition attribute
        optiSplitVal = 0  # the value of the optimal partition point for continuous attribute

        for i in range(attrsNum):
            attrValues = [sample[i] for sample in dataset]
            if type(attrValues[0]).__name__ in ('float', 'int'):
                # deal with the continuous feature
                sortedAttrVals = sorted(attrValues)
                infoGain = -1000.0
                # build candidate partition set: size = len(attrValues) - 1
                n, splitLst = len(sortedAttrVals), []
                for j in range(n - 1):
                    splitLst.append((sortedAttrVals[j] + sortedAttrVals[j + 1]) / 2.0)
                for j in range(n - 1):
                    value = splitLst[j]
                    # calculate the information gain when the jth candidate partition is used
                    splitInfoGain = entropy
                    # dividing dataset into two sub dataset
                    negsubdataset = self.splitContinuousDataSet(dataset, i, value, 0)  # negative sub dataset
                    possubdataset = self.splitContinuousDataSet(dataset, i, value, 1)  # positive sub dataset
                    splitInfoGain -= float(len(negsubdataset)) / len(dataset) * self.calcEntropy(negsubdataset)
                    splitInfoGain -= float(len(possubdataset)) / len(dataset) * self.calcEntropy(possubdataset)
                    if splitInfoGain > infoGain:
                        infoGain = splitInfoGain  # update the infoGain
                        optiSplitVal = max(optiSplitVal, value)
            else:
                # deal with the distinct feature
                uniqueVals = set(attrValues)
                infoGain = entropy
                for value in uniqueVals:
                    subdataset = self.splitDataSet(dataset, i, value)
                    prob = float(len(subdataset)) / len(dataset)
                    infoGain -= (prob * self.calcEntropy(subdataset))
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                optiAttr = i

        """If the best partition feature of the current node is continuous feature, then
            it is binarized with the previously recorded division point as the boundary.
        """
        if type(dataset[0][optiAttr]).__name__ in ('float', 'int'):
            for sample in dataset:
                sample[optiAttr] = 1 if sample[optiAttr] <= optiSplitVal else 0
            labels[optiAttr] = labels[optiAttr] + '<=' + str(optiSplitVal)
        return optiAttr


if __name__ == '__main__':
    id3 = ID3_Algorithm(csv_file='watermelon3.0.csv')
    tree = id3.treeFactory()
    createPlot(tree)