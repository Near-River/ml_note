#!/usr/bin/env python3
# coding=utf-8

"""
CART(classification and regression tree)决策树 使用基尼指数（Gini_Index）来选择划分属性
西瓜数据集
属性：色泽   根蒂 	 敲声 	  纹理     脐部 	触感   密度 	 含糖率 	    标签
      color  root  knocks  texture  navel  touch  density  sugar_ratio  label
"""
import pandas as pd
from math import log2
from decision_tree.plot_tree_tools import *


class CART_Algorithm(object):
    class TreeNode(object):
        def __init__(self, cls=None, val=None, kind=None):
            self.cls = cls  # leaf node: cls | intermediate node: label
            self.val = val  # (previous node) attribute's value
            self.kind = kind  # decision node & leaf node
            self.children = []

        def __str__(self):
            return '%s, %s, %s' % (self.kind, self.val, self.cls)

        def printSelf(self):
            print(self.kind, self.val, self.cls)
            if self.kind != 'leaf':
                for c in self.children: c.printSelf()

    def __init__(self, csv_file):
        self.loadData(csv_file)

    def loadData(self, csv_file):
        df = pd.read_csv(csv_file, encoding="utf-8")
        self.dataset = df.values[:10, 1:].tolist()
        self.labels = df.columns.values[1:-1].tolist()
        self.validationSet = df.values[10:, 1:].tolist()

    def treeFactory(self):
        self.root = self.TreeNode()
        self.AttrsMap = {}
        for i in range(len(self.labels)):
            label = self.labels[i]
            attrValues = [sample[i] for sample in self.dataset]
            self.AttrsMap[label] = set(attrValues)
        self.inOrderNodes = []  # recorded decision tree nodes in the middle order
        self.generateTree(self.root, self.dataset[:], self.labels[:])  # no-pruning
        # self.generateTree2(self.root, self.dataset[:], self.labels[:])  # pre-pruning
        # self.generateTree3()  # post-pruning
        self.root.printSelf()

    def generateTree(self, root, dataset, labels):
        self.inOrderNodes.append(root)
        # case one: all the samples of dataset belong to the same classification C
        classLst = [sample[-1] for sample in dataset]
        if classLst.count(classLst[0]) == len(classLst):
            root.cls = classLst[0]
            root.kind = 'leaf'
            return
        # case two: attributes set A of dataset is nil OR all samples of dataset have the same value in A
        if len(dataset[0]) == 1 or self.isSameOnAllAttrs(dataset):
            root.cls = self.majorClsCnt(classLst)
            root.kind = 'leaf'
            return

        # select the optimal partition attribute
        optiAttr = self.selectOptimalAttr(dataset, labels)
        optiAttrLabel = labels[optiAttr]
        # recursively generate the decision tree
        attrValues = [sample[optiAttr] for sample in dataset]
        uniqueVals = set(attrValues)
        del labels[optiAttr]

        root.kind = 'intermediate'
        root.cls = optiAttrLabel
        for value in uniqueVals:
            node = self.TreeNode()
            root.children.append(node)
            node.val = value
            subdataset = self.splitDataSet(dataset, optiAttr, value)
            self.generateTree(node, subdataset, labels[:])
        for value in self.AttrsMap[optiAttrLabel]:
            if value not in uniqueVals:
                node = self.TreeNode()
                root.children.append(node)
                node.val = value
                node.cls = self.majorClsCnt(classLst)
                node.kind = 'leaf'

    def generateTree2(self, root, dataset, labels):
        # case one: all the samples of dataset belong to the same classification C
        classLst = [sample[-1] for sample in dataset]
        if classLst.count(classLst[0]) == len(classLst):
            root.cls = classLst[0]
            root.kind = 'leaf'
            return
        # case two: attributes set A of dataset is nil OR all samples of dataset have the same value in A
        if len(dataset[0]) == 1 or self.isSameOnAllAttrs(dataset):
            root.cls = self.majorClsCnt(classLst)
            root.kind = 'leaf'
            return

        # select the optimal partition attribute
        optiAttr = self.selectOptimalAttr(dataset, labels)
        optiAttrLabel = labels[optiAttr]
        # recursively generate the decision tree
        attrValues = [sample[optiAttr] for sample in dataset]
        uniqueVals = set(attrValues)
        del labels[optiAttr]

        root.kind = 'intermediate'
        root.cls = optiAttrLabel

        # Pre-pruning strategy:
        root.kind = 'leaf'
        root.cls = self.majorClsCnt(classLst)
        prePartitionAccuracy = self.calcAccuracy(self.root)  # pre-partition accuracy

        root.kind = 'intermediate'
        root.cls = optiAttrLabel
        valueToDataset = {}
        for value in uniqueVals:
            node = self.TreeNode()
            root.children.append(node)
            node.val = value
            subdataset = self.splitDataSet(dataset, optiAttr, value)
            valueToDataset[value] = (subdataset, node)
            subClassLst = [sample[-1] for sample in subdataset]
            node.kind = 'leaf'
            node.cls = self.majorClsCnt(subClassLst)
        for value in self.AttrsMap[optiAttrLabel]:
            if value not in uniqueVals:
                node = self.TreeNode()
                root.children.append(node)
                node.val = value
                node.cls = self.majorClsCnt(classLst)
                node.kind = 'leaf'
        postPartitionAccuracy = self.calcAccuracy(self.root)  # post-partition accuracy

        if prePartitionAccuracy > postPartitionAccuracy:
            root.kind = 'leaf'
            root.cls = self.majorClsCnt(classLst)
            root.children = []
        else:
            for value in uniqueVals:
                subdataset, node = valueToDataset[value]
                self.generateTree(node, subdataset, labels[:])

    def generateTree3(self):
        self.generateTree(self.root, self.dataset[:], self.labels[:])
        for i in range(len(self.inOrderNodes) - 1, -1, -1):
            node = self.inOrderNodes[i]
            if node.kind == 'intermediate':
                label = node.cls
                prePruningAccuracy = self.calcAccuracy(self.root)
                node.cls = self.calcSameSamplesCls(node)
                node.kind = 'leaf'
                postPruningAccuracy = self.calcAccuracy(self.root)
                if prePruningAccuracy >= postPruningAccuracy:
                    node.kind = 'intermediate'  # no needing for pruning, recover the node
                    node.cls = label

    def calcSameSamplesCls(self, node):
        """ calculate the samples number subject to the relevant judgement conditions.
                conditions: denote by nodes along the way from root down to node.
        """
        subdataset = []

        def match(root, sample):
            if root.kind == 'intermediate':
                if root == node: return True
                attrIdx = self.labels.index(root.cls)
                for _node in root.children:
                    if _node.val == sample[attrIdx]:
                        return match(_node, sample)
            return False

        for sample in self.dataset:
            if match(self.root, sample):
                subdataset.append(sample)
        classLst = [sample[-1] for sample in subdataset]
        return self.majorClsCnt(classLst)

    def calcAccuracy(self, root):
        """ calculate the accuracy of the tree(based on the validation set). """
        corr_count = 0

        def validate(root, sample):
            if root.kind == 'intermediate':
                label = root.cls
                attrIdx = self.labels.index(label)
                for node in root.children:
                    if node.val == sample[attrIdx]:
                        return validate(node, sample)
            else:
                return root.cls == sample[-1]

        for sample in self.validationSet:
            if validate(root, sample): corr_count += 1
        return corr_count / len(self.validationSet) * 100.0

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

    def calcGini(self, dataset):
        """ Calculate the Gini value. """
        samplesCnt = len(dataset)
        clsCount = {}
        for sample in dataset:
            cls = sample[-1]
            if cls not in clsCount: clsCount[cls] = 0
            clsCount[cls] += 1
        gini = 1.0
        for count in clsCount.values():
            prob = float(count) / samplesCnt
            gini -= prob * prob
        return gini

    def splitDataSet(self, dataset, attrNo, value):
        """ Divide the dataset for the discrete variables, and take out all the samples whose value equals value. """
        subdataset = []
        for sample in dataset:
            if sample[attrNo] == value:
                reducedsample = sample[:attrNo]
                reducedsample.extend(sample[attrNo + 1:])
                subdataset.append(reducedsample)
        return subdataset

    def selectOptimalAttr(self, dataset, labels):
        """ The optimal attribute partition selection. """
        attrsNum = len(dataset[0]) - 1  # attributes number
        bestGainIndex = 1000.0  # the best Gini_Index respect to attribute ai(ai belongs to A)
        optiAttr = -1  # the No. of the optimal partition attribute
        for i in range(attrsNum):
            attrValues = [sample[i] for sample in dataset]
            uniqueVals = set(attrValues)
            giniIndex = 0.0
            for value in uniqueVals:
                subdataset = self.splitDataSet(dataset, i, value)
                prob = float(len(subdataset)) / len(dataset)
                giniIndex += (prob * self.calcGini(subdataset))
            if giniIndex < bestGainIndex:
                bestGainIndex = giniIndex
                optiAttr = i
        return optiAttr


if __name__ == '__main__':
    cart = CART_Algorithm(csv_file='watermelon2.0.csv')
    cart.treeFactory()
