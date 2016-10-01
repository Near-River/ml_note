#!/usr/bin/env python3
# coding=utf-8

"""
绘制决策树的工具类
"""
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# 计算树的叶子节点数量
def getLeavesNum(tree):
    leavesNum = 0
    firstKey = list(tree.keys())[0]
    subDict = tree[firstKey]
    for key in subDict.keys():
        if type(subDict[key]).__name__ == 'dict':
            leavesNum += getLeavesNum(subDict[key])
        else:
            leavesNum += 1
    return leavesNum


# 计算树的最大深度
def getTreeDepth(tree):
    maxDepth = 0
    firstKey = list(tree.keys())[0]
    subDict = tree[firstKey]
    for key in subDict.keys():
        maxDepth = max(maxDepth, 1)
        if type(subDict[key]).__name__ == 'dict':
            maxDepth = max(maxDepth, 1 + getTreeDepth(subDict[key]))
    return maxDepth


# 画节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction', va="center", ha="center",
                            bbox=nodeType, arrowprops=arrow_args)


# 画箭头上的文字
def plotMidText(cntrPt, parentPt, txtString):
    lens = len(txtString)
    xMid = (parentPt[0] + cntrPt[0]) / 2.0 - lens * 0.002
    yMid = (parentPt[1] + cntrPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(tree, parentPt, nodeTxt):
    leavesNum = getLeavesNum(tree)
    firstKey = list(tree.keys())[0]
    cntrPt = (plotTree.x0ff + (1.0 + float(leavesNum)) / 2.0 / plotTree.totalW, plotTree.y0ff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstKey, cntrPt, parentPt, decisionNode)
    subDict = tree[firstKey]
    plotTree.y0ff -= 1.0 / plotTree.totalD
    for key in subDict.keys():
        if type(subDict[key]).__name__ == 'dict':
            plotTree(subDict[key], cntrPt, str(key))
        else:
            plotTree.x0ff += 1.0 / plotTree.totalW
            plotNode(subDict[key], (plotTree.x0ff, plotTree.y0ff), cntrPt, leafNode)
            plotMidText((plotTree.x0ff, plotTree.y0ff), cntrPt, str(key))
    plotTree.y0ff += 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getLeavesNum(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.x0ff = -0.5 / plotTree.totalW
    plotTree.y0ff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
