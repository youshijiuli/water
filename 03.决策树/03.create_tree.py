#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   03.create_tree.py
@Author  :   Cat 
@Version :   3.11
"""

import math
import operator
from utils import createTreeDataSet


# 我们使用字典存储决策树的结构，比如上小节我们分析出来的决策树，用字典可以表示为：
# {'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}

# 创建函数majorityCnt统计classList中出现此处最多的元素(类标签)，创建函数createTree用来递归构建决策树。编写代码如下：


# 计算香农熵
def calcShannonEnt(dataSet):
    numEntires = len(dataSet)  # 返回数据集的行数
    labelCounts = {}  # 保存每个标签(Label)出现次数的字典
    for featVec in dataSet:  # 对每组特征向量进行统计
        currentLabel = featVec[-1]  # 提取标签(Label)信息
        if currentLabel not in labelCounts.keys():  # 如果标签(Label)没有放入统计次数的字典,添加进去
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1  # Label计数
    shannonEnt = 0.0  # 经验熵(香农熵)
    for key in labelCounts:  # 计算香农熵
        prob = float(labelCounts[key]) / numEntires  # 选择该标签(Label)的概率
        shannonEnt -= prob * math.log(prob, 2)  # 利用公式计算
    return shannonEnt


# def splitDataSet(dataSet, axis, value):
#     retDataSet = []                                        #创建返回的数据集列表
#     for featVec in dataSet:                             #遍历数据集
#         if featVec[axis] == value:
#             reducedFeatVec = featVec[:axis]                #去掉axis特征
#             reducedFeatVec.extend(featVec[axis+1:])     #将符合条件的添加到返回的数据集
#             retDataSet.append(reducedFeatVec)
#     return retDataSet                                      #返回划分后的数据集
def splitDataSet(dataSet, axis, value):
    result = []

    for feature in dataSet:
        if feature[axis] == value:
            reduceFeatureVec = feature[:axis]
            reduceFeatureVec.extend(feature[axis + 1 :])
            result.append(reduceFeatureVec)

    return result


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    bestInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征的索引值
    for i in range(numFeatures):  # 遍历所有特征
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  # 创建set集合{},元素不可重复
        newEntropy = 0.0  # 经验条件熵
        for value in uniqueVals:  # 计算信息增益
            subDataSet = splitDataSet(dataSet, i, value)  # subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy  # 信息增益
        # print("第%d个特征的增益为%.3f" % (i, infoGain))            #打印每个特征的信息增益
        if infoGain > bestInfoGain:  # 计算信息增益
            bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature  # 返回信息增益最大的特征的索引值


"""
函数说明:统计classList中出现此处最多的元素(类标签)
 
Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素(类标签)
Author:
    Jack Cui
Blog:
    http://blog.csdn.net/c406495762
Modify:
    2017-07-24
"""


def majorityCnt(classList):
    classCount = {}
    for vote in classList:  # 统计classList中每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True
    )  # 根据字典的值降序排序
    return sortedClassCount[0][0]


def createTree(dataset, labels, featLabels):
    # 1. #取分类标签(是否放贷:yes or no)
    classList = [example[-1] for example in dataset]

    # #如果类别完全相同则停止继续划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # #遍历完所有特征时返回出现次数最多的类标签
    if len(dataset[0]) == 1 or len(labels) == 0:
        return majorityCnt(classList)

    # 选择最优的特征
    bestFeature = chooseBestFeatureToSplit(dataset)
    # print(bestFeature, "============================")

    # 最后话特征标签
    bestFeatLabel = labels[bestFeature]
    featLabels.append(bestFeatLabel)

    # 根据最有标签生成树
    myTree = {bestFeatLabel: {}}
    # 删除已经使用特征标签
    del labels[bestFeature]

    # # 得到训练集中所有最优特征的属性值
    featureValue = [example[bestFeature] for example in dataset]
    # # 去掉重复的属性值
    uniqueVals = set(featureValue)
    # # #遍历特征，创建决策树。
    for value in uniqueVals:
        subLabel = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataset, bestFeature, value), subLabel, featLabels
        )

    return myTree


# def createTree(dataSet, labels, featLabels):
#     classList = [example[-1] for example in dataSet]  # 取分类标签(是否放贷:yes or no)
#     if classList.count(classList[0]) == len(classList):  # 如果类别完全相同则停止继续划分
#         return classList[0]
#     if len(dataSet[0]) == 1 or len(labels) == 0:  # 遍历完所有特征时返回出现次数最多的类标签
#         return majorityCnt(classList)
#     bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
#     bestFeatLabel = labels[bestFeat]  # 最优特征的标签
#     featLabels.append(bestFeatLabel)
#     myTree = {bestFeatLabel: {}}  # 根据最优特征的标签生成树
#     del labels[bestFeat]  # 删除已经使用特征标签
#     featValues = [example[bestFeat] for example in dataSet]  # 得到训练集中所有最优特征的属性值
#     uniqueVals = set(featValues)  # 去掉重复的属性值
#     for value in uniqueVals:  # 遍历特征，创建决策树。
#         subLabels = labels[:]
#         myTree[bestFeatLabel][value] = createTree(
#             splitDataSet(dataSet, bestFeat, value), subLabels, featLabels
#         )
#     return myTree


def main():
    dataset, labels = createTreeDataSet()
    # print(labels, "labels")
    featureLabels = []
    tree = createTree(dataset, labels, featureLabels)
    print(tree)
    # return tree


if __name__ == "__main__":
    main()

"""递归创建决策树时，递归有两个终止条件：第一个停止条件是所有的类标签完全相同，则直接返回该类标签；第二个停止条件是使用完了所有特征，仍然不能将数据划分仅包含唯一类别的分组，即决策树构建失败，特征不够用。此时说明数据纬度不够，由于第二个停止条件无法简单地返回唯一的类标签，这里挑选出现数量最多的类别作为返回值。

运行上述代码，我们可以看到如下结果：
"""

# {'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
