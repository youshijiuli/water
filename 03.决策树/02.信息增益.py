#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   02.信息增益.py
@Author  :   Cat 
@Version :   3.11
"""


import math
from utils import createDataSet


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
    return shannonEnt  # 返回经验熵(香农熵)


"""
    使用了三个输人参数:待划分的数据集、划分数据集的特征、特征的返回值。
    需要注意的是，Python语言不用考虑内存分配问题。
    Python语言在函数中传递的是列表的引用，在函数内部对列表对象的修改，将会影响该列表对象的整个生存周期。
    为了消除这个不良影响，我们需要在函数的开始声明一个新列表对象。
    因为该函数代码在同一数据集上被调用多次为了不修改原始数据集，创建一个新的列表对象0。
    数据集这个列表中的各个元素也是列表，我们要遍历数据集中的每个元素，一旦发现符合要求的值，
    则将其添加到新创建的列表中。在if语句中，程序将符合特征的数据抽取出来2。
    后面讲述得更简单，这里我们可以这样理解这段代码:当我们按照某个特征划分数据集时，
    就需要将所有符合要求的元素抽取出来。代码中使用了Python语言列表类型自带的extend()和appen()方法。
    这两个方法功能类似，但是在处理多个列表时，这两个方法的处理结果是完全不同的。
"""


def splitDataSet(dataSet, axis, value):
    """Parameters:
    dataSet - 待划分的数据集
    axis - 划分数据集的特征
    value - 需要返回的特征的值
    """
    return_dataset = []
    for item in dataSet:
        if item[axis] == value:
            reduceFeatVec = item[:axis]
            reduceFeatVec.extend(item[axis + 1 :])
            return_dataset.append(reduceFeatVec)

    return return_dataset


# 选择最好的数据划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseWntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseWntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


if __name__ == "__main__":
    dataset, features = createDataSet()
    print(f"最优特征值是：{chooseBestFeatureToSplit(dataset)}")

# splitDataSet函数是用来选择各个特征的子集的，比如选择年龄(第0个特征)的青年(用0代表)的自己，
# 我们可以调用splitDataSet(dataSet,0,0)这样返回的子集就是年龄为青年的5个数据集。
# chooseBestFeatureToSplit是选择选择最优特征的函数。运行代码结果如下：



# 对比我们自己计算的结果，发现结果完全正确！最优特征的索引值为2，也就是特征A3(有自己的房子)。