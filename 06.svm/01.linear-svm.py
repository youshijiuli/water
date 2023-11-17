#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   01.linear-svm.py
@Author  :   Cat 
@Version :   3.11
"""

import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():  # 逐行读取，滤除空格等
        lineArr = line.strip().split("\t")
        dataMat.append([float(lineArr[0]), float(lineArr[1])])  # 添加数据
        labelMat.append(float(lineArr[2]))  # 添加标签
    return dataMat, labelMat


def showDataSet(dataMat, labelMat):
    # dataMat  二维数组，一行就是一个样本
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(dataMat)):
        # label只有两个选择 1，-1
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np = np.array(data_plus)  # 转换为numpy矩阵
    data_minus_np = np.array(data_minus)  # 转换为numpy矩阵
    plt.scatter(np.transpose(data_plus_np)[0], np.transpose(data_plus_np)[1])  # 正样本散点图
    plt.scatter(
        np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1]
    )  # 负样本散点图
    plt.show()


if __name__ == "__main__":
    dataMat, labelMat = loadDataSet("testSet.txt")
    showDataSet(dataMat, labelMat)
