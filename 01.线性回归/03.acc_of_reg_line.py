#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   03.acc_of_reg_line.py
@Author  :   Cat 
@Version :   3.11
"""

"""如何判断拟合曲线的拟合效果的如何呢？当然，我们可以根据自己的经验进行观察，
除此之外，我们还可以使用corrcoef方法，来比较预测值和真实值的相关性。编写代码如下："""


# -*- coding:utf-8 -*-
import numpy as np


def loadDataSet(fileName):
    """
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        xArr - x数据集
        yArr - y数据集
    Website:
        https://www.cuijiahua.com/
    Modify:
        2017-11-12
    """
    numFeat = len(open(fileName).readline().split("\t")) - 1
    xArr = []
    yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


def standRegres(xArr, yArr):
    """
    函数说明:计算回归系数w
    Parameters:
        xArr - x数据集
        yArr - y数据集
    Returns:
        ws - 回归系数
    Website:
        https://www.cuijiahua.com/
    Modify:
        2017-11-12
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat  # 根据文中推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


if __name__ == "__main__":
    xArr, yArr = loadDataSet("ex0.txt")  # 加载数据集
    ws = standRegres(xArr, yArr)  # 计算回归系数
    xMat = np.mat(xArr)  # 创建xMat矩阵
    yMat = np.mat(yArr)  # 创建yMat矩阵
    yHat = xMat * ws
    print(np.corrcoef(yHat.T, yMat))


"""[[1.         0.98647356]
 [0.98647356 1.        ]]"""