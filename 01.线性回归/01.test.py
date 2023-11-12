#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   01.test.py
@Author  :   Cat 
@Version :   3.11
"""

import numpy as np
from matplotlib import pyplot as plt

# 1.加载数据集


def loadDataSet(filepath):
    xArr = []
    yArr = []
    fr = open(filepath)
    for line in fr.readlines():
        # print(line.strip().split('\t'))
        # print(line[1],type(line[1]))
        # print(float(line[1]))
        line = line.strip().split("\t")
        xArr.append(float(line[1]))
        yArr.append(float(line[-1]))
    return xArr, yArr


def showData(xArr, yArr):
    x_array = np.array(xArr)
    y_array = np.array(yArr)

    plt.scatter(x_array, y_array, c="r", alpha=0.4)
    plt.grid()
    plt.title("DataSet")  # 绘制title
    plt.xlabel("X")
    plt.show()


if __name__ == "__main__":
    filepath = "./ex0.txt"
    xArr, yArr = loadDataSet(filepath)

    showData(xArr, yArr)
