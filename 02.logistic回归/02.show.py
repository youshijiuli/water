#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   02.show.py
@Author  :   Cat 
@Version :   3.11
"""


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 1.加载数据

# 加载testSet数据，绘制散点图


def loadData(filepath):
    dataMat = []
    dataLabels = []

    with open(filepath, "r") as file:
        content = file.readlines()

        for line in content:
            res = line.strip().split("\t")
            # print(res)
            dataMat.append([1.0, float(res[0]), float(res[1])])
            dataLabels.append(float(res[-1]))

        return dataMat, dataLabels


def showData(filepath):
    dataMat, dataLabels = loadData(filepath)
    dataSet = np.array(dataMat)
    n = dataSet.shape[0]  # 获取数据数目
    positive_x1 = []
    positive_y1 = []
    negtive_x2 = []
    negtive_y2 = []
    for i in range(n):
        if int(dataLabels[i]) == 1:
            positive_x1.append(dataSet[i, 1])
            positive_y1.append(dataSet[i, 2])
        else:
            negtive_x2.append(dataSet[i, 1])
            negtive_y2.append(dataSet[i, 2])

    plt.figure()
    plt.subplot(111)
    plt.scatter(positive_x1, positive_y1, c="g", alpha=0.5)
    plt.scatter(negtive_x2, negtive_y2, c="r")
    plt.legend()
    plt.title("DataSet")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def main():
    filepath = "./testSet.txt"
    # dataMat,dataLabels = loadData(filepath)
    # print(dataMat)
    # print(dataLabels)
    showData(filepath)


if __name__ == "__main__":
    main()

"""从上图可以看出数据的分布情况。假设Sigmoid函数的输入记为z，那么z=w0x0 + w1x1 + w2x2，即可将数据分割开。
其中，x0为全是1的向量，x1为数据集的第一列数据，x2为数据集的第二列数据。
另z=0，则0=w0 + w1x1 + w2x2。横坐标为x1，纵坐标为x2。
这个方程未知的参数为w0，w1，w2，也就是我们需要求的回归系数(最优参数)。"""


