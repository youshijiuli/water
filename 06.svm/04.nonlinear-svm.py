# -*-coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(filepath):
    dataMat = []
    dataLabel = []
    with open(filepath, "r") as file:
        lines = file.readlines()
        for line in lines:
            res = line.strip().split()
            dataMat.append([float(res[0]), float(res[1])])
            dataLabel.append(float(res[-1]))

    return dataMat, dataLabel


def showDataSet(dataMat, labelMat):
    """
    数据可视化
    Parameters:
        dataMat - 数据矩阵
        labelMat - 数据标签
    Returns:
        无
    """
    data_plus = []  # 正样本
    data_minus = []  # 负样本
    for i in range(len(dataMat)):
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
    dataArr, labelArr = loadDataSet("testSetRBF.txt")  # 加载训练集
    showDataSet(dataArr, labelArr)


"""可见，数据明显是线性不可分的。
下面我们根据公式，编写核函数，并增加初始化参数kTup用于存储核函数有关的信息，
同时我们只要将之前的内积运算变成核函数的运算即可。最后编写testRbf()函数，用于测试。
创建svmMLiA.py文件，编写代码如下："""

# 05.xxx.py