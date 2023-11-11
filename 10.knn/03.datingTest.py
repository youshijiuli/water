#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   date.py
@Author  :   Cat 
@Version :   3.11
"""

"""
    
1、实战背景
海伦女士一直使用在线约会网站寻找适合自己的约会对象。尽管约会网站会推荐不同的任选，但她并不是喜欢每一个人。经过一番总结，她发现自己交往过的人可以进行如下分类：

不喜欢的人
魅力一般的人
极具魅力的人
海伦收集约会数据已经有了一段时间，她把这些数据存放在文本文件datingTestSet.txt中，每个样本数据占据一行，总共有1000行。datingTestSet.txt数据下载： 数据集下载

海伦收集的样本数据主要包含以下3种特征：

每年获得的飞行常客里程数
玩视频游戏所消耗时间百分比
每周消费的冰淇淋公升数
    """

# 1.书局街读取，加载

# 在将上述特征数据输入到分类器前，必须将待处理的数据的格式改变为分类器可以接收的格式。
# 分类器接收的数据是什么格式的？
# 从上小结已经知道，要将数据分类两部分，即特征矩阵和对应的分类标签向量。
# 在kNN_test02.py文件中创建名为file2matrix的函数，以此来处理输入格式问题。
# 将datingTestSet.txt放到与kNN_test02.py相同目录下，编写代码如下：
import operator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def myfile2matrix(filepath):
    with open(filepath, "r") as file:
        lines = file.readlines()

        m = len(lines)

        return_matrix = np.zeros((m, 3))

        labelclasses = []

        index = 0

        for line in lines:
            return_matrix[index, :] = line.strip().split("\t")[:-1]
            if line[-1] == "didntLike":
                labelclasses.append(1)
            elif line[-1] == "smallDoses":
                labelclasses.append(2)
            elif line[-1] == "largeDoses":
                labelclasses.append(3)
        index += 1

    return return_matrix, labelclasses


# 数据加载
def file2matrix(filename):
    """_summary_

        函数说明:打开并解析文件，对数据进行分类：1代表不喜欢,2代表魅力一般,3代表极具魅力

    Parameters:
        filename - 文件名
    Returns:
        returnMat - 特征矩阵
        classLabelVector - 分类Label向量
    """
    # 打开文件
    fr = open(filename)
    # 读取文件所有内容
    arrayOLines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOLines)
    # 返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
    returnMat = np.zeros((numberOfLines, 3))
    # 返回的分类标签向量
    classLabelVector = []
    # 行的索引值
    index = 0
    for line in arrayOLines:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line = line.strip()
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine = line.split("\t")
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index, :] = listFromLine[0:3]
        # 根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        if listFromLine[-1] == "didntLike":
            classLabelVector.append(1)
        elif listFromLine[-1] == "smallDoses":
            classLabelVector.append(2)
        elif listFromLine[-1] == "largeDoses":
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


# 可视化
def showdatas(datingDataMat, datingLabels):
    """_summary_

    函数说明:可视化数据

        Parameters:
            datingDataMat - 特征矩阵
            datingLabels - 分类Label
        Returns:
            无
    """
    # 设置汉字格式
    # 将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    # 当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(
        nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8)
    )

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append("black")
        if i == 2:
            LabelsColors.append("orange")
        if i == 3:
            LabelsColors.append("red")
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(
        x=datingDataMat[:, 0],
        y=datingDataMat[:, 1],
        color=LabelsColors,
        s=15,
        alpha=0.5,
    )
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title("每年获得的飞行常客里程数与玩视频游戏所消耗时间占比")
    axs0_xlabel_text = axs[0][0].set_xlabel("每年获得的飞行常客里程数")
    axs0_ylabel_text = axs[0][0].set_ylabel("玩视频游戏所消耗时间占")
    plt.setp(axs0_title_text, size=9, weight="bold", color="red")
    plt.setp(axs0_xlabel_text, size=7, weight="bold", color="black")
    plt.setp(axs0_ylabel_text, size=7, weight="bold", color="black")

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(
        x=datingDataMat[:, 0],
        y=datingDataMat[:, 2],
        color=LabelsColors,
        s=15,
        alpha=0.5,
    )
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title("每年获得的飞行常客里程数与每周消费的冰激淋公升数")
    axs1_xlabel_text = axs[0][1].set_xlabel("每年获得的飞行常客里程数")
    axs1_ylabel_text = axs[0][1].set_ylabel("每周消费的冰激淋公升数")
    plt.setp(axs1_title_text, size=9, weight="bold", color="red")
    plt.setp(axs1_xlabel_text, size=7, weight="bold", color="black")
    plt.setp(axs1_ylabel_text, size=7, weight="bold", color="black")

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(
        x=datingDataMat[:, 1],
        y=datingDataMat[:, 2],
        color=LabelsColors,
        s=15,
        alpha=0.5,
    )
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title("玩视频游戏所消耗时间占比与每周消费的冰激淋公升数")
    axs2_xlabel_text = axs[1][0].set_xlabel("玩视频游戏所消耗时间占比")
    axs2_ylabel_text = axs[1][0].set_ylabel("每周消费的冰激淋公升数")
    plt.setp(axs2_title_text, size=9, weight="bold", color="red")
    plt.setp(axs2_xlabel_text, size=7, weight="bold", color="black")
    plt.setp(axs2_ylabel_text, size=7, weight="bold", color="black")
    # 设置图例
    didntLike = mlines.Line2D(
        [], [], color="black", marker=".", markersize=6, label="didntLike"
    )
    smallDoses = mlines.Line2D(
        [], [], color="orange", marker=".", markersize=6, label="smallDoses"
    )
    largeDoses = mlines.Line2D(
        [], [], color="red", marker=".", markersize=6, label="largeDoses"
    )
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()


# 数据归一化
# 飞机历程数显然远远大于其他两个特征的数值，
def dataNorm(dataset: np.array):
    """_summary_

        函数说明:对数据进行归一化

    Parameters:
        dataSet - 特征矩阵
    Returns:
        normDataSet - 归一化后的特征矩阵
        ranges - 数据范围
        minVals - 数据最小值
    """
    # 注意dataset是一个数组，那么min和max返回的不是一个数  ？
    minVal = dataset.min(0)
    maxVal = dataset.max(0)

    ranges = maxVal - minVal
    normDataSet = np.zeros(np.shape(dataset))

    # newValue = (oldValue - min) / (max - min)
    m = np.shape(dataset)[0]
    normDataSet = dataset - np.tile(minVal, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))

    return normDataSet, ranges, minVal


# 分类器
def classify0(inX, dataSet, labels, k):
    """_summary_

        函数说明:kNN算法,分类器

    Parameters:
        inX - 用于分类的数据(测试集)
        dataSet - 用于训练的数据(训练集)
        labes - 分类标签
        k - kNN算法参数,选择距离最小的k个点
    Returns:
        sortedClassCount[0][0] - 分类结果
    """
    # numpy函数shape[0]返回dataSet的行数
    dataSetSize = dataSet.shape[0]
    # 在列向量方向上重复inX共1次(横向),行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # 二维特征相减后平方
    sqDiffMat = diffMat**2
    # sum()所有元素相加,sum(0)列相加,sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方,计算出距离
    distances = sqDistances**0.5
    # 返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    # 定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(
        classCount.items(), key=operator.itemgetter(1), reverse=True
    )
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]


# 分类器测试函数
def datingClassTest(numTestVecs, normDataSets):
    # 分类错误计数
    errorCount = 0.0
    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify0(
            normDataSets[i, :],
            normDataSet[numTestVecs:m, :],
            datingLabels[numTestVecs:m],
            4,
        )
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount / float(numTestVecs) * 100))


if __name__ == "__main__":
    # 打开的文件名
    filename = "datingTestSet.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    # print(datingDataMat)
    # print(datingLabels)

    # 可视化
    # showdatas(datingDataMat, datingLabels)

    # 归一化
    normDataSet, ranges, minVals = dataNorm(datingDataMat)
    # print(normDataSet)
    # print(ranges)
    # print(minVals)

    # 取所有数据的10%
    hoRatio = 0.10
    # 获取normdata的行数
    m = normDataSet.shape[0]

    numTestVecs = int(m * hoRatio)

    # 分类
    datingClassTest(numTestVecs, normDataSet)
