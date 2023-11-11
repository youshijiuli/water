#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   05.minstTest.py
@Author  :   Cat 
@Version :   3.11
"""

# -*- coding: UTF-8 -*-
import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier


# 大致思路
"""为了使用前面两个例子的分类器，我们必须将图像格式化处理为一个向量。
我们将把一个32x32的二进制图像矩阵转换为1 x 1024的向量，
这样前两节使用的分类器就可以处理数字图像信息了我们首先编写一段函数img2vector，
将图像转换为向量:该函数创建1x 1024的NumPy数组，然后打开给定的文件，循环读出文件的前32行，
并将每行的头32个字符值存储在NumPy数组中，最后返回数组。
    """
    
def test(a,b,c):
    """
    test func
    
    :param a: int 
    :type a: [type]
    :return: [description]
    :rtype: [type]
    """ 
    pass


def img2Vect(filepath):
    """_summary_

    Args:
        filepaath (_type_): _description_

    Returns:
        _type_: _description_
    """
    """
    函数说明:将32x32的二进制图像转换为1x1024向量。

    Parameters:
        filename - 文件名
    Returns:
        returnVect - 返回的二进制图像的1x1024向量

    Modify:
        2017-07-15
    """
    returnVec = np.zeros((1, 1024))
    file = open(filepath)
    for i in range(32):
        # 行的操作
        line = file.readline()
        for j in range(32):
            returnVec[0, 32 * i + j] = int(line[j])

    return returnVec


# 2.3.2测试算法:使用k-近邻算法识别手写数字
# 上节我们已经将数据处理成分类器可以识别的格式，本节我们将这些数据输入到分类器，检测分类器的执行效果。
# 程序清单2-6所示的自包含函数handwritingclassTest()是测试分类器的代码，将其写入kNNpy文件中。
# 在写人这些代码之前，我们必须确保将from os importlistdir写人文件的起始部分，这段代码的主要功能是从os模块中导人函数istdir，它可以列出给定目录的文件名。


def myhandwriteClassTest():
    hwLabel = []  # # 测试集的Labels
    trainingFileList = listdir("./digits/trainingDigits")  # 返回trainingDigits目录下的文件名
    m = len(trainingFileList)  # 饭会文件夹下文件个数
    trainingMatrix = np.zeros((m, 1024))  # 初始化训练的Mat矩阵,测试集
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split("_")[0])
        hwLabel.append(classNumber)
        trainingMatrix[i:] = img2Vect(f"./digits/trainingDigits/{fileNameStr}")
    # 构建KNN分类器
    neigthbor = KNeighborsClassifier(n_neighbors=3, algorithm="auto")
    neigthbor.fit(trainingMatrix, hwLabel)
    testFileList = listdir("./digits/testDigits")
    # 错误检测计数
    true_count = 0
    # test_true_list = []
    # test_list = []
    mTest = len(testFileList)
    print(mTest)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNameStr = int(fileNameStr.split("_")[0])
        # test_true_list.append(classNameStr)
        # # 从文件中解析出测试集的类别并进行分类测试
        vectorUnderTest = img2Vect(f"./digits/testDigits/{fileNameStr}")
        classifierResult = int(neigthbor.predict(vectorUnderTest)[0])
        # test_list.append(classifierResult)
        print(
            f"分类结果为：{classifierResult},真是结果为：{classNameStr}",
            classifierResult == classNameStr,
        )
        if classifierResult == classNameStr:
            true_count += 1

    print(f"一共对了{true_count}个，总共错了{m - true_count}个数据,正确率为：{true_count / mTest}")


# def handwritingClassTest():
#     # 测试集的Labels
#     hwLabels = []
#     # 返回trainingDigits目录下的文件名
#     trainingFileList = listdir("trainingDigits")
#     # 返回文件夹下文件的个数
#     m = len(trainingFileList)
#     # 初始化训练的Mat矩阵,测试集
#     trainingMat = np.zeros((m, 1024))
#     # 从文件名中解析出训练集的类别
#     for i in range(m):
#         # 获得文件的名字
#         fileNameStr = trainingFileList[i]
#         # 获得分类的数字
#         classNumber = int(fileNameStr.split("_")[0])
#         # 将获得的类别添加到hwLabels中
#         hwLabels.append(classNumber)
#         # 将每一个文件的1x1024数据存储到trainingMat矩阵中
#         trainingMat[i, :] = img2vector("trainingDigits/%s" % (fileNameStr))
#     # 构建kNN分类器
#     neigh = kNN(n_neighbors=3, algorithm="auto")
#     # 拟合模型, trainingMat为训练矩阵,hwLabels为对应的标签
#     neigh.fit(trainingMat, hwLabels)
#     # 返回testDigits目录下的文件列表
#     testFileList = listdir("testDigits")
#     # 错误检测计数
#     errorCount = 0.0
#     # 测试数据的数量
#     mTest = len(testFileList)
#     # 从文件中解析出测试集的类别并进行分类测试
#     for i in range(mTest):
#         # 获得文件的名字
#         fileNameStr = testFileList[i]
#         # 获得分类的数字
#         classNumber = int(fileNameStr.split("_")[0])
#         # 获得测试集的1x1024向量,用于训练
#         vectorUnderTest = img2vector("testDigits/%s" % (fileNameStr))
#         # 获得预测结果
#         # classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
#         classifierResult = neigh.predict(vectorUnderTest)
#         print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
#         if classifierResult != classNumber:
#             errorCount += 1.0
#     print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))


if __name__ == "__main__":
    myhandwriteClassTest()
    # res = img2Vect(r'./digits/trainingDigits/0_0.txt')
    # print(res.shape)
    # # print(type(res),len(res))
    # # print(res)
    # for i in range(1024):
    #     print(res[0,i],end=' ')
    #     if (i+1 ) % 32 == 0:
    #         print('')

    # 一共对了934个，总共错了1000个数据,正确率为：0.9873150105708245


"""
k-近邻算法识别手写数字数据集，错误率为1.2%。改变变量k的值、修改函数handwriting-classTest随机选取训练样本、改变训练样本的数目，都会对k-近邻算法的错误率产生影响，
感兴趣的话可以改变这些变量值，观察错误率的变化。实际使用这个算法时，算法的执行效率并不高。
因为算法需要为每个测试向量做2000次距离计算，每个距离计算包括了1024个维度浮点运算，总计要执行900次，此外，
我们还需要为测试向量准备2MB的存储空间。是否存在一种算法减少存储空间和计算时间的开销呢? k决策树就是k-近邻算法的优化版，可以节省大量的计算开销。"""
