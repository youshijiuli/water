#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   01.test.py
@Author  :   Cat 
@Version :   3.11
"""


def loadDataSet():
    postingList = [
        ["my", "dog", "has", "flea", "problems", "help", "please"],  # 切分的词条
        ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
        ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
        ["stop", "posting", "stupid", "worthless", "garbage"],
        ["mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
        ["quit", "buying", "worthless", "dog", "food", "stupid"],
    ]
    vec = [0, 1, 0, 1, 0, 1]
    return postingList, vec


def word2Vector(vocabList, inputSet):
    """函数说明:根据vocabList词汇表，将inputSet向量化，向量的每个元素为1或0

    Parameters:
        vocabList - createVocabList返回的列表
        inputSet - 切分的词条列表
    Returns:
        returnVec - 文档向量,词集模型
    """
    returnVec = [0] * len(vocabList)
    for item in inputSet:
        if item in vocabList:
            returnVec[vocabList.index(item)] = 1
        else:
            print(f"the word:{item} is not in my Vocabulary!")

    return returnVec


def createVocabList(dataSet):
    """_summary_函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

    Parameters:
        dataSet - 整理的样本数据集
    Returns:
        vocabSet - 返回不重复的词条列表，也就是词汇表
    """
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)

    return list(vocabSet)


if __name__ == "__main__":
    # postingList, vectors = loadDataSet()
    # for item in postingList:
    #     print(item)
    # print(vectors)

    """从运行结果可以看出，我们已经将postingList是存放词条列表中，
    classVec是存放每个词条的所属类别，1代表侮辱类 ，0代表非侮辱类。"""

    postingList, classVec = loadDataSet()
    print("postingList:\n", postingList)
    myVocabList = createVocabList(postingList)
    print("myVocabList:\n", myVocabList)
    trainMat = []
    for postinDoc in postingList:
        trainMat.append(word2Vector(myVocabList, postinDoc))
    print("trainMat:\n", trainMat)
    
    # 继续编写代码，前面我们已经说过我们要先创建一个词汇表，并将切分好的词条转换为词条向量。

    """从运行结果可以看出，postingList是原始的词条列表，myVocabList是词汇表。
    myVocabList是所有单词出现的集合，没有重复的元素。词汇表是用来干什么的？没错，它是用来将词条向量化的，
    一个单词在词汇表中出现过一次，那么就在相应位置记作1，如果没有出现就在相应位置记作0。
    trainMat是所有的词条向量组成的列表。它里面存放的是根据myVocabList向量化的词条向量。"""
    
    
    # 我们已经得到了词条向量。接下来，我们就可以通过词条向量训练朴素贝叶斯分类器。
