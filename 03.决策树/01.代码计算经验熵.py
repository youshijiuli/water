#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   01.代码计算经验熵.py
@Author  :   Cat 
@Version :   3.11
"""


"""代码计算经验熵"""

# 在编写代码之前，我们先对数据集进行属性标注。

# 年龄：0代表青年，1代表中年，2代表老年；
# 有工作：0代表否，1代表是；
# 有自己的房子：0代表否，1代表是；
# 信贷情况：0代表一般，1代表好，2代表非常好；
# 类别(是否给贷款)：no代表否，yes代表是。
# 确定这些之后，我们就可以创建数据集，并计算经验熵了，代码编写如下：

import math
from utils import createDataSet



def calShannonent(dataset):
    # 1.返回数据集的行数
    m = len(dataset)

    label_counts = {}

    # 对每组特征向量进行统计
    for item in dataset:
        currentLabel = item[-1]
        # #提取标签(Label)信息
        if currentLabel not in label_counts.keys():
            label_counts[currentLabel] = 0
        label_counts[currentLabel] += 1

        # #如果标签(Label)没有放入统计次数的字典,添加进去

    # 计算熵
    shangnonent = 0.0
    for item, value in label_counts.items():
        prob = value / m
        shangnonent -= prob * math.log(prob, 2)

    return shangnonent


if __name__ == "__main__":
    datasets, labels = createDataSet()
    print(datasets)
    print(calShannonent(datasets))

# 代码运行结果如下图所示，代码是先打印训练数据集，然后打印计算的经验熵H(D)，程序计算的结果与我们统计计算的结果是一致的，程序没有问题。
"""
[[0, 0, 0, 0, 'no'], [0, 0, 0, 1, 'no'], [0, 1, 0, 1, 'yes'], [0, 1, 1, 0, 'yes'], [0, 0, 0, 0, 'no'], [1, 0, 0, 0, 'no'], [1, 0, 0, 1, 'no'], [1, 1, 1, 1, 'yes'], [1, 0, 1, 2, 'yes'], [1, 0, 1, 2, 'yes'], [2, 0, 1, 2, 'yes'], [2, 0, 1, 1, 'yes'], [2, 1, 0, 1, 'yes'], [2, 1, 0, 2, 'yes'], [2, 0, 0, 0, 'no']]
0.9709505944546686
"""