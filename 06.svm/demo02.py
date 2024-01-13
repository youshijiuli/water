#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   demo02.py
@Author  :   Cat 
@Version :   3.11
"""

# 文件路径 -- 桌面svm-mnist-datasets
# 测试不同SVM在Mnist数据集上的分类情况
# 添加目录到系统路径方便导入模块，该项目的根目录为".../machine-learning-toy-code"
import sys
from pathlib import Path


from sklearn import svm
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from struct import unpack
from sklearn.model_selection import GridSearchCV


def readimage(path):
    with open(path, "rb") as f:
        magic, num, rows, cols = unpack(">4I", f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img


def readlabel(path):
    with open(path, "rb") as f:
        magic, num = unpack(">2I", f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab


train_data = readimage("datasets/MNIST/raw/train-images-idx3-ubyte")  # 读取数据
train_label = readlabel("datasets/MNIST/raw/train-labels-idx1-ubyte")
test_data = readimage("datasets/MNIST/raw/t10k-images-idx3-ubyte")
test_label = readlabel("datasets/MNIST/raw/t10k-labels-idx1-ubyte")
print(train_data.shape)
print(train_label.shape)
# 数据集中数据太多，为了节约时间，我们只使用前2000张进行训练
train_data = train_data[:2000]
train_label = train_label[:2000]
test_data = test_data[:200]
test_label = test_label[:200]
svc = svm.SVC()
parameters = {"kernel": ["rbf"], "C": [1]}  # 使用了高斯核
print("Train...")
clf = GridSearchCV(svc, parameters, n_jobs=-1)
start = time()
clf.fit(train_data, train_label)
end = time()
t = end - start
print("Train：%dmin%.3fsec" % (t // 60, t - 60 * (t // 60)))  # 显示训练时间
prediction = clf.predict(test_data)  # 对测试数据进行预测
print("accuracy: ", accuracy_score(prediction, test_label))
accurate = [0] * 10
sumall = [0] * 10
i = 0
j = 0
while i < len(test_label):  # 计算测试集的准确率
    sumall[test_label[i]] += 1
    if prediction[i] == test_label[i]:
        j += 1
    i += 1
print("测试集准确率：", j / 200)

parameters = {"kernel": ["poly"], "C": [1]}  # 使用了多项式核
print("Train...")
clf = GridSearchCV(svc, parameters, n_jobs=-1)
start = time()
clf.fit(train_data, train_label)
end = time()
t = end - start
print("Train：%dmin%.3fsec" % (t // 60, t - 60 * (t // 60)))
prediction = clf.predict(test_data)
print("accuracy: ", accuracy_score(prediction, test_label))
accurate = [0] * 10
sumall = [0] * 10
i = 0
j = 0
while i < len(test_label):  # 计算测试集的准确率
    sumall[test_label[i]] += 1
    if prediction[i] == test_label[i]:
        j += 1
    i += 1
print("测试集准确率：", j / 200)

parameters = {"kernel": ["linear"], "C": [1]}  # 使用了线性核
print("Train...")
clf = GridSearchCV(svc, parameters, n_jobs=-1)
start = time()
clf.fit(train_data, train_label)
end = time()
t = end - start
print("Train：%dmin%.3fsec" % (t // 60, t - 60 * (t // 60)))
prediction = clf.predict(test_data)
print("accuracy: ", accuracy_score(prediction, test_label))
accurate = [0] * 10
sumall = [0] * 10
i = 0
j = 0
while i < len(test_label):  # 计算测试集的准确率
    sumall[test_label[i]] += 1
    if prediction[i] == test_label[i]:
        j += 1
    i += 1
print("测试集准确率：", j / 200)
