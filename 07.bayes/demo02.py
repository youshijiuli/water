#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   demo02.py
@Author  :   Cat 
@Version :   3.11
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB


# 生成随机数据
# make_blobs：为聚类产生数据集
# n_samples：样本点数，n_features：数据的维度，centers:产生数据的中心点，默认值3
# cluster_std：数据集的标准差，浮点数或者浮点数序列，默认值1.0，random_state：随机种子

X, y = make_blobs(
    n_samples=100, n_features=2, centers=2, cluster_std=1.5, random_state=2
)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="RdBu")
plt.grid()
plt.title("init sample scatter grapha")
plt.show()


bayes = GaussianNB()
bayes.fit(X, y)

# 生成测试机和
rng = np.random.RandomState(0)
X_test = [-6, -14] + [14, 18] * rng.rand(2000, 2)  # 生成训练集
y_pred = bayes.predict(X_test)

# 将训练集和测试集的数据用图像表示出来，颜色深直径大的为训练集，颜色浅直径小的为测试集
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="RdBu")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, alpha=0.2, s=20, cmap="RdBu")
plt.grid()
plt.title("train-test datasets")
plt.show()


y_prop = bayes.predict_proba(X_test)
print(y_prop[:5].round(2))
