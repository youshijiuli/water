#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   test.py
@Author  :   Cat 
@Version :   3.11
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# 聚类之前
X = np.random.rand(100, 2)
plt.scatter(X[:, 0], X[:, 1], marker="o", s=30, alpha=0.6, c="g")
plt.grid()
plt.title("befor cluster")
plt.show()


# 出书画质心，从原有数据选取K个质心
def initCentrioes(X, k):
    index = np.random.randint(0, len(X) - 1, k)
    return X[index]


# 聚类之后假设K=2
kmearns = KMeans(n_clusters=2, n_init=1, random_state=1)
kmearns.fit(X)
label = kmearns.labels_
plt.scatter(X[:, 0], X[:, 1], c=label, marker="o", s=30, alpha=0.6, cmap="jet")
plt.grid()
plt.title("after cluster")
plt.show()
