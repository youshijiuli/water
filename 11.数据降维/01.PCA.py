#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   01.PCA.py
@Author  :   Cat 
@Version :   3.11
"""

# 先生成数据，并且可视化
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs


# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本3个特征，共4个簇
X, y = make_blobs(
    n_samples=10000,
    centers=[[3, 3, 3], [0, 0, 0], [1, 1, 1], [2, 2, 2]],
    n_features=3,
    cluster_std=[0.2, 0.1, 0.2, 0.2],
    random_state=9,
)


# fig = plt.figure()
# ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
# plt.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c="g", alpha=0.3)

fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
plt.scatter(X[:, 0], X[:, 1], X[:, 2],marker='o')



# plt.grid()
# plt.show()


#我们先不降维，只对数据进行投影，看看投影后的三个维度的方差分布，代码如下：
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print('===============================================')

# 降维  3 --explained_variance_
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print('===============================================')
#为了有个直观的认识，我们看看此时转化后的数据分布，代码如下：
print('n_component=0.95')
pca = PCA(n_components=0.95)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
print('===============================================')
print('n_components=0.99')
pca = PCA(n_components=0.99)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)
print('===============================================')
print('n_components=mle')
pca = PCA(n_components='mle')
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
print(pca.n_components_)