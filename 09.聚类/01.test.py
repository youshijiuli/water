#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   01.test.py
@Author  :   Cat 
@Version :   3.11
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs


X,y_true = make_blobs(n_samples=300,centers=4,cluster_std=0.6,random_state=0)

# 1.简单可视化
# print(X,y_true)
# print(X.shape)
# (300, 2)
# (300,)
# print(y_true.shape)
# plt.scatter(X[:,0],X[:,1],s=50)  # s表示size.点的大小
# plt.grid()
# plt.show()

# print('hhh')

# 2.k-mearns聚类

from sklearn.cluster import KMeans
from sklearn import metrics

m_kmeans = KMeans(n_clusters=4)
 
def draw(m_kmeans,X,y_pred,n_clusters):
    centers = m_kmeans.cluster_centers_
    # (4,2)  # 四个点的中心坐标
    print(centers)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
    #中心点（质心）用红色标出
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
    print("Calinski-Harabasz score：%lf"%metrics.calinski_harabasz_score(X, y_pred) )
    plt.title("K-Means (clusters = %d)"%n_clusters,fontsize=20)
    plt.show()
m_kmeans.fit(X)
# KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
#     n_clusters=4, n_init=10, n_jobs=None, precompute_distances='auto',
#     random_state=None, tol=0.0001, verbose=0)
y_pred = m_kmeans.predict(X)
draw(m_kmeans,X,y_pred,4)


print(
'ff'
)

"""[[ 0.94973532  4.41906906]
 [ 1.98258281  0.86771314]
 [-1.37324398  7.75368871]
 [-1.58438467  2.83081263]]
Calinski-Harabasz score：1210.089914"""



# 总结
# K-Means 聚类是最简单、经典的聚类算法，因为聚类中心个数，即 K 是需要提前设置好的，所以能使用的场景也比较局限。

# 比如可以使用 K-Means 聚类算法，对一张简单的表情包图片，进行前后背景的分割，对一张文本图片，进行文字的前景提取等。

# K-Means 聚类能使用的距离度量方法不仅仅是欧式距离，也可以使用曼哈顿距离、马氏距离，思想都是一样，只是使用的度量公式不同而已。

# 聚类算法有很多，且看我慢慢道来。

# 想看我讲解其它好玩实用的聚类算法，例如 CDP 等，不妨来个三连，给我来点动力。