#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   demo.py
@Author  :   Cat 
@Version :   3.11
"""


# 1.导包

from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


# 可视化
def draw_result(train_x, labels, centers, title):
    """画出聚类后的图像
    labels: 聚类后的label, 从0开始的数字
    cents: 质心坐标
    n_cluster: 聚类后簇的数量
    color: 每一簇的颜色"""
    n_clusters = np.unique(labels).shape[0]
    print(n_clusters, "========================")
    colors = ["red", "yellow", "green"]
    plt.figure()
    plt.title(title)
    for i in range(n_clusters):
        current_data = train_x[labels == i]
        plt.scatter(current_data[:, 0], current_data[:, 1], c=colors[i])
        print(centers[i], "-----------------------------")
        plt.scatter(centers[i, 0], centers[i, 1], c="blue", s=200, alpha=0.3)
        plt.grid()
    plt.show()


if __name__ == "__main__":
    # 2.加载数据集
    iris = load_iris()  # data:150 x 4 target:150 x 1
    train_x = iris.data

    # 设定聚类数目为3
    # clf = KMeans(n_clusters=3, max_iter=10,n_init=10, init="k-means++", algorithm="full", tol=1e-4,n_jobs= -1,random_state=1)
    # clf = KMeans(n_clusters=3, max_iter=10,n_init=10, init="k-means++", algorithm="full", tol=1e-4,random_state=1)
    # clf = KMeans(n_clusters=3,max_iter=10,n_init=10,init="k-means++",algorithm='lloyd',tol=1e-4,random_state=1)  # 这样就没有警告
    # FutureWarning: algorithm='full' is deprecated, it will be removed in 1.3. Using 'lloyd' instead.

    clf = KMeans(
        n_clusters=3, max_iter=100, n_init=10, init="k-means++", random_state=1
    )
    
    clf.fit(train_x)
    print(f"SSE = {clf.inertia_}")

    # clf.label_ : (150,)

    # print(clf.cluster_centers_,'.....................') # 3 x 4
    """clf.cluster_centers_的形状是3x4，是因为在加载Iris数据集时，数据集的特征数量为4（即每个样本有4个特征），
    而聚类数目设置为3。所以，每个聚类的中心点由4个特征值组成，共有3个聚类中心，因此形状为3x4。"""

    draw_result(train_x, clf.labels_, clf.cluster_centers_, "Kmearns")

    print("...")


# 输出结果
# SSE = 78.851441426146
# 注：SSE是误差平方和，这个值越接近0说明效果越好

# 通过运行上面的代码，会输出下面的这幅图像，
# 当然，我们的鸢尾花数据集的属性有四个维度，
# 这里输出的图像我们只使用了两个维度，
# 但是仍然可以看出通过 K-means 计算出的中心点与数据分布基本上是一致的，而且效果也还不错。