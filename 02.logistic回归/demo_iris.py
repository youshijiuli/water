#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   demo_iris.py
@Author  :   Cat 
@Version :   3.11
"""


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# 加载数据集
iris = load_iris()

X = iris.data
y = iris.target

# 筛选特征
X = X[y < 2, :2]
y = y[y < 2]

print(X.shape, y.shape)
# (100, 2) (100,)

# 画图
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="r")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="b")
plt.show()


# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=1
)

lr = LogisticRegression()
lr.fit(X_train, y_train)

print(lr.score(X_test, y_test))
print(lr.predict(X_test))
print(lr.predict_proba(X_test))
