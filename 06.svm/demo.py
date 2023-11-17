#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   demo.py
@Author  :   Cat 
@Version :   3.11
"""


# 使用SVM去进行分类，对iris数据
# 与之前的课时相同，下面我们亲自来练练手，用代码实现使用 SVM 算法。在本节的代码中，在数据获取与数据处理阶段仍然沿用了之前的方法，没有任何改动，主要的区别是我们引入的算法包为 SVM 包，在进行分类时使用的是 SVM 分类器。


from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm  # 引入svm包
import numpy as np

np.random.seed(0)
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    iris_x, iris_y, test_size=0.25, random_state=1
)

svc = svm.SVC(kernel="linear")
svc.fit(X_train, y_train)

# 预测
y_pred = svc.predict(X_test)
print(svc.score(X_test, y_test))
print(y_test)
print(np.sum(y_test == y_pred))
# 1.0
# 38
# 150 x 0.25 = 37.5   