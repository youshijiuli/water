#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   bagging.py
@Author  :   Cat 
@Version :   3.11
"""


import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


wine = load_wine()
print(f"数据集合特征：{wine.feature_names}")


# 数据集合的切分
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# 模型训练

tree = DecisionTreeClassifier(criterion="gini", max_depth=1, random_state=1)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

print(tree.score(X_test, y_test))

print(f"accuracy:{accuracy_score(y_test,y_pred):.3f}")
"""0.6944444444444444
accuracy:0.694"""


# ===================== 一棵树不行，来集成


from sklearn.ensemble import BaggingClassifier

# 建立AdaBoost分类器，每个基本分类模型为前面训练的决策树模型，最大的弱学习器的个数为50
trees = BaggingClassifier(estimator=tree, n_estimators=50, random_state=1)
# FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4
trees.fit(X_train, y_train)
y_pred = trees.predict(X_test)

print(f"BaggingClassifier的准确率：{accuracy_score(y_test,y_pred):.3f}")

# BaggingClassifier的准确率：0.917


# 测试u估计机器个数的影响
# 估计器个数即n_estimators，
estimator_numbers = list(range(2, 102, 2))
y = []


for item in estimator_numbers:
    i_tree = BaggingClassifier(estimator=tree, n_estimators=item, random_state=1)
    i_tree.fit(X_train, y_train)
    y_pred = i_tree.predict(X_test)
    y.append(accuracy_score(y_test, y_pred))


plt.title("Effect of n_estimators", pad=20)
plt.xlabel("Number of base estimators")
plt.ylabel("Test accuracy of BaggingClassifier")
plt.plot(np.array(estimator_numbers), np.array(y))
plt.grid()
plt.show()
