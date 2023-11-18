#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   随机森林.py
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

# 接着bagging去搞的


## 随机森林

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=50, random_state=1)
model.fit(X_train, y_train)  # 训练
y_pred = model.predict(X_test)  # 预测
print(f"RandomForestClassifier的准确率：{accuracy_score(y_test,y_pred):.3f}")

x = list(range(2, 102, 2))  # 估计器个数即n_estimators，在这里我们取[2,102]的偶数
y = []

for i in x:
    model = RandomForestClassifier(n_estimators=i, random_state=1)

    model.fit(X_train, y_train)
    model_test_sc = accuracy_score(y_test, model.predict(X_test))
    y.append(model_test_sc)

plt.title("Effect of n_estimators", pad=20)
plt.xlabel("Number of base estimators")
plt.ylabel("Test accuracy of RandomForestClassifier")
plt.plot(
    np.array(x),
    np.array(y),
    "b-",
    label="RandomForestClassifier",
    linewidth=3,
    alpha=0.5,
    c="g",
)
plt.grid()
plt.legend()
plt.show()
