#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   demo.py
@Author  :   Cat 
@Version :   3.11
"""


import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression


data = fetch_openml("mnist_784")

X, y = data["data"], data["target"]
X_train = X[:60000]
y_train = y[:60000]
x_test = X[60000:]
y_test = y[60000:]


lg = LogisticRegression()
lg.fit(np.array(X_train).reshape(-1, 784), np.array(y_train).reshape(-1, 1))
y_pred = lg.predict(np.array(x_test).reshape(-1, 784))

print(
    f"score:{lg.score(np.array(x_test).reshape(-1,784),np.array(y_test).reshape(-1,1))}"
)

# score:0.9255

# 明天先把机器学习搞定  再来考虑 向量数据库