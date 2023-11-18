#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   demo-mnist.py
@Author  :   Cat 
@Version :   3.11
"""

# 神经网络-数字识别

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier


mnist = fetch_openml("mnist_784")
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, alpha=1e-2,
    solver='adam', verbose=100, tol=1e-3, random_state=1)

# verbose: 设置为10，表示每10次迭代时输出一次日志信息
# tol: 设置容差值，当损失函数的值低于此值时，迭代停止。
# learning_rate_init: 设置初始学习率，即每次更新权重时的步长。


# mlp = MLPClassifier(max_iter=100,alpha=1e-2, hidden_layer_sizes=(100,), random_state=1)

# 学习率  隐含层(h1:15个神经元，h2:15个神经元)

mlp.fit(np.array(X_train), np.array(y_train))

print(mlp.score(X_test, y_test))

# 0.9672