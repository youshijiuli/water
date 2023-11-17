#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   demo02.py
@Author  :   Cat 
@Version :   3.11
"""

# 1.导入包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 这是我们设定的真实函数，即ground truth的模型
def true_func(X):
    return 1.5 * X + 0.5


np.random.seed(0)
n_samples = 100


"""生成随机数据作为训练集，并且加一些噪声"""
X_train = np.sort(np.random.rand(n_samples))
y_train = true_func((X_train) + np.random.randn(n_samples) * 0.1).reshape(n_samples,1)


model = LinearRegression()
model.fit(X_train.reshape(-1, 1), y_train)
# model.fit(X_train[:, np.newaxis], y_train)
# 输出偏置和权重
print(model.intercept_, model.coef_)


x_test = np.linspace(0, 1, 100)
y_pred = model.predict(x_test.reshape(-1, 1))


plt.plot(x_test, y_pred, "r-", label="Predict")
plt.plot(x_test, true_func(x_test), "b", label="true")
plt.scatter(X_train, y_train, label="Train")
plt.grid()
plt.legend()
plt.show()
