#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   多元线性回归.py
@Author  :   Cat 
@Version :   3.11
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import PolynomialFeatures # 导入能够计算多项式特征的类
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score



# 这是我们设定的真实函数，即ground truth的模型
def true_func(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(0)
n_samples = 100


# y需要添加一些噪声
X = np.sort(np.random.rand(n_samples))
y = true_func(X) + np.random.randn(n_samples) * 0.1


# 多项式最高次
degrees = [1,4,15]
plt.figure(figsize=(16,10))

for i in range(len(degrees)):
    ax = plt.subplot(1,len(degrees),i+1)
    # plt.step(ax,xticks=(),yticks=())
    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipline = Pipeline([("polynomial_features", polynomial_features), ("linear_regression", linear_regression)])
    pipline.fit(X[:, np.newaxis], y)
    scores = cross_val_score(pipline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10)
    x_test= np.linspace(0, 1, 1000)
    plt.plot(x_test, pipline.predict(x_test[:, np.newaxis]),c='r',alpha=.3, label="Model")
    plt.plot(x_test,true_func(x_test),label="True function",c='b',alpha=.3)
    plt.scatter(X, y, c="g", label="Samples",alpha=.3,s=20)
    plt.grid()
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()
