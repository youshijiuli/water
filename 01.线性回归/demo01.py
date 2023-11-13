#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   demo01.py
@Author  :   Cat 
@Version :   3.11
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 1.生成数据
def gennerate_data():
    X = []
    Y = []
    for i in range(100):
        # lr.fit(X_train, y_train)
        # ValueError: Expected 2D array, got 1D array instead:
        X.append([i])
        # X.append(i)
        Y.append(i + 2.128 + np.random.uniform(-10, 10))
    plt.scatter(X, Y, c="g", s=30, alpha=0.4)
    # plt.show()
    return X, Y


def linear_model(X: np.array, Y: np.array):
    np.random.seed(1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    # print(f'acc:{lr.predict(X_test)}')
    y_pred = lr.predict(X_test)
    plt.plot(X_test, y_pred, c="b", linewidth=3, alpha=0.5, label="Predicted Line")
    plt.grid()
    plt.legend()
    plt.title("linear model")
    plt.show()


if __name__ == "__main__":
    X, Y = gennerate_data()
    X = np.array(X)
    Y = np.array(Y)
    linear_model(X, Y)
