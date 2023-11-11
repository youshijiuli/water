#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   01.test.py
@Author  :   Cat 
@Version :   3.11
"""


"""函数说明:梯度上升算法测试函数
 
求函数f(x) = -x^2 + 4x的极大值"""


def partial_f(x):
    return -2 * x + 4


def Gradient_Ascent_test():
    x_old = 1
    x_new = 0
    alpha = 0.01
    presision = 1e-6
    while abs(x_new - x_old) > presision:
        x_old = x_new
        x_new = x_old + alpha * partial_f(x_old)
    print(x_new)


if __name__ == "__main__":
    Gradient_Ascent_test()
