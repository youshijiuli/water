#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   test02.py
@Author  :   Cat 
@Version :   3.11
"""


import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x):
    # S1gnold actiwatlon funct1on:
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # s1gmo1d面数的导数:
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # Y_true and y-pred are numpy arrays of the same length
    return ((y_true - y_pred) ** 2).mean()


class QurNeuralNetwork:
    # 设置权重和偏执

    def __init__(self) -> None:
        # self.w1 = np.random.normal() * 0.01
        # self.w2 = np.random.normal() * 0.01
        # self.w3 = np.random.normal() * 0.01
        # self.w4 = np.random.normal() * 0.01
        # self.w5 = np.random.normal() * 0.01
        # self.w6 = np.random.normal() * 0.01

        self.w1 = np.random.randn() * 0.01
        self.w2 = np.random.randn() * 0.01
        self.w3 = np.random.randn() * 0.01
        self.w4 = np.random.randn() * 0.01
        self.w5 = np.random.randn() * 0.01
        self.w6 = np.random.randn() * 0.01

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

        self.acc = []
        self.loss = []

        self.lr = 0.1

    def forward(self, x: np.array):
        self.h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        self.h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        self.o1 = sigmoid(self.w5 * self.h1 + self.w6 * self.h2 + self.b3)
        return self.o1

    def train(self, data, all_true_features):
        # 开始训练
        epochs = 10000
        for epoch in range(epochs):
            for x, y_true in zip(data, all_true_features):
                # 前向传播
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)
                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)
                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                d_L_d_pred = -2 * (y_true - y_pred)

                # 反向传播
                # o1
                d_y_pred_d_w5 = deriv_sigmoid(sum_o1) * h1
                d_y_pred_d_w6 = deriv_sigmoid(sum_o1) * h2
                d_y_pred_d_b3 = deriv_sigmoid(sum_o1)

                d_y_pred_d_h1 = deriv_sigmoid(sum_o1) * self.w5
                d_y_pred_d_h2 = deriv_sigmoid(sum_o1) * self.w6

                # h1
                d_h1_d_w1 = deriv_sigmoid(sum_h1) * x[0]
                d_h1_d_w2 = deriv_sigmoid(sum_h1) * x[1]
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # h2
                d_h2_d_w3 = deriv_sigmoid(sum_h2) * x[0]
                d_h2_d_w4 = deriv_sigmoid(sum_h2) * x[1]
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # 权重更新
                self.w5 -= self.lr * d_L_d_pred * d_y_pred_d_w5
                self.w6 -= self.lr * d_L_d_pred * d_y_pred_d_w6
                self.b3 -= self.lr * d_L_d_pred * d_y_pred_d_b3

                self.w1 -= self.lr * d_L_d_pred * d_y_pred_d_h1 * d_h1_d_w1
                self.w2 -= self.lr * d_L_d_pred * d_y_pred_d_h1 * d_h1_d_w2
                self.b1 -= self.lr * d_L_d_pred * d_y_pred_d_h1 * d_h1_d_b1

                self.w3 -= self.lr * d_L_d_pred * d_y_pred_d_h2 * d_h2_d_w3
                self.w4 -= self.lr * d_L_d_pred * d_y_pred_d_h2 * d_h2_d_w4
                self.b2 -= self.lr * d_L_d_pred * d_y_pred_d_h2 * d_h2_d_b2

            if epoch % 20 == 0:
                y_preds = np.apply_along_axis(self.forward, 1, data)
                loss = mse_loss(all_true_features, y_preds)
                self.loss.append(loss)
                self.acc.append(float(1 - loss))
                print(f"第{epoch}轮次,loss:{loss}")

                if loss < 1e-3:
                    break

        self.show_loss()

    def show_loss(self):
        plt.plot(np.array(self.acc), "r", label="acc", alpha=0.3)
        plt.plot(np.array(self.loss), "b", label="loss", alpha=0.3)
        plt.grid()
        plt.legend()
        plt.show()


data = np.array(
    [[-8, -0.5], [19, 6.5], [11, 4.5], [-21, -5.5]]  # Alice  # Bob[11,4.5], # Charlie
)  # diana
# 训练样本标签
all_y_trues = np.array([1, 0, 0, 1])  # ALice  # Bob  # Charlie  # diana
# 训练我们的神经网络
network = QurNeuralNetwork()
# 初始化我们的权重和偏置
network.train(data, all_y_trues)


# 进行预测
emily = np.array([-13, -2.5])  # 128 pounds,63 inches
frank = np.array([14, 2.5])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.forward(emily))  # 0.961 -F
print("Frank: %.3f" % network.forward(frank))  # .056 - M


# loss下降明显

# 第0轮次,loss:0.29203172135097805

# 第9960轮次,loss:0.0001316402451741911
