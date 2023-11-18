#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   demo01.py
@Author  :   Cat 
@Version :   3.11
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


data = np.array(
    [
        [0.1, 0.7],
        [0.3, 0.6],
        [0.4, 0.1],
        [0.5, 0.4],
        [0.8, 0.04],
        [0.42, 0.6],
        [0.9, 0.4],
        [0.6, 0.5],
        [0.7, 0.2],
        [0.7, 0.67],
        [0.27, 0.8],
        [0.5, 0.72],
    ]
)  # 建立数据集


label = [1] * 6 + [0] * 6
x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1

# meshgrid如何生成网格
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

print(xx)
print(yy)

# # =======================================  线性高斯核=================================================


# model_linear = svm.SVC(kernel="linear", C=0.001)  # 线性SVM
# model_linear.fit(data, label)

# Z = model_linear.predict(np.c_[xx.ravel(), yy.ravel()])  # 预测
# # np.c_[xx.ravel(),yy.ravel()]: 创建一个二维数组，其中每一行包含一个特征（x和y轴上的数据）。这是用于多层感知器分类器进行预测的数据。
# # xx.ravel(): 将x轴上的数据展平成一个一维数组。
# # yy.ravel(): 将y轴上的数据展平成一个一维数组。
# # model_linear.predict(np.c_[xx.ravel(),yy.ravel()]): 使用线性分类器进行预测，并对展平的数据进行分类。
# # Z: 预测结果，是一个一维数组，其中每个元素表示对应数据点所属的类别。

# Z = Z.reshape(xx.shape)

# plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)
# plt.scatter(data[:6, 0], data[:6, 1], c="r", s=50, alpha=0.3, marker="*", lw=3)
# plt.scatter(data[6:, 0], data[6:, 1], c="b", s=50, alpha=0.3, marker="o", lw=3)
# plt.grid()
# plt.title("Linear SVM")
# plt.show()


# =======================================  线性高斯核 OK =============================================


# =========================================  多项式核 ===========================================

# plt.figure(figsize=(16, 24))

# for i, degree in enumerate([1, 3, 5, 7, 9, 12]):
#     model_poly = svm.SVC(kernel="poly", degree=degree, C=0.001)
#     # kernel: 指定用于分类的核函数。在这个例子中，我们使用了多项式核函数。
#     # degree: 设置多项式核函数的阶数。阶数决定了多项式核函数的复杂度。
#     # C: 设置惩罚参数。这个参数决定了分类器对错误分类的敏感性。值越小，分类器对错误分类的敏感性越低，对正确分类的惩罚也越低。
#     model_poly.fit(data, label)
#     # 把后面两个压扁之后变成了x1和x2，然后进行判断，得到结果在压缩成一个矩形
#     Z = model_poly.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)

#     plt.subplot(3, 2, i + 1)
#     plt.subplots_adjust(wspace=0.4, hspace=0.4)
#     #     这行代码用于调整子图之间的间距。subplots_adjust()函数可以用来调整子图之间的水平和垂直间距，以提高图表的可读性。

#     # wspace参数用于调整子图之间的水平间距。值越小，子图之间的水平间距越小。

#     # hspace参数用于调整子图之间的垂直间距。值越小，子图之间的垂直间距越小。

#     # 总之，这行代码的作用是调整子图之间的水平和垂直间距，以提高图表的可读性。
#     plt.contour(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)
#     #     xx和yy: 表示数据点的x和y轴坐标。
#     # Z: 表示数据点的分类标签。
#     # cmap: 指定用于绘制等高线的颜色映射。在这个例子中，我们使用了plt.cm.Paired颜色映射。
#     # alpha: 设置等高线的透明度。值越小，等高线越透明。
#     plt.scatter(data[:6, 0], data[:6, 1], marker="*", color="r",alpha=0.3, s=100, lw=3)
#     plt.scatter(data[6:, 0], data[6:, 1], marker="o", color="b",alpha=0.3, s=100, lw=3)
#     plt.title("Poly SVM with $\degree=$" + str(degree))
#     plt.grid()
# plt.show()


# ===========================  多项式核 OK =================================================


plt.figure(figsize=(24, 16))

# for index,gamma in enumerate([0.001,0.01,0.1,1,10,100]):
for index, gamma in enumerate([1, 5, 15, 35, 45, 55]):
    model = svm.SVC(kernel="rbf", gamma=gamma, C=0.0001).fit(data, label)

    ZZ = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # 开始绘图
    plt.subplot(2, 3, index + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.contourf(xx, yy, ZZ, camp=plt.cm.summer, alpha=0.3)

    # 画出训练点
    plt.scatter(data[:6, 0], data[:6, 1], marker="*", color="r", alpha=0.3, s=30, lw=3)
    plt.scatter(data[6:, 0], data[6:, 1], marker="o", color="b", alpha=0.3, s=30, lw=3)
    plt.grid()
    plt.title(f"SVM with rbf-{gamma}")
plt.show()
