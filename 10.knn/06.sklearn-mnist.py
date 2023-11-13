#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   06.sklearn-mnist.py
@Author  :   Cat 
@Version :   3.11
"""

import numpy as np
import matplotlib.pyplot as plt

# plt.cm.cool

# # 创建一个随机的二维数组
# image_array = np.random.rand(100, 100)
# print(image_array)

# # 使用plt.imshow()显示二维数组
# plt.imshow(image_array, cmap='gray')
# plt.show()

# 2. 显示图像文件
# 使用plt.imread()读取图像文件
# image = plt.imread('image.png')

# # 使用plt.imshow()显示图像文件
# plt.imshow(image)
# plt.show()


# 3. 自定义颜色映射
# 使用plt.imshow()显示二维数组，并使用自定义的颜色映射（ colormap）
# plt.imshow(image_array, cmap='hot')
# plt.show()


"""使用内置数据集做数字识别"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 小测试，拿出市长图片测试


digits = load_digits()

data = digits.data
target = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.2, random_state=1
)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)
# print(f"score:{knn.score(y_test,y_pred)}")

plt.figure(figsize=(36, 36))
for i in range(1, 101):
    plt.subplot(10, 10, i)
    plt.imshow(X_test[i].reshape(8, 8), cmap=plt.cm.summer)
    plt.title(
        f"T:{y_test[i]},P:{y_pred[i]}",
        color="green" if y_test[i] == y_pred[i] else "red",
    )


# 效果太好--> 100全正确