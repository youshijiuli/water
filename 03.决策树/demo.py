#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   demo.py
@Author  :   Cat 
@Version :   3.11
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

np.random.seed(0)


data = load_iris()
X_data = data.data
y_data = data.target


# random.shuffle(X_data)
# random.shuffle(y_data)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)


# 在模型训练时，我们设置了树的最大深度为 4。
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(X_train, y_train)


# 下面这个是ipoynb中的可视化
# 根据上面的介绍，我们可以知道，经过调用 fit 方法进行模型训练，决策树算法会生成一个树形的判定模型，今天我们尝试把决策树算法生成的模型使用画图的方式展示出来。
# 引入画图相关的包
from IPython.display import Image
from sklearn import tree

# dot是一个程式化生成流程图的简单语言
# import pydotplus
# dot_data = tree.export_graphviz(clf, out_file=None,feature_names=data.feature_names,class_names=data.target_names,filled=True, rounded=True,special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# print(Image(graph.create_png())) #


# matplotlib可视化
# fig,ax = plt.subplots(figsize=(20,12))
# tree.plot_tree(clf,feature_names=data['feature_names'],class_names=data.target_names,filled=True)

iris_y_predict = clf.predict(X_test)
score = clf.score(X_test, y_test, sample_weight=None)
print(score)
print("iris_y_predict = ")
print(iris_y_predict)
print("iris_y_test = ")
print(y_test)
print("Accuracy:", score)


# 测试结果：
# 1.
# random.shuffle(X_data)
# random.shuffle(y_data)

# acc:0.36


# 2.不达伦

# 1.0