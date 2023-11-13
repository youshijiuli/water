#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   demo.py
@Author  :   Cat 
@Version :   3.11
'''

# 1.导包
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 2.设置随机种子
# np.random.seed(1)

# 设置随机种子，不设置的话默认是按系统时间作为参数，设置后可以保证我们每次产生的随机数是一样的




# 3.加载切分数据
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# 4.训练

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)

print(knn.score(X_test,y_test))

y_pred = knn.predict(X_test)

print(np.sum(np.array(y_test) == np.array(y_pred)) / len(y_test))

# 1.0
# 1.0


print("=================")
# 计算各测试样本预测的概率值 这里我们没有用概率值，但是在实际工作中可能会参考概率值来进行最后结果的筛选，而不是直接使用给出的预测标签
probility = knn.predict_proba(X_test)
print(probility)
print("=================")
# 计算与最后一个测试样本距离最近的5个点，返回的是这些样本的序号组成的数组
neighborpoint = knn.kneighbors([X_test[-1]], 5)
print(neighborpoint)

print("=================")
# 调用该对象的打分方法，计算出准确率
score = knn.score(X_test, y_test, sample_weight=None)
# 输出测试的结果
print("iris_y_predict = ")
print(y_pred)
# 输出原始测试数据集的正确标签，以方便对比
print("iris_y_test = ")
print(y_test)

# 输出准确率计算结果
print("Accuracy:", score)


print("=================")

test_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(test_df)