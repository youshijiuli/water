#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   03.adaboost.py
@Author  :   Cat 
@Version :   3.11
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


iris = load_iris()

X = iris.data
y = iris.target


# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = AdaBoostClassifier(n_estimators=50, learning_rate=1e-2)
model = model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"accuracy：{accuracy_score(y_test, y_pred)}")
# accuracy：0.9666666666666667


# 当然我们这里也可以使用类似随机森林那种，比较不同估计机器数量的涌向绘图
# TODO


# ## 使用GridSearchCV自动调参
hyperparameter_space = {
    "n_estimators": list(range(2, 102, 2)),
    "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
}

# 使用准确率为标准，将得到的准确率最高的参数输出，cv=5表示交叉验证参数，这里使用五折交叉验证，n_jobs=-1表示并行数和cpu一致
gs = GridSearchCV(
    AdaBoostClassifier(algorithm="SAMME.R", random_state=1),
    param_grid=hyperparameter_space,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
)

gs.fit(X_train, y_train)
print(f"最好的超参数：{gs.best_params_}")
# 最好的超参数：{'learning_rate': 0.9, 'n_estimators': 22}

# 当然其实这个因为本身就很高了，可以尝试在原本数据集上再跑一遍，看看准确率会不会再提高
# 一棵树很低，集成，然后再来GridCV  
