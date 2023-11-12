#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   demo.py
@Author  :   Cat 
@Version :   3.11
"""


# boston房价预测

# m1:降低版本
# m2:方案二如下所示
"""
# 移除之后的另一种解决方案
import pandas as pd
import numpy as np
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
"""

# 方案三
from sklearn.datasets import fetch_openml

data_x, data_y = fetch_openml(
    name="boston", version=1, as_frame=True, return_X_y=True, parser="pandas"
)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=0.2, random_state=1
)

lr = LinearRegression()

lr.fit(X_train, y_train)

# k和b
print(lr.intercept_)
print(lr.coef_)

# 查看斜率
coeff_df = pd.DataFrame(lr.coef_, data_x.keys(), columns=["Coefficient"])
print(coeff_df)

# 斜率也就是我们的特征系数，所以每一个特征都会有一个系数。如果系数是正的，说明这个属性对房价提升有帮助；如果系数是负的，说明这个属性会导致房价下跌。
# print(lr.score(X_test,y_test))

x_numpy_test = X_test.to_numpy()
y_pred = lr.predict(x_numpy_test)
print(y_pred.shape)

y_numpy_test = y_test.to_numpy()

test_df = pd.DataFrame({"Actual": y_numpy_test, "Predicted": y_pred})
print(test_df)

# 当然，在 sklearn 中也已经有了封装好的方法供我们调用，具体可以参见下面的代码。
print('MeanAbsolute Error:', metrics.mean_absolute_error(y_test, y_pred))#MAE
print('MeanSquared Error:', metrics.mean_squared_error(y_test, y_pred))#MSE
print('RootMean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))#RMSE


# 画图
def plot_res():
    # test_df = test_df.head(102)
    test_df.plot(kind='bar',figsize=(24,16))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    
    # 当然我们也可以使用matplotlib绘图，没有关系
    
plot_res()
    
    

