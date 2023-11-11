# 电影分类 - 从零实现KNN算法，并进行电影数据分类
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 1.准备数据
rawdata = {
    "电影名称": ["无问西东", "后来的我们", "前任3", "红海行动", "唐人街探案", "战狼"],
    "打斗镜头": [1, 5, 12, 108, 112, 115],
    "接吻镜头": [101, 89, 97, 5, 9, 8],
    "电影类型": ["爱情片", "爱情片", "爱情片", "动作片", "动作片", "动作片"],
}
movie_data = pd.DataFrame(rawdata)

plt.scatter(rawdata["打斗镜头"], rawdata["接吻镜头"])
plt.xlabel("打斗")
plt.ylabel("接吻")
plt.grid()
plt.show()
