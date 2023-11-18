#!/usr/bin/env python
# coding=utf-8
"""
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-05-22 10:55:13
@LastEditor: John
LastEditTime: 2021-08-27 15:36:18
@Discription:
@Environment: python 3.7.7
"""
# 参考https://github.com/Dod-o/Statistical-Learning-Method_Code/blob/master/KNN/KNN.py

"""
数据集：Mnist
训练集数量：60000
测试集数量：10000（实际使用：200）
------------------------------
运行机器：CPU i7-9750H
超参数：k=25
运行结果：
向量距离使用算法——L2欧式距离
    正确率：0.9698
    运行时长：266.36s
"""

import time
import numpy as np
import sys
import os

# 导入处于不同目录下的Mnist.load_data
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # 获取上级目录
sys.path.append(parent_path)  # 修改sys.path
print()

from datasets.MNIST.raw.load_data import load_local_mnist
# from Mnist.raw.load_data import load_local_mnist




class KNN:
    def __init__(self, x_train, y_train, x_test, y_test, k):
        """
        Args:
            x_train [Array]: 训练集数据
            y_train [Array]: 训练集标签
            x_test [Array]: 测试集数据
            y_test [Array]: 测试集标签
            k [int]: k of kNN
        """
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        # 将输入数据转为矩阵形式，方便运算
        self.x_train_mat, self.x_test_mat = np.mat(self.x_train), np.mat(self.x_test)
        self.y_train_mat, self.y_test_mat = np.mat(self.y_test).T, np.mat(self.y_test).T
        self.k = k

    def _calc_dist(self, x1, x2):
        """计算两个样本点向量之间的距离,使用的是欧氏距离
        :param x1:向量1
        :param x2:向量2
        :return: 向量之间的欧式距离
        """
        return np.sqrt(np.sum(np.square(x1 - x2)))

    def _get_k_nearest(self, x):
        """
        预测样本x的标记。
        获取方式通过找到与样本x最近的topK个点，并查看它们的标签。
        查找里面占某类标签最多的那类标签
        :param trainDataMat:训练集数据集
        :param trainLabelMat:训练集标签集
        :param x:待预测的样本x
        :param topK:选择参考最邻近样本的数目（样本数目的选择关系到正确率，详看3.2.3 K值的选择）
        :return:预测的标记
        """
        # 初始化距离列表，dist_list[i]表示待预测样本x与训练集中第i个样本的距离
        dist_list = [0] * len(self.x_train_mat)

        # 遍历训练集中所有的样本点，计算与x的距离
        for i in range(len(self.x_train_mat)):
            # 获取训练集中当前样本的向量
            x0 = self.x_train_mat[i]
            # 计算向量x与训练集样本x0的距离
            dist_list[i] = self._calc_dist(x0, x)

        # 对距离列表排序并返回距离最近的k个训练样本的下标
        # ----------------优化点-------------------
        # 由于我们只取topK小的元素索引值，所以其实不需要对整个列表进行排序，而argsort是对整个
        # 列表进行排序的，存在时间上的浪费。字典有现成的方法可以只排序top大或top小，可以自行查阅
        # 对代码进行稍稍修改即可
        # 这里没有对其进行优化主要原因是KNN的时间耗费大头在计算向量与向量之间的距离上，由于向量高维
        # 所以计算时间需要很长，所以如果要提升时间，在这里优化的意义不大。
        k_nearest_index = np.argsort(np.array(dist_list))[: self.k]  # 升序排序
        return k_nearest_index

    def _predict_y(self, k_nearest_index):
        # label_list[1]=3，表示label为1的样本数有3个，由于此处label为0-9，可以初始化长度为10的label_list
        label_list = [0] * 10
        for index in k_nearest_index:
            one_hot_label = self.y_train[index]
            number_label = np.argmax(one_hot_label)
            label_list[number_label] += 1
        # 采用投票法，即样本数最多的label就是预测的label
        y_predict = label_list.index(max(label_list))
        return y_predict

    def test(self, n_test=200):
        """
        测试正确率
        :param: n_test: 待测试的样本数
        :return: 正确率
        """
        print("start test")

        # 错误值计数
        error_count = 0
        # 遍历测试集，对每个测试集样本进行测试
        # 由于计算向量与向量之间的时间耗费太大，测试集有6000个样本，所以这里人为改成了
        # 测试200个样本点，若要全跑，更改n_test即可
        for i in range(n_test):
            # print('test %d:%d'%(i, len(trainDataArr)))
            print("test %d:%d" % (i, n_test))
            # 读取测试集当前测试样本的向量
            x = self.x_test_mat[i]
            # 获取距离最近的训练样本序号
            k_nearest_index = self._get_k_nearest(x)
            # 预测输出y
            y = self._predict_y(k_nearest_index)
            # 如果预测label与实际label不符，错误值计数加1
            if y != np.argmax(self.y_test[i]):
                error_count += 1
            print("accuracy=", 1 - (error_count / (i + 1)))

        # 返回正确率
        return 1 - (error_count / n_test)


if __name__ == "__main__":
    k = 25
    start = time.time()
    (x_train, y_train), (x_test, y_test) = load_local_mnist()
    model = KNN(x_train, y_train, x_test, y_test, k)
    accur = model.test()
    end = time.time()
    print("total acc:", accur)
    print("time span:", end - start)


"""
start test
test 0:200
accuracy= 1.0
test 1:200
accuracy= 1.0
test 2:200
accuracy= 1.0
test 3:200
accuracy= 1.0
test 4:200
accuracy= 1.0
test 5:200
accuracy= 1.0
test 6:200
accuracy= 1.0
test 7:200
accuracy= 1.0
test 8:200
accuracy= 1.0
test 9:200
accuracy= 1.0
test 10:200
accuracy= 1.0
test 11:200
accuracy= 1.0
test 12:200
accuracy= 1.0
test 13:200
accuracy= 1.0
test 14:200
accuracy= 1.0
test 15:200
accuracy= 1.0
test 16:200
accuracy= 1.0
test 17:200
accuracy= 1.0
test 18:200
accuracy= 1.0
test 19:200
accuracy= 1.0
test 20:200
accuracy= 1.0
test 21:200
accuracy= 1.0
test 22:200
accuracy= 1.0
test 23:200
accuracy= 1.0
test 24:200
accuracy= 1.0
test 25:200
accuracy= 1.0
test 26:200
accuracy= 1.0
test 27:200
accuracy= 1.0
test 28:200
accuracy= 1.0
test 29:200
accuracy= 1.0
test 30:200
accuracy= 1.0
test 31:200
accuracy= 1.0
test 32:200
accuracy= 1.0
test 33:200
accuracy= 1.0
test 34:200
accuracy= 1.0
test 35:200
accuracy= 1.0
test 36:200
accuracy= 1.0
test 37:200
accuracy= 1.0
test 38:200
accuracy= 1.0
test 39:200
accuracy= 1.0
test 40:200
accuracy= 1.0
test 41:200
accuracy= 1.0
test 42:200
accuracy= 1.0
test 43:200
accuracy= 0.9772727272727273
test 44:200
accuracy= 0.9777777777777777
test 45:200
accuracy= 0.9782608695652174
test 46:200
accuracy= 0.9787234042553191
test 47:200
accuracy= 0.9791666666666666
test 48:200
accuracy= 0.9795918367346939
test 49:200
accuracy= 0.98
test 50:200
accuracy= 0.9803921568627451
test 51:200
accuracy= 0.9807692307692307
test 52:200
accuracy= 0.9811320754716981
test 53:200
accuracy= 0.9814814814814815
test 54:200
accuracy= 0.9818181818181818
test 55:200
accuracy= 0.9821428571428571
test 56:200
accuracy= 0.9824561403508771
test 57:200
accuracy= 0.9827586206896551
test 58:200
accuracy= 0.9830508474576272
test 59:200
accuracy= 0.9833333333333333
test 60:200
accuracy= 0.9836065573770492
test 61:200
accuracy= 0.9838709677419355
test 62:200
accuracy= 0.9841269841269842
test 63:200
accuracy= 0.984375
test 64:200
accuracy= 0.9846153846153847
test 65:200
accuracy= 0.9848484848484849
test 66:200
accuracy= 0.9850746268656716
test 67:200
accuracy= 0.9852941176470589
test 68:200
accuracy= 0.9855072463768116
test 69:200
accuracy= 0.9857142857142858
test 70:200
accuracy= 0.9859154929577465
test 71:200
accuracy= 0.9861111111111112
test 72:200
accuracy= 0.9863013698630136
test 73:200
accuracy= 0.9864864864864865
test 74:200
accuracy= 0.9866666666666667
test 75:200
accuracy= 0.9868421052631579
test 76:200
accuracy= 0.987012987012987
test 77:200
accuracy= 0.9743589743589743
test 78:200
accuracy= 0.9746835443037974
test 79:200
accuracy= 0.975
test 80:200
accuracy= 0.9753086419753086
test 81:200
accuracy= 0.975609756097561
test 82:200
accuracy= 0.9759036144578314
test 83:200
accuracy= 0.9761904761904762
test 84:200
accuracy= 0.9764705882352941
test 85:200
accuracy= 0.9767441860465116
test 86:200
accuracy= 0.9770114942528736
test 87:200
accuracy= 0.9772727272727273
test 88:200
accuracy= 0.9775280898876404
test 89:200
accuracy= 0.9777777777777777
test 90:200
accuracy= 0.978021978021978
test 91:200
accuracy= 0.9782608695652174
test 92:200
accuracy= 0.978494623655914
test 93:200
accuracy= 0.9787234042553191
test 94:200
accuracy= 0.9789473684210527
test 95:200
accuracy= 0.9791666666666666
test 96:200
accuracy= 0.979381443298969
test 97:200
accuracy= 0.9795918367346939
test 98:200
accuracy= 0.9797979797979798
test 99:200
accuracy= 0.98
test 100:200
accuracy= 0.9801980198019802
test 101:200
accuracy= 0.9803921568627451
test 102:200
accuracy= 0.9805825242718447
test 103:200
accuracy= 0.9807692307692307
test 104:200
accuracy= 0.9809523809523809
test 105:200
accuracy= 0.9811320754716981
test 106:200
accuracy= 0.9813084112149533
test 107:200
accuracy= 0.9814814814814815
test 108:200
accuracy= 0.981651376146789
test 109:200
accuracy= 0.9818181818181818
test 110:200
accuracy= 0.9819819819819819
test 111:200
accuracy= 0.9821428571428571
test 112:200
accuracy= 0.9823008849557522
test 113:200
accuracy= 0.9824561403508771
test 114:200
accuracy= 0.9826086956521739
test 115:200
accuracy= 0.9741379310344828
test 116:200
accuracy= 0.9743589743589743
test 117:200
accuracy= 0.9745762711864406
test 118:200
accuracy= 0.9747899159663865
test 119:200
accuracy= 0.975
test 120:200
accuracy= 0.9752066115702479
test 121:200
accuracy= 0.9754098360655737
test 122:200
accuracy= 0.975609756097561
test 123:200
accuracy= 0.9758064516129032
test 124:200
accuracy= 0.976
test 125:200
accuracy= 0.9761904761904762
test 126:200
accuracy= 0.9763779527559056
test 127:200
accuracy= 0.9765625
test 128:200
accuracy= 0.9767441860465116
test 129:200
accuracy= 0.9769230769230769
test 130:200
accuracy= 0.9770992366412213
test 131:200
accuracy= 0.9772727272727273
test 132:200
accuracy= 0.9774436090225564
test 133:200
accuracy= 0.9776119402985075
test 134:200
accuracy= 0.9777777777777777
test 135:200
accuracy= 0.9779411764705882
test 136:200
accuracy= 0.9781021897810219
test 137:200
accuracy= 0.9782608695652174
test 138:200
accuracy= 0.9784172661870504
test 139:200
accuracy= 0.9785714285714285
test 140:200
accuracy= 0.9787234042553191
test 141:200
accuracy= 0.9788732394366197
test 142:200
accuracy= 0.9790209790209791
test 143:200
accuracy= 0.9791666666666666
test 144:200
accuracy= 0.9793103448275862
test 145:200
accuracy= 0.9794520547945206
test 146:200
accuracy= 0.9795918367346939
test 147:200
accuracy= 0.9797297297297297
test 148:200
accuracy= 0.9798657718120806
test 149:200
accuracy= 0.9733333333333334
test 150:200
accuracy= 0.9735099337748344
test 151:200
accuracy= 0.9736842105263158
test 152:200
accuracy= 0.9738562091503268
test 153:200
accuracy= 0.974025974025974
test 154:200
accuracy= 0.9741935483870968
test 155:200
accuracy= 0.9743589743589743
test 156:200
accuracy= 0.9745222929936306
test 157:200
accuracy= 0.9746835443037974
test 158:200
accuracy= 0.9748427672955975
test 159:200
accuracy= 0.975
test 160:200
accuracy= 0.9751552795031055
test 161:200
accuracy= 0.9753086419753086
test 162:200
accuracy= 0.9754601226993865
test 163:200
accuracy= 0.975609756097561
test 164:200
accuracy= 0.9757575757575757
test 165:200
accuracy= 0.9759036144578314
test 166:200
accuracy= 0.9760479041916168
test 167:200
accuracy= 0.9761904761904762
test 168:200
accuracy= 0.9763313609467456
test 169:200
accuracy= 0.9764705882352941
test 170:200
accuracy= 0.9766081871345029
test 171:200
accuracy= 0.9767441860465116
test 172:200
accuracy= 0.976878612716763
test 173:200
accuracy= 0.9770114942528736
test 174:200
accuracy= 0.9771428571428571
test 175:200
accuracy= 0.9715909090909091
test 176:200
accuracy= 0.9717514124293786
test 177:200
accuracy= 0.9719101123595506
test 178:200
accuracy= 0.9720670391061452
test 179:200
accuracy= 0.9722222222222222
test 180:200
accuracy= 0.9723756906077348
test 181:200
accuracy= 0.9725274725274725
test 182:200
accuracy= 0.9726775956284153
test 183:200
accuracy= 0.9728260869565217
test 184:200
accuracy= 0.972972972972973
test 185:200
accuracy= 0.9731182795698925
test 186:200
accuracy= 0.9732620320855615
test 187:200
accuracy= 0.973404255319149
test 188:200
accuracy= 0.9735449735449735
test 189:200
accuracy= 0.9736842105263158
test 190:200
accuracy= 0.9738219895287958
test 191:200
accuracy= 0.9739583333333334
test 192:200
accuracy= 0.9740932642487047
test 193:200
accuracy= 0.9742268041237113
test 194:200
accuracy= 0.9743589743589743
test 195:200
accuracy= 0.9693877551020408
test 196:200
accuracy= 0.9695431472081218
test 197:200
accuracy= 0.9696969696969697
test 198:200
accuracy= 0.9698492462311558
test 199:200
accuracy= 0.97
total acc: 0.97
time span: 90.9543194770813
"""