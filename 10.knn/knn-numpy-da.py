#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   knn-numpy-da.py
@Author  :   Cat 
@Version :   3.11
'''

import numpy as np
import time
import os
import sys

# 导入处于不同目录下的Mnist.load_data
print(sys.path[0])

# os.path.dirname(os.path.dirname(sys.path[0]))


class MyKnn:
    def def __init__(self, x_train, y_train,x_test,y_test,k):
      '''
        Args:
            x_train [Array]: 训练集数据
            y_train [Array]: 训练集标签
            x_test [Array]: 测试集数据
            y_test [Array]: 测试集标签
            k [int]: k of kNN
        '''

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # mat:将输入的二维数组转化为矩阵
        self.x_train_mat = np.mat(self.x_train)
        self.x_test_mat = np.mat(self.x_test)
        self.y_train_mat = np.mat(self.y_train)
        self.y_test_mat = np.mat(self.y_test)
        self.k = k
        
    def cal_distance(self,x1,x2):
        '''计算两个样本点向量之间的距离,使用的是欧氏距离
        :param x1:向量1
        :param x2:向量2
        :return: 向量之间的欧式距离
        '''
        return np.sqrt(np.sum(np.square(x1-x2)))
    
    
    def get_k_nearset(self,x):
        '''
        预测样本x的标记。
        获取方式通过找到与样本x最近的topK个点，并查看它们的标签。
        查找里面占某类标签最多的那类标签
        :param trainDataMat:训练集数据集
        :param trainLabelMat:训练集标签集
        :param x:待预测的样本x
        :param topK:选择参考最邻近样本的数目（样本数目的选择关系到正确率，详看3.2.3 K值的选择）
        :return:预测的标记
        '''
        # 1.初始化距离列表，distance[i]表示带预测样本x域训练集中第i个样本的距离
        distance_list = [0] * len(self.x_train_mat)

        
        # # 遍历训练集中所有的样本点，计算与x的距离
        for i in range(len(self.x_train_mat)):
            cal_distance = self.cal_distance(x,self.x_train_mat[i])
            distance_list[i] = cal_distance
            
        # 对距离列表排序并返回距离最近的k个训练样本的下标
        # ----------------优化点-------------------
        # 由于我们只取topK小的元素索引值，所以其实不需要对整个列表进行排序，而argsort是对整个
        # 列表进行排序的，存在时间上的浪费。字典有现成的方法可以只排序top大或top小，可以自行查阅
        # 对代码进行稍稍修改即可
        # 这里没有对其进行优化主要原因是KNN的时间耗费大头在计算向量与向量之间的距离上，由于向量高维
        # 所以计算时间需要很长，所以如果要提升时间，在这里优化的意义不大。
        
        k_nearest _index = np.argsort(distance_list)[:self.k]
        return k_nearest _index
    
    def predict_y(self,k_nearest_index):
        # label_list[1]=3，表示label为1的样本数有3个，由于此处label为0-9，可以初始化长度为10的label_list
        label_list = [0] * 10
        for index in k_nearest_index:
            one_hot_label = self.y_train[index]
            number_label = np.argmax(one_hot_label) # 查找数组中最大元素索引
            label_list[number_label] += 1
        # 采用投票法
        y_predict = label_list.index(max(label_list))
        return y_predict
    
    
    def test(self,n_test=20):
        '''
        测试正确率
        :param: n_test: 待测试的样本数
        :return: 正确率
        '''
        print('test start')
        # 错误值计数
        error_count = 0
        # 遍历测试集，对每个测试集样本进行测试
        # 由于计算向量与向量之间的时间耗费太大，测试集有6000个样本，所以这里人为改成了
        # 测试200个样本点，若要全跑，更改n_test即可
        for i in range(n_test):
            print(f'test:{i+1},{n_test}')
            # 读取测试集当前测试样本的向量
            current_test = self.x_test_mat[i]
            # 获取距离最近的训练样本序号
            k_nearest_index = self.get_k_nearest_index(current_test)
            # 预测输出k
            y_pred = self.predict_y(k_nearest_index)
            
            # 如果与预测值不符合，error + 1
            if y_pred != np.argmax(self.y_test[i]):
                error_count +=1
        error = error_count / n_test
        print(f'accuracy:{1 - error}')
        
if __name__ == "__main__":
    k = 25
    start = time.time()
    
    (x_train, y_train), (x_test, y_test) = load_local_mnist()
    model=KNN( x_train, y_train, x_test, y_test,k)
    accur=model.test()
    end = time.time()
    print("total acc:",accur)
    
    
    
    print(f'time:{time.time() - start}')
            
            
## 伪代码

# > 抽空重写

# 对未知类别属性的数据集中的每个点依次执行以下操作：
# (1) 计算已知类别数据集中的点与当前点之间的距离；
# (2) 按照距离递增次序排序；
# (3) 选取与当前点距离最小的k个点(k<20)；
# (4) 确定前k个点所在类别的出现频率；
# (5) 返回前k个点出现频率最高的类别作为当前点的预测分类。