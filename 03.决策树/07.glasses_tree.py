#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   03.glasses_tree.py
@Author  :   Cat 
@Version :   3.11
'''



# 决策树实战篇之为自己配个隐形眼镜


# 进入本文的正题：眼科医生是如何判断患者需要佩戴隐形眼镜的类型的？一旦理解了决策树的工作原理，我们甚至也可以帮助人们判断需要佩戴的镜片类型。

# 隐形眼镜数据集是非常著名的数据集，它包含很多换着眼部状态的观察条件以及医生推荐的隐形眼镜类型。隐形眼镜类型包括硬材质(hard)、软材质(soft)以及不适合佩戴隐形眼镜(no lenses)。

# 数据来源与UCI数据库，数据集下载地址：https://github.com/Jack-Cherish/Machine-Learning/blob/master/Decision%20Tree/classifierStorage.txt

# 一共有24组数据，数据的Labels依次是age、prescript、astigmatic、tearRate、class，也就是第一列是年龄，第二列是症状，第三列是是否散光，第四列是眼泪数量，第五列是最终的分类标签。数据如下图所示：


import pandas as pd
from six import StringIO
import pydotplus
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def main1(filepath):
    f = open(filepath)
    res = [line.strip().split('\t') for line in f.readlines()]
    print(res)
    tree = DecisionTreeClassifier()
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    
    cls = tree.fit(res,lensesLabels)
    
    # ValueError: could not convert string to float: 'young'
    
    """我们可以看到程序报错了，这是为什么？因为在fit()函数不能接收string类型的数据，通过打印的信息可以看到，数据都是string类型的。
    在使用fit()函数之前，我们需要对数据集进行编码，这里可以使用两种方法："""
    
    
    """LabelEncoder ：将字符串转换为增量值
    OneHotEncoder：使用One-of-K算法将字符串转换为整数"""
    
    """为了对string类型的数据序列化，需要先生成pandas数据，这样方便我们的序列化工作。这里我使用的方法是，原始数据->字典->pandas数据，编写代码如下："""

    
def main2(filepath):
    with open(filepath,'r') as file:
        res = [line.strip().split('\t') for line in file.readlines()]
    target = []
    print(res)
    
    for item in res:
        target.append(item[-1])
        
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']            #特征标签       
    lenses_list = []                                                        #保存lenses数据的临时列表
    lenses_dict = {}  
    
    for label in lensesLabels:
        for each in res:
            # lenses_list.append(each(lensesLabels.index(label)))
            lenses_list.append(each[lensesLabels.index(label)])
        lenses_dict[label] = lenses_list
        lenses_list = []
        
    print(lenses_dict)                                                        #打印字典信息
    lenses_pd = pd.DataFrame(lenses_dict)                                    #生成pandas.DataFrame
    print(lenses_pd)
    
    
def main3(filepath):
    with open(filepath, 'r') as fr:                                        #加载文件
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]        #处理文件
    lenses_target = []                                                        #提取每组数据的类别，保存在列表里
    for each in lenses:
        lenses_target.append(each[-1])
 
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']            #特征标签       
    lenses_list = []                                                        #保存lenses数据的临时列表
    lenses_dict = {}                                                        #保存lenses数据的字典，用于生成pandas
    for each_label in lensesLabels:                                            #提取信息，生成字典
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # print(lenses_dict)                                                        #打印字典信息
    lenses_pd = pd.DataFrame(lenses_dict)                                    #生成pandas.DataFrame
    print(lenses_pd)                                                        #打印pandas.DataFrame
    le = LabelEncoder()                                                        #创建LabelEncoder()对象，用于序列化            
    for col in lenses_pd.columns:                                            #为每一列序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)
    
    
    
def main(filepath):
    with open(filepath, 'r') as fr:                                        #加载文件
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]        #处理文件
    lenses_target = []                                                        #提取每组数据的类别，保存在列表里
    for each in lenses:
        lenses_target.append(each[-1])
    print(lenses_target)
 
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']            #特征标签       
    lenses_list = []                                                        #保存lenses数据的临时列表
    lenses_dict = {}                                                        #保存lenses数据的字典，用于生成pandas
    for each_label in lensesLabels:                                            #提取信息，生成字典
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # print(lenses_dict)                                                        #打印字典信息
    lenses_pd = pd.DataFrame(lenses_dict)                                    #生成pandas.DataFrame
    # print(lenses_pd)                                                        #打印pandas.DataFrame
    le = LabelEncoder()                                                        #创建LabelEncoder()对象，用于序列化           
    for col in lenses_pd.columns:                                            #序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # print(lenses_pd)                                                        #打印编码信息
 
    clf = DecisionTreeClassifier(max_depth = 4)                        #创建DecisionTreeClassifier()类
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)                    #使用数据，构建决策树
    dot_data = StringIO()
    clf.export_graphviz(clf, out_file = dot_data,                            #绘制决策树
                        feature_names = lenses_pd.keys(),
                        class_names = clf.classes_,
                        filled=True, rounded=True,
                        special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")   
    
    
if __name__ == "__main__":
    filepath = './lenses.txt'
    main2(filepath)
    
    
