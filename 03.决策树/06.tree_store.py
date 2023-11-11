#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    :   06.tree_store.py
@Author  :   Cat 
@Version :   3.11
"""

# 五、决策树的存储
# 构造决策树是很耗时的任务，即使处理很小的数据集，如前面的样本数据，也要花费几秒的时间，如果数据集很大，将会耗费很多计算时间。然而用创建好的决策树解决分类问题，则可以很快完成。因此，为了节省计算时间，最好能够在每次执行分类时调用已经构造好的决策树。为了解决这个问题，需要使用Python模块pickle序列化对象。序列化对象可以在磁盘上保存对象，并在需要的时候读取出来。

# 假设我们已经得到决策树{'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}，使用pickle.dump存储决策树。

import pickle


def store_tree(inputTree, filepath):
    with open(filepath, "wb") as file:
        pickle.dump(inputTree, file)


# if __name__ == "__main__":
#     myTree = {"有自己的房子": {0: {"有工作": {0: "no", 1: "yes"}}, 1: "yes"}}
    
#     store_tree(myTree,'treeStore.txt')
    
    
# 一串串二进制代码
    
"""看不懂？没错，因为这个是个二进制存储的文件，我们也无需看懂里面的内容，会存储，会用即可。那么问题来了。将决策树存储完这个二进制文件，然后下次使用的话，怎么用呢？

很简单使用pickle.load进行载入即可，编写代码如下："""


# 如何使用呢？

def grobTree(filepath):
    with open(filepath,'rb') as file:
        res = pickle.load(file)
        print(res)
        # {'有自己的房子': {0: {'有工作': {0: 'no', 1: 'yes'}}, 1: 'yes'}}
        
if __name__ == "__main__":
    grobTree('./treeStore.txt')
