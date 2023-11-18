训练数据量小，模型在训练样本上能收敛，但预测准确率很低

解决方案：
1.标注更多的数据
2.尝试构造训练样本（数据增强）
	给chatGPT 生成一些数据
3.更换模型（如使用预训练模型等）减少数据需求
	cnn --> bert
4.增加规则弥补
	常用
5.调整阈值，用召回率换准确率
	宁可不答也不要错（有些场景）
6.重新定义类别（减少类别）





标签不均衡问题

部分类别样本充裕，部分类别样本极少

类别	金融	体育	科技	生活	教育
样本数量	3000	3200	2800	3100	50


解决办法：
    解决数据稀疏的所有的方法依然适用
    1.过采样       复制指定类别的样本，在采样中重复 （50 -->  500）
    2.降采样      减少多样本类别的采样，随机使用部分
    3.调整样本权重     通过损失函数权重调整来体现



常见报错：




常见警告：
"""C:\Users\hechang\Desktop\watermelon\guaml\Lib\site-packages\sklearn\ensemble\_base.py:156: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4."""

解决方案：