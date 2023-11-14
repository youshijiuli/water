from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn import datasets
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# 加载数据
digits = datasets.load_digits()
print(type(digits))
# print(dir(digits))
# ['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']

d_data = digits["data"]
print(len(d_data))
# 可视化观察前10个样本的效果（0~9的数字）
for i in range(1, 11):
    # subplot的索引从1开始
    plt.subplot(2, 5, i)
    # 将画布拆分为2行5列，依次定位到每个axes
    plt.imshow(digits.images[i - 1], cmap=plt.cm.gray_r)

    # 辅助对象处理
    plt.xticks([])  # 隐藏x轴坐标刻度和刻度标签
    plt.yticks([])
    plt.text(3, 10, digits.target[i - 1])
    # 将每个图像对应的数字写在图像下方
plt.show()

# 拆分数据集
train_X, test_X, train_y, test_y = train_test_split(
    d_data, digits["target"], test_size=0.25, random_state=123456
)

# 5.对数据进行标准化处理以便计算距离
ss = preprocessing.StandardScaler()
train_ss_X = ss.fit_transform(train_X)
test_ss_X = ss.fit_transform(test_X)


# 模型训练和评估
knn_model = KNeighborsClassifier()
knn_model.fit(train_ss_X, train_y)

# 在测试集上面预测
pred_y = knn_model.predict(test_ss_X)

# 正确率
print("KNN模型总体正确率:", metrics.accuracy_score(pred_y, test_y))
# KNN模型总体正确率: 0.9733333333333334


# 使用分类评估报告观察模型效果
print(metrics.classification_report(pred_y, test_y))
