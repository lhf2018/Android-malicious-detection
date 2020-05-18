"""具有读取功能，具有绘制图象功能，具有聚类实现"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans

# 读入数据
df = pd.read_csv('F:\\pycharmproject\\GraduationProject\\data\\svmdata\\iris_data.csv')
df.columns = ['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度', '鸢尾种类']
attribute = df[['萼片长度', '萼片宽度', '花瓣长度', '花瓣宽度']]
# print(df['鸢尾种类'])
# print(attribute)

list = attribute.values
print(list)


# 聚类前有标签的情况下进行绘图
def scatter_plot_by_category(feat, x, y):
    alpha = 0.5
    gs = df.groupby(feat)
    cs = cm.rainbow(np.linspace(0, 1, len(gs)))
    for g, c in zip(gs, cs):
        plt.scatter(g[1][x], g[1][y], color=c, alpha=alpha)


plt.figure(figsize=(20, 5))
plt.subplot(131)
# scatter_plot_by_category('鸢尾种类', '萼片长度', '花瓣长度')
# plt.xlabel('萼片长度')
# plt.ylabel('花瓣长度')
# plt.title('鸢尾种类')
# plt.show()

# scatter_plot_by_category('鸢尾种类', '萼片长度', '花瓣宽度')
# plt.xlabel('sepal_lenth')
# plt.ylabel('petal_width')
# plt.title('class')
# plt.show()

# 聚类前进行绘图画图方法2

plt.scatter(list[:, 0], list[:, 1], c="red", marker='o', label='see')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

# 进行聚类
estimator = KMeans(n_clusters=3)  # 构造聚类器
estimator.fit(list)  # 聚类
label_pred = estimator.labels_  # 获取聚类标签
# 绘制k-means结果
x0 = list[label_pred == 0]
x1 = list[label_pred == 1]
x2 = list[label_pred == 2]
plt.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()
