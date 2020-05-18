# 使用神经网络对特征向量进行处理

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def knn():
    df = pd.read_csv('F:\\pycharmproject\\GraduationProject\\data\\feature_data.csv')
    list = df.values
    # print(df)
    X = list[:, 0:190]  # 取数据集的特征向量
    Y = list[:, 191]  # 取数据集的标签（类型）
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=1)
    # 使用神经网络
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,), random_state=1,max_iter=10000)
    clf.fit(x_train, y_train)
    # 模型效果获取
    # r = clf.score(x_train, y_train)
    # print("R值(准确率):", r)
    y_predict =clf.predict(x_test)
    print(classification_report(y_predict, y_test))

if __name__ == "__main__":
    knn()