# 使用svm算法来实现android strace log日志数据集的分类

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report


def svm1():
    df = pd.read_csv('F:\\pycharmproject\\GraduationProject\\data\\feature_data.csv')
    list = df.values
    X = list[:, 0:190]  # 取数据集的特征向量
    Y = list[:, 191]  # 取数据集的标签（类型）
    # print(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=1)
    # SVM 分类器
    clf = svm.SVC(C=0.8, kernel='rbf',gamma=10, decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    y_predict=clf.predict(x_test)
    # print("The scores of train set is %f" % (clf.score(x_train, y_train)))  # 训练集准确率
    # print("The scores of test set is %f" % (clf.score(x_test, y_test)))  # 测试集准确率
    print(classification_report(y_predict, y_test))

if __name__ == "__main__":
    svm1()
