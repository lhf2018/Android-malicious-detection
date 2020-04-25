# 使用svm算法来实现wdbc数据集的分类

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

df=pd.read_csv('F:\\pycharmproject\\GraduationProject\\data\\svmdata\\wdbc_csv.csv')
list=df.values
# print(df)
X = list[:,0:30] # 取数据集的特征向量
Y = list[:,30] # 取数据集的标签（类型）
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.8, random_state = 1)
# SVM 分类器
clf=svm.SVC(C = 0.8, kernel = 'linear', decision_function_shape = 'ovr')
clf.fit(x_train, y_train.ravel())
print("The scores of train set is %f" %(clf.score(x_train, y_train))) # 训练集准确率
print("The scores of test set is %f" %(clf.score(x_test, y_test))) # 测试集准确率
