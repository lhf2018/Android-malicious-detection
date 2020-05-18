# 使用卡方过滤对svm算法来实现android strace log日志数据集的分类
# 使用matplotlib绘制验证曲线（C、gamma）

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve


def knn():
    df = pd.read_csv('F:\\pycharmproject\\GraduationProject\\data\\feature_data_new_statistic_part.csv')
    list = df.values
    # print(df)
    X = list[:, 0:191]  # 取数据集的特征向量
    Y = list[:, 192]  # 取数据集的标签（类型）
    # 使用卡方过滤
    model1 = SelectKBest(chi2, k=60)  # 60结果还不错
    X = model1.fit_transform(X, Y)
    # 使用标准化
    ss = StandardScaler()
    X = ss.fit_transform(X)
    # 绘制图像
    param_range = np.arange(0.1, 1, 0.1)
    train_scores, test_scores = validation_curve(svm.SVC(gamma=20,kernel='rbf', decision_function_shape='ovo',max_iter=-1)
                                                 , X, Y, param_name="C", param_range=param_range, cv=10)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve with SVM")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.xlabel("惩罚系数C")
    plt.ylim(0.0, 1.1)
    plt.xticks(np.arange(0.1, 1, 0.1))
    lw = 2
    # 半对数坐标函数：只有一个坐标轴是对数坐标，另一个是普通算术坐标
    # plt.semilogx(param_range, train_scores_mean, label="Training score",
    #              color="darkorange", lw=lw)
    plt.plot(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
    # 在区域内绘制函数包围的区域
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    # plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
    #              color="navy", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    knn()
