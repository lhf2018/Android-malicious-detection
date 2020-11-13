# 使用卡方过滤对神经网络算法来实现android strace log日志数据集的分类
# 使用matplotlib绘制验证曲线（alpha、hidden_layer_size）

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import validation_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


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
    # param_range = np.arange(1, 11, 1)
    param_range1 = np.arange(1, 11, 1)
    param_range = ((1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,))
    # param_range = np.logspace(-6,-4 ,10)
    train_scores, test_scores = validation_curve(
        MLPClassifier(solver='lbfgs',
                      alpha=1e-5,
                      random_state=1, max_iter=100000)
        , X, Y, param_name="hidden_layer_sizes",
        param_range=param_range, cv=10)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.title("Validation Curve with MLPClassifier")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.xlabel("hidden_layer_sizes")
    plt.ylim(0.0, 1.1)
    # plt.xticks(np.arange(0.1, 1, 0.1))
    # plt.xticks(np.logspace(-8,-3 ,1))
    plt.xticks(np.arange(1, 11, 1))
    lw = 2
    # 半对数坐标函数：只有一个坐标轴是对数坐标，另一个是普通算术坐标
    # plt.semilogx(param_range, train_scores_mean, label="Training score",
    #              color="darkorange", lw=lw)
    plt.plot(param_range1, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
    # 在区域内绘制函数包围的区域
    plt.fill_between(param_range1, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    # plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
    #              color="navy", lw=lw)
    plt.plot(param_range1, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
    plt.fill_between(param_range1, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()


if __name__ == "__main__":
    knn()
