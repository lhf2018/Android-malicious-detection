# 使用卡方过滤对knn算法来实现android strace log日志数据集的分类
# 使用matplotlib绘制验证曲线（n_neighbors）

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier
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
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=1)
    # 使用knn
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    knc = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
    knc.fit(x_train, y_train)
    y_predict = knc.predict(x_test)
    # print(knc.score(x_test,y_test))
    # print(knc.score(x_train,y_train))
    print(classification_report(y_predict, y_test))
    # 绘制图像
    param_range = np.arange(1, 100, 10)
    train_scores, test_scores = validation_curve(KNeighborsClassifier(metric='minkowski')
                                                 , X, Y, param_name="n_neighbors", param_range=param_range, cv=10)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with KNN")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.xlabel("n_neighbors")
    plt.ylim(0.0, 1.1)
    plt.xticks(np.arange(0, 100, 10))
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
