# 用这个

import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def knn():
    starttime = datetime.datetime.now()
    df = pd.read_csv('H:\\A数据集\\others\\ring - 副本.csv')
    list = df.values
    X = list[:, 0:19]  # 取数据集的特征向量
    Y = list[:, 20]  # 取数据集的标签（类型）
    # 使用卡方过滤
    # model1 = SelectKBest(chi2, k=500)  # 60结果还不错
    # X = model1.fit_transform(X, Y)
    # 使用标准化
    ss = StandardScaler()
    X = ss.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, random_state=1)
    # 机器学习算法
    lr = LogisticRegression(C=1.0, tol=0.02)
    lr.fit(x_train, y_train)

    gbm = lgb.LGBMClassifier(num_leaves=200, learning_rate=0.02, n_estimators=500)
    gbm.fit(x_train, y_train)

    gbr = GradientBoostingClassifier(n_estimators=300, max_depth=2, min_samples_split=2, learning_rate=0.1)
    gbr.fit(x_train, y_train)
    y_predict1 = lr.predict(x_test)
    y_predict2 = gbm.predict(x_test)
    y_predict3 = gbr.predict(x_test)
    y_predict4=np.array(y_predict1+y_predict2+y_predict3).tolist()
    endtime = datetime.datetime.now()
    for i, val in enumerate(y_predict4):
        if val<2:
            y_predict4[i]=0
        else:
            y_predict4[i]=1
    print(endtime - starttime)
    print(classification_report(np.array(y_predict4), y_test, digits=5))


if __name__ == "__main__":
    knn()
