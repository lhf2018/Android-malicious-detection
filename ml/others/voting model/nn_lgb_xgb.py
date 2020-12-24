import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def knn():
    starttime = datetime.datetime.now()
    df = pd.read_csv('H:\\A数据集\\others\\ring - 副本.csv')
    list = df.values
    # print(df)
    X = list[:, 0:19]  # 取数据集的特征向量
    Y = list[:, 20]  # 取数据集的标签（类型）
    # 使用卡方gbdt_lgb_xgb.py过滤
    # model1 = SelectKBest(chi2, k=5000)  # 60结果还不错
    # X = model1.fit_transform(X, Y)
    # 使用标准化
    ss = StandardScaler()
    X = ss.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, random_state=0)
    # 机器学习算法
    # xgb
    xgbr = xgb.XGBClassifier(n_estimators=300, scale_pos_weight=2, max_depth=10, min_child_weight=5)
    xgbr.fit(x_train, y_train)
    # lgb
    gbm = lgb.LGBMClassifier(num_leaves=50, learning_rate=0.01, n_estimators=500)
    gbm.fit(x_train, y_train)
    # nn
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1, max_iter=1000)
    mlp.fit(x_train, y_train)
    y_predict1 = xgbr.predict(x_test)
    y_predict2 = gbm.predict(x_test)
    y_predict3 = mlp.predict(x_test)
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
