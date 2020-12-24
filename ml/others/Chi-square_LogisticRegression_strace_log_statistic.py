# 使用卡方过滤对逻辑回归算法来实现android strace log日志数据集的分类

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def xgb1():
    df = pd.read_csv('H:\\A数据集\\others\\Features_file_csv_OmniDroid_v2\\features_file_ml7_generate.csv',dtype='float32')
    list = df.values
    X = list[1:, 1:45000]  # 取数据集的特征向量
    Y = list[1:, 0]  # 取数据集的标签（类型）
    # 使用卡方过滤
    # model1 = SelectKBest(chi2, k=60)  # 60结果还不错
    # X = model1.fit_transform(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.9, random_state=1)
    # 使用lr
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    print("==========start============")
    lr = LogisticRegression(C=1.0, tol=0.01)
    lr.fit(x_train, y_train)
    y_predict = lr.predict(x_test)
    print(classification_report(y_predict, y_test,digits=5))
    print("==========end============")


if __name__ == "__main__":
    xgb1()
