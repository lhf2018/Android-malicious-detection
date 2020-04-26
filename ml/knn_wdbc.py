# 使用knn算法来实现wdbc数据集的分类

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def knn():
    df = pd.read_csv('F:\\pycharmproject\\GraduationProject\\data\\svmdata\\wdbc_csv.csv')
    list = df.values
    # print(df)
    X = list[:, 0:30]  # 取数据集的特征向量
    Y = list[:, 30]  # 取数据集的标签（类型）
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=1)
    # 使用knn
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    knc = KNeighborsClassifier()
    knc.fit(x_train, y_train)
    y_predict = knc.predict(x_test)
    # print(knc.score(x_test,y_test))
    print(classification_report(y_predict, y_test))


if __name__ == "__main__":
    knn()
