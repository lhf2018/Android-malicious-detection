# 使用卡方过滤对lgb算法来实现features_file_ml7_generate数据集的分类
# 使用matplotlib绘制验证曲线（n_neighbors）

import datetime

import lightgbm as lgb
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def ml():
    starttime = datetime.datetime.now()
    df = pd.read_csv('H:\\A数据集\\others\\Features_file_csv_OmniDroid_v2\\features_file_ml7_generate.csv',dtype='float32')
    list = df.values
    # print(df)
    X = list[1:, 1:45000]  # 取数据集的特征向量
    Y = list[1:, 0]  # 取数据集的标签（类型）
    # 使用卡方过滤
    model1 = SelectKBest(chi2, k=2000)  # 60结果还不错
    X = model1.fit_transform(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, random_state=0)
    # 使用xgb
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    print("==========start============")
    gbm = lgb.LGBMClassifier(num_leaves=50, learning_rate=0.02, n_estimators=2000)
    gbm.fit(x_train, y_train)
    y_predict = gbm.predict(x_test)
    print(classification_report(y_predict, y_test,digits=5))
    # print(gbm.score(x_test,y_test))
    print("==========end============")
    endtime = datetime.datetime.now()
    print(endtime - starttime)


if __name__ == "__main__":
    ml()
