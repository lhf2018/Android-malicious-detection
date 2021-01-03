#coding:utf-8
#基于树的预测模型（见 sklearn.tree 模块，森林见 sklearn.ensemble 模块）能够用来计算特征的重要程度，因此能用来去除不相关的特征
# https://www.cnblogs.com/stevenlk/p/6543628.html#43-%E5%9F%BA%E4%BA%8E%E6%A0%91%E7%9A%84%E7%89%B9%E5%BE%81%E9%80%89%E6%8B%A9-tree-based-feature-selection
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def xgb1():
    # pyplot.rc("font", family='YouYuan', weight="light")
    df = pd.read_csv('H:\\A数据集\\others\\ring - 副本.csv')
    list = df.values
    # print(df)
    X = list[:, 0:19]  # 取数据集的特征向量
    Y = list[:, 20]  # 取数据集的标签（类型）
    # PCA
    # estimator = PCA(n_components=10)
    # X=estimator.fit_transform(X)
    # 使用卡方过滤
    # model1 = SelectKBest(chi2, k=100)  # 60结果还不错
    # X = model1.fit_transform(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, random_state=0)
    # 使用xgb
    # ss = StandardScaler()
    # x_train = ss.fit_transform(x_train)
    # x_test = ss.transform(x_test)
    selection_model = ExtraTreesClassifier(n_estimators=30)
    selection_model.fit(x_train, y_train)
    y_p=selection_model.predict(x_test)
    print(classification_report(y_p, y_test, digits=5))
    print("==========start============")
    for index in range(1,20):
        # if index<103:
        #     index+=1
        #     continue
        # select features using threshold
        estimator = PCA(n_components=index)
        select_X_train = estimator.fit_transform(x_train)
        # train model
        selection_model = ExtraTreesClassifier(n_estimators=30)
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = estimator.transform(x_test)
        y_pred = selection_model.predict(select_X_test)
        # predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, y_pred)
        print("n=%d, Accuracy: %.2f%%" % (index, accuracy * 100.0))


    # plot_importance(xgbr,max_num_features=20,grid=False,importance_type='gain',title='特征重要性',xlabel='信息熵值',ylabel='特征序号')
    # # pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    # pyplot.show()
    # print(classification_report(y_predict, y_test,digits=5))
    # xgb.plot_importance(model,height=0.5,max_num_features=25)
    # plt.show()
    print("==========end============")

if __name__ == "__main__":
    xgb1()
