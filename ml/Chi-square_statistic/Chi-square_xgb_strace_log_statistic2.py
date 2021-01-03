#coding:utf-8
# 使用卡方过滤对xgb算法来实现android strace log日志数据集的分类
# 使用matplotlib绘制验证曲线（n_neighbors）
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel
from numpy import sort
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def xgb1():
    # pyplot.rc("font", family='YouYuan', weight="light")
    df = pd.read_csv('F:\\pycharmproject\\GraduationProject\\data\\feature_data_new_statistic_part.csv')
    list = df.values
    # print(df)
    X = list[:, 0:191]  # 取数据集的特征向量
    Y = list[:, 192]  # 取数据集的标签（类型）
    # 使用卡方过滤
    # model1 = SelectKBest(chi2, k=60)  # 60结果还不错
    # X = model1.fit_transform(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, random_state=0)
    # 使用xgb
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    print("==========start============")
    xgbr = xgb.XGBClassifier(n_estimators=30,scale_pos_weight=2,max_depth=10,min_child_weight=5)
    xgbr.fit(x_train, y_train)
    y_predict = xgbr.predict(x_test)

    predictions=[round(value) for value in y_predict]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    thresholds = sort(xgbr.feature_importances_)
    # print(thresholds)
    index=0
    for thresh in thresholds:
        if index<103:
            index+=1
            continue
        # select features using threshold
        selection = SelectFromModel(xgbr, threshold=thresh, prefit=True)
        select_X_train = selection.transform(x_train)
        # train model
        selection_model = xgb.XGBClassifier(scale_pos_weight=2,max_depth=10,min_child_weight=5)
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = selection.transform(x_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, predictions)
        print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy * 100.0))


    # plot_importance(xgbr,max_num_features=20,grid=False,importance_type='gain',title='特征重要性',xlabel='信息熵值',ylabel='特征序号')
    # # pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    # pyplot.show()
    # print(classification_report(y_predict, y_test,digits=5))
    # xgb.plot_importance(model,height=0.5,max_num_features=25)
    # plt.show()
    print("==========end============")

if __name__ == "__main__":
    xgb1()
