# 使用卡方过滤对xgb算法来实现android strace log日志数据集的分类
# 使用matplotlib绘制验证曲线
import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler


def xgb1():
    starttime = datetime.datetime.now()
    df = pd.read_csv('H:\\A数据集\\others\\ring - 副本.csv')
    list = df.values
    # print(df)
    X = list[:, 0:19]  # 取数据集的特征向量
    Y = list[:, 20]  # 取数据集的标签（类型）
    # 使用卡方过滤
    # model1 = SelectKBest(chi2, k=60)  # 60结果还不错
    # X = model1.fit_transform(X, Y)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.75, random_state=5)

    # 使用xgb
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    print("==========start============")

    # xgbr = xgb.XGBClassifier(n_estimators=800,learning_rate=0.2,min_child_weight=1,max_depth=8,gamma=1,colsample_bytree=0.5,scale_pos_weight=1)
    xgbr=xgb.XGBClassifier(n_estimators=800,learning_rate=0.2,min_child_weight=1,max_depth=8
                      ,gamma=1,colsample_bytree=0.5,scale_pos_weight=1)
    # PCA
    startPCAtime = datetime.datetime.now()
    estimator = PCA(n_components=18)
    x_train = estimator.fit_transform(x_train)
    x_test = estimator.transform(x_test)
    endPCAtime = datetime.datetime.now()

    xgbr.fit(x_train, y_train)
    startdectecttime = datetime.datetime.now()
    y_predict = xgbr.predict(x_test)
    enddectecttime = datetime.datetime.now()
    print(classification_report(y_predict, y_test,digits=6))
    # print(xgbr.score(x_test,y_test))
    # precision, recall, thresholds = precision_recall_curve(
    # y_test, y_predict)
    # plt.figure("P-R Curve")
    # plt.title('Precision/Recall Curve')
    # plt.xlabel('Recall')
    # plt.ylabel('Precision')
    # plt.plot(recall, precision)
    # plt.show()
    endtime = datetime.datetime.now()
    print(endtime - starttime)
    print(enddectecttime - startdectecttime)
    print(endPCAtime - startPCAtime)
    print("==========end============")
    return
    for index in range(1,20):
        # if index<103:
        #     index+=1
        #     continue
        # select features using threshold
        estimator = PCA(n_components=index)
        select_X_train = estimator.fit_transform(x_train)
        # train model
        selection_model = xgb.XGBClassifier(n_estimators=800,learning_rate=0.2,min_child_weight=1,max_depth=8
                      ,gamma=1,colsample_bytree=0.5,scale_pos_weight=1)
        selection_model.fit(select_X_train, y_train)
        # eval model
        select_X_test = estimator.transform(x_test)
        y_pred = selection_model.predict(select_X_test)
        # predictions = [round(value) for value in y_pred]
        accuracy = accuracy_score(y_test, y_pred)
        print("n=%d, Accuracy: %.4f%%" % (index, accuracy * 100.0))
    return
    ##绘制roc曲线
    plt.figure("ROC Curve")
    fpr, tpr, threshold = roc_curve(y_test, y_predict)
    plt.plot(fpr, tpr, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.show()
    return
    # 绘制图像
    param_range = np.arange(0, 5, 0.5)
    train_scores, test_scores = validation_curve(xgbr, X, Y,param_name='scale_pos_weight', param_range=param_range, cv=10)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with XGBOOST")
    plt.xlabel("$\gamma$")
    plt.ylabel("Score")
    plt.xlabel("scale_pos_weight")
    plt.ylim(0.0, 1.1)
    plt.xticks(param_range)
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
    xgb1()
