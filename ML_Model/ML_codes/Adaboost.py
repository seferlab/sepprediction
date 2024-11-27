import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier  #adaboost引入方法
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def tudou(CSV,target,xianshi=True):
    feature = pd.read_csv(CSV,header=None)

    print(feature.shape)

    # 标准化
    std = StandardScaler()
    feature = std.fit_transform(feature)
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.20)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10)
    # 构建adaboost模型
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),
                               # 可以不写
                             n_estimators=200)
    # 数据量大的时候，可以增加内部分类器的树深度，也可以不限制树深
    # max_depth树深，数据量大的时候，一般范围在10——100之间
    # 数据量小的时候，一般可以设置树深度较小，或者n_estimators较小
    # n_estimators 迭代次数或者最大弱分类器数：200次
    # base_estimator：DecisionTreeClassifier 选择弱分类器，默认为CART树
    # algorithm：SAMME 和SAMME.R 。运算规则，后者是优化算法，以概率调整权重，迭代速度快，
    # 需要能计算概率的分类器支持
    # learning_rate：0<v<=1，默认为1，正则项 衰减指数
    # loss：linear、‘square’exponential’。误差计算公式：一般用linear足够
    bdt.fit(x_train, y_train)


    pred_res = bdt.predict_proba(x_test)[:, 1]
    y_pre = [0 if x < 0.5 else 1 for x in pred_res]

    tn, fp, fn, tp = metrics.confusion_matrix(y_pred=y_pre, y_true=y_test).ravel()
    se = tp / (tp + fn)
    sp = tn / (tn + fp)

    recall = metrics.recall_score(y_pred=y_pre, y_true=y_test)
    precise = metrics.precision_score(y_pred=y_pre, y_true=y_test)
    auc = metrics.roc_auc_score(y_score=pred_res, y_true=y_test)
    acc = metrics.accuracy_score(y_pred=y_pre, y_true=y_test)
    f1 = metrics.f1_score(y_pred=y_pre, y_true=y_test)
    mcc = metrics.matthews_corrcoef(y_pred=y_pre, y_true=y_test)
    ap = metrics.average_precision_score(y_score=pred_res, y_true=y_test)

    if xianshi == True:
        score_train = bdt.score(x_train, y_train)
        score_val = bdt.score(x_val, y_val)
        # print("参数：", lg.coef_)
        # print("截距：", lg.intercept_)
        # 打印召回率，F1
        # print(classification_report(y_test, predict, labels=[0, 1], target_names=["负样本", "正样本"]))
        # print("最好的模型参数：", kn_.best_params_)
        print("在测试集上准确率：", score_train)
        print("在验证集上准确率：", score_val)
        print("训练集上的准确率：", acc)
        print("recall:", recall, flush=True)
        print("precise:", precise, flush=True)
        print("se:", se, flush=True)
        print("sp:", sp, flush=True)
        print("f1:", f1, flush=True)
        print("mcc:", mcc, flush=True)
        print("auc:", auc, flush=True)
        print("ap:", ap, flush=True)



    return  acc,recall,precise,auc,mcc,f1,se,sp


if __name__ == "__main__":
    from statistics import median

    features = '../features/AAC.csv'
    target = pd.read_csv('../features/my.csv',header=None)
    target = target.values.ravel()
    a = []
    for i in range(20):
        acc, _, _, _, _, _, _, _ = tudou(features, target, )
        a.append(acc)

    print('平均值测试集准确率为:', median(a))