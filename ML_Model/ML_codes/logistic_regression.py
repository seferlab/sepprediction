"""
    逻辑回归：将线性回归函数的输出，作为Sigmoid函数的输入，然后输出为0-1之间的
"""

from sklearn import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']#指定默认字体 SimHei为黑体
plt.rcParams['axes.unicode_minus']=False#用来正常显示负号

def tudou(CSV,target,huatu=False,xianshi=True):

    test_prob = []
    test_true = []

    feature = pd.read_csv(CSV,header=None)
    print(feature.shape)



    std = StandardScaler()
    feature= std.fit_transform(feature)

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.20)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10)
    # print("训练集：", x_train.shape, y_train.shape)
    # print("验证集：", x_val.shape, y_val.shape)
    # print("测试集：", x_test.shape, y_test.shape)


    # 建立模型
    lg = LogisticRegression()
    # lg = LogisticRegression()
    # 训练
    lg.fit(x_train, y_train)



    pred_res = lg.predict_proba(x_test)[:, 1]
    y_pre = [0 if x < 0.5 else 1 for x in pred_res]

    test_prob.append(pred_res)
    test_true.append(y_test)



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

    test_prob = np.concatenate(test_prob)
    test_true = np.concatenate(test_true)


    name = 'D:/xuexi/ORF/试验/model/RAT_MiPepid'
    # name = 'D:/xuexi/ORF/试验/First_kind/RAT/logistic_regression_RAT_SEPs_sORFs_Finally_features'

    np.save(name + 'test_prob.npy', test_prob)
    np.save(name + 'test_true.npy', test_true)

    if xianshi==True:
        score_train = lg.score(x_train, y_train)
        score_val = lg.score(x_val, y_val)
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

    # "%.3f" % a
    if huatu==True:
        x_len = range(len(x_test))
        plt.figure(figsize=(42, 21), facecolor='w')
        plt.ylim(-1, 3)
        plt.plot(x_len, y_test, 'ro', markersize=8, zorder=3, label=u'真实值')  # 真实值
        plt.plot(x_len, y_pre, 'go', markersize=16, zorder=2, label=u'预测值,准确率=%.3f' % acc)  # 预测值
        plt.legend(loc='upper left')
        plt.xlabel(u'刻度', fontsize=30)
        plt.ylabel(u'准确率示意图', fontsize=30)

        plt.show()

    return acc, recall, precise, auc, mcc, f1, se, sp


if __name__ == "__main__":
    from statistics import mean
    features = '../features/AAC.csv'
    target = pd.read_csv('../features/my.csv',header=None)
    target = target.values.ravel()
    a=[]
    for i in range(20):
        acc,_,_,_,_,_,_,_=tudou(features, target,)
        a.append(acc)

    print('平均值测试集准确率为:',mean(a))


    #
    #
    # plt.figure(figsize=(10, 10), dpi=100)
    # plt.ylim(0.5, 1)
    # plt.xlim(0, 20)
    # plt.xlabel(u'刻度', fontsize=25)
    # plt.xticks(range(20),range(20) , fontsize=15)
    # plt.title(u'模型预测准确率', fontsize=13)
    # plt.ylabel(u'准确率示意图', fontsize=25)
    # plt.grid(True, linestyle="--", alpha=0.8)
    # plt.plot(range(20), a,markersize=11,color="red",marker=".",lw=2,label="测试集准群率")
    # plt.legend(loc='lower left')
    #
    #
    # plt.show()


# import shap
# # shap.initjs()
# from sklearn import metrics
# from catboost import CatBoostClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
#
# from sklearn.model_selection import GridSearchCV
# import numpy as np
#
# from sklearn.metrics import classification_report
# from sklearn.feature_selection import mutual_info_classif
# def tudou(CSV,target,eva=True):
#     # 加载数据集
#     feature = pd.read_csv(CSV,header=None)
#
#     print(feature.shape)
#
#     #标准化
#     std = StandardScaler()
#     feature = std.fit_transform(feature)
#
#
#     # scores = mutual_info_classif(feature, target)
#     # # 将特征与对应的得分进行配对
#     # feature_scores = list(zip(range(len(scores)), scores))
#     # # 根据特征得分进行排序
#     # feature_scores.sort(key=lambda x: x[1], reverse=True)
#     # # 打印特征得分
#     # selected_features = [fs[0] for fs in feature_scores[:150]]
#     # feature = feature[:, selected_features]
#
#     # 划分数据集
#     x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.20,random_state=8)
#
#     # print("训练集：", x_train.shape, y_train.shape)
#     # print("验证集：", x_val.shape, y_val.shape)
#     # print("测试集：", x_test.shape, y_test.shape)
#     # 建立KNN模型
#     # kn_ = CatBoostClassifier(
#     #     iterations=1800, learning_rate=0.000988457333835635, depth=11, subsample=0.8359313311759509,
#     #     rsm=0.7815456244033883
#     #
#     # )
#     from sklearn.model_selection import KFold
#     k_=10
#     KF = KFold(n_splits=k_)
#
#     acc_all, se_all, sp_all, mcc_all, auc_all = 0, 0, 0, 0, 0
#     acc_test, se_test, sp_test, mcc_test, auc_test = 0, 0, 0, 0, 0
#     yan_li = []
#
#     test_prob = []
#     test_true = []
#
#     for train_index, test_index in KF.split(x_train, y_train):
#
#         X_train, X_test = x_train[train_index], x_train[test_index]
#         Y_train, Y_test = y_train[train_index], y_train[test_index]
#         kn_ = LogisticRegression()
#         kn_.fit(X_train, Y_train)
#         pred_res = kn_.predict_proba(X_test)[:, 1]
#
#         pred_label = [0 if x < 0.5 else 1 for x in pred_res]
#         acc = metrics.accuracy_score(y_true=Y_test, y_pred=pred_label)
#         tn, fp, fn, tp = metrics.confusion_matrix(y_pred=pred_label, y_true=Y_test).ravel()
#         yan_se = tp / (tp + fn)
#         yan_sp = tn / (tn + fp)
#         yan_auc = metrics.roc_auc_score(y_score=pred_res, y_true=Y_test)
#         yan_mcc = metrics.matthews_corrcoef(y_pred=pred_label, y_true=Y_test)
#         yan_li.append(round(acc, 3))
#         acc_all += acc
#         se_all += yan_se
#         sp_all += yan_sp
#         mcc_all += yan_mcc
#         auc_all += yan_auc
#
#
#
#     # explainer = shap.TreeExplainer(kn_)
#     # shap_values = explainer.shap_values(x_train)
#     # shap.force_plot(explainer.expected_value, shap_values[0, :], x_train[0, :],matplotlib=True)
#
#
#
#         pred_res = kn_.predict_proba(x_test)[:, 1]
#         y_pre = [0 if x < 0.5 else 1 for x in pred_res]
#         test_prob.append(pred_res)
#         test_true.append(y_test)
#         tn, fp, fn, tp = metrics.confusion_matrix(y_pred=y_pre, y_true=y_test).ravel()
#         se = tp / (tp + fn)
#         sp = tn / (tn + fp)
#
#         recall = metrics.recall_score(y_pred=y_pre, y_true=y_test)
#         precise = metrics.precision_score(y_pred=y_pre, y_true=y_test)
#         auc = metrics.roc_auc_score(y_score=pred_res, y_true=y_test)
#         acc = metrics.accuracy_score(y_pred=y_pre, y_true=y_test)
#         f1 = metrics.f1_score(y_pred=y_pre, y_true=y_test)
#         mcc = metrics.matthews_corrcoef(y_pred=y_pre, y_true=y_test)
#         ap = metrics.average_precision_score(y_score=pred_res, y_true=y_test)
#         acc_test += acc
#         se_test += se
#         sp_test += sp
#         mcc_test += mcc
#         auc_test += auc
#     if eva:
#         import re
#
#
#         name = 'D:/xuexi/ORF/试验/model/qq'
#
#
#         np.save(name + 'test_prob.npy', test_prob)
#         np.save(name + 'test_true.npy', test_true)
#
#     return  acc_all/k_,se_all/k_,sp_all/k_,mcc_all/k_,auc_all/k_,acc_test/k_, se_test/k_,sp_test/k_,mcc_test/k_,auc_test/k_,yan_li
#
#
#
# if __name__ == "__main__":
#     from statistics import mean,median
#
#     features = '../features/CTDC.csv'
#     target = pd.read_csv('../features/my.csv',header=None)
#     target = target.values.ravel()
#     a = []
#     for i in range(1):
#         acc, se, sp, mcc, auc, test_acc, test_se, test_sp, test_mcc, test_auc, _ = tudou(features, target, )
#         print(acc, se, sp, mcc, auc, test_acc, test_se, test_sp, test_mcc, test_auc, )
#         a.append(acc)
#         print('验证集ACC:{},测试集ACC:{}'.format(acc, test_acc))