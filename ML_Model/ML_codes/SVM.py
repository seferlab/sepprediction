"""
    支持向量机：通过寻找划分超平面来进行分类的算法，这个划分超平面只由支持向量有关，与其他样本无关
"""

from sklearn import metrics
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import numpy as np

def tudou(CSV,target,xianshi=True):

    test_prob = []
    test_true = []
    # 加载数据集
    feature = pd.read_csv(CSV,header=None)
    print(feature.shape)


    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.20)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20)
    # print("训练集：", x_train.shape, y_train.shape)
    # print("验证集：", x_val.shape, y_val.shape)
    # print("测试集：", x_test.shape, y_test.shape)

    # 建立模型
    # gamma=8.2
    clf = SVC( probability=True)
    # clf = SVC( kernel = 'rbf', C = 16, gamma = 0.0313, probability = True)
    # clf = SVC( kernel = 'rbf',  probability = True)
    # param = {'gamma': [0.1,0.01,0.001,1,2,3,4,5,6,7,8,9]}
    # clf = GridSearchCV(clf, param_grid=param,)
    # 训练
    clf.fit(x_train, y_train)


    pred_res = clf.predict_proba(x_test)[:, 1]
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


    name = 'D:/xuexi/ORF/试验/model/HUM_CPPred'
    # name = 'D:/xuexi/ORF/试验/First_kind/RAT/SVM_RAT_SEPs_sORFs_Finally_features'

    np.save(name + 'test_prob.npy', test_prob)
    np.save(name + 'test_true.npy', test_true)

    if xianshi == True:
        score_train = clf.score(x_train, y_train)
        score_val = clf.score(x_val, y_val)
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
    from statistics import mean

    features = 'D:/xuexi/ORF/试验/features/AAC.csv'
    target = pd.read_csv('D:/xuexi/ORF/试验/features/human_label.csv',header=None)
    print(target.shape)
    target = target.values.ravel()
    a = []
    for i in range(50):
        acc, _, _, _, _, _, _, _ = tudou(features, target, )
        a.append(acc)

    print('平均值测试集准确率为:', mean(a))

