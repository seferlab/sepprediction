"""
    KNN算法也叫做K近邻算法，它的主要思想是：
        计算测试样本与训练集中各个样本之间的距离，选择与测试样本距离最近的K个，然后统计这K个样本中出现标记最多的那个，
        将这个标记作为测试样本的标记
"""
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
import numpy as np

from sklearn.metrics import classification_report
def tudou(CSV,target,xianshi=True):
    test_prob = []
    test_true = []
    # 加载数据集
    feature = pd.read_csv(CSV,header=None)

    print(feature.shape)

    #标准化
    std = StandardScaler()
    feature = std.fit_transform(feature)


    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.20)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10)
    # print("训练集：", x_train.shape, y_train.shape)
    # print("验证集：", x_val.shape, y_val.shape)
    # print("测试集：", x_test.shape, y_test.shape)
    # 建立KNN模型
    # kn_ = GradientBoostingClassifier( n_estimators=2000, learning_rate=0.028673784262499755,
    #                                 max_depth=11, min_samples_leaf=59, min_samples_split=19,
    #                                 subsample=0.6447933070462003, max_features=0.0001)
    kn_ = GradientBoostingClassifier()
    # 训练
    kn_.fit(x_train, y_train)

    pred_res = kn_.predict_proba(x_test)[:, 1]
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

    # name = 'D:/xuexi/ORF/试验/test_model/RAT_Amino Acid-Nucleotide fused features'
    # name = 'D:/xuexi/ORF/试验/test_model/RAT_Amino Acid fused features'
    name = 'D:/xuexi/ORF/试验/First_kind/RAT/TESTGDBT_RAT_SEPs_sORFs_Finally_features'

    np.save(name + 'test_prob.npy', test_prob)
    np.save(name + 'test_true.npy', test_true)

    if xianshi == True:
        score_train = kn_.score(x_train, y_train)
        score_val = kn_.score(x_val, y_val)
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

    features = '../features/AAC.csv'
    target = pd.read_csv('../features/my.csv',header=None)
    target = target.values.ravel()
    a = []
    for i in range(2):
        acc, _, _, _, _, _, _, _ = tudou(features, target, )
        a.append(acc)

    print('平均值测试集准确率为:', mean(a))
