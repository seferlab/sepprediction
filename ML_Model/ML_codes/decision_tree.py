"""
    通过树形结构来实现分类的一种算法，关键在于如何选择最优属性
    通常用三种方式：信息增益（ID3）、增益率（C4.5）、基尼系数（CART）
"""
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler  #数据归一化
def tudou(CSV,target,xianshi=True):
    test_prob = []
    test_true = []
    # 加载数据
    feature = pd.read_csv(CSV,header=None)
    print(feature.shape)


    # 字典特征抽取
    # dv = DictVectorizer()
    # feature = dv.fit_transform(feature.to_dict(orient="records"))
    # feature = feature.toarray()
    # print(feature.shape)
    # print(feature)
    std = StandardScaler()
    feature = std.fit_transform(feature)

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.2)



    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    # print("训练集：", x_train.shape, y_train.shape)
    # print("验证集：", x_val.shape, y_val.shape)
    # print("测试集：", x_test.shape, y_test.shape)

    # 建立模型
    tree = DecisionTreeClassifier()
    param = {'max_depth':[2,3,4,5,6]}
    tree = GridSearchCV(tree, param_grid=param,cv=5)
    # 训练
    tree.fit(x_train, y_train)

    pred_res = tree.predict_proba(x_test)[:, 1]
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
    name = 'D:/xuexi/ORF/试验/First_kind/RAT/decision_tree_RAT_SEPs_sORFs_Finally_features'

    np.save(name + 'test_prob.npy', test_prob)
    np.save(name + 'test_true.npy', test_true)

    if xianshi == True:
        score_train = tree.score(x_train, y_train)
        score_val = tree.score(x_val, y_val)
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
    for i in range(20):
        acc, _, _, _, _, _, _, _ = tudou(features, target, )
        a.append(acc)

    print('平均值测试集准确率为:', mean(a))