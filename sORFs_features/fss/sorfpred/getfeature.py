import re, os, sys
from collections import Counter
import Bio.SeqIO as Seq
import fickett
import argparse as agp

from hexamer import get_fhCTD_feature
from fss.fss import get_fs
from fss.word2vec import get_w2feature

import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from Bio import SeqIO

import numpy as np


def get_feature(s):

    feature3 = get_fhCTD_feature(s)

    feature2 = get_fs(s)

    feature1 = get_w2feature(s)

    feature =feature1 + feature2 + feature3

    print(np.array(feature).shape)
    '2679维'

    return feature


# 读取fasta文件并提取特征
def get_features(fasta_file, label):
    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        # length = get_length(seq)
        fs = get_feature(seq)

        # row = [length, fick]
        data.append(fs)
    return data


def get_data10(pos, neg):
    from sklearn.model_selection import train_test_split

    train_p_data =get_features(path_p_train,1)
    # print(np.array(train_p_data).shape)
    train_n_data = get_features(path_n_train,0)
    # print(np.array(train_n_data).shape)
    # exit()

    # 合并所有数据的特征
    all_data = np.concatenate([train_p_data, train_n_data], axis=0)

    # 保存总特征为CSV文件
    all_features = pd.DataFrame(all_data)
    all_features.to_csv("NEW_ORF_RAT_shap_test.csv", index=False, header=False)
    print("csv文件已生成")

    train_p_data, test_p_data = train_test_split(train_p_data, test_size=0.4, random_state=3)
    val_p_data, test_p_data = train_test_split(test_p_data, test_size=0.5, random_state=3)
    train_n_data, test_n_data = train_test_split(train_n_data, test_size=0.4, random_state=3)
    val_n_data, test_n_data = train_test_split(test_n_data, test_size=0.5, random_state=3)
    train_p_data = np.array(train_p_data)
    train_n_data = np.array(train_n_data)
    val_p_data = np.array(val_p_data)
    val_n_data = np.array(val_n_data)
    train_data = np.concatenate([train_p_data, train_n_data], axis=0)
    val_data = np.concatenate([val_p_data, val_n_data], axis=0)
    test_data = np.concatenate([test_p_data, test_n_data], axis=0)
    train_label = [1] * len(train_p_data) + [0] * len(train_n_data)
    val_label = [1] * len(val_p_data) + [0] * len(val_n_data)
    test_label = [1] * len(test_p_data) + [0] * len(test_n_data)
    train_label = np.array(train_label)
    val_label = np.array(val_label)
    test_label = np.array(test_label)
    return train_data, val_data, test_data, train_label, val_label, test_label
# fastas1 = readFasta.readFasta('train_hum+.txt')
# fastas = readFasta.readFasta('train_hum-.txt')
path_p_train = "Orf_NEW_P_Rat.fasta"
path_n_train = "Orf_NEW_N_Rat.fasta"
train_data, val_data, test_data, train_label, val_label, test_label = get_data10(path_p_train, path_n_train)


# print(train_data)
# print(train_data.shape)
# exit()
from catboost import CatBoostClassifier
clf = CatBoostClassifier(learning_rate=0.01,
                             iterations=100000000,
                             depth=8,
                             loss_function="Logloss",
                             early_stopping_rounds=200,
                             # eval_metric="AUC",
                             # eval_metric="MCC",
                             eval_metric="Accuracy",
                             thread_count=50,
                             # od_wait=500,
                             # task_type="CPU",
                             task_type="CPU",
                             # devices='0:1:2:3'
                             devices='0'
                             )
# from sklearn import svm
# svc = svm.SVC(probability=True)


# from sklearn.linear_model import LogisticRegression as LR
# l = LR(C=1, penalty='l1', solver='liblinear')
#
# l.fit(X=train_data,
#             y=train_label)

clf.fit(X=train_data,
            y=train_label,
        eval_set=(val_data,
                  val_label))

pred_res = clf.predict_proba(test_data)[:,1]

pred_label = [0 if x < 0.5 else 1 for x in pred_res]

from sklearn import metrics
tn, fp, fn, tp = metrics.confusion_matrix(y_true=test_label, y_pred=pred_label).ravel()

recall = metrics.recall_score(y_pred=pred_label, y_true=test_label)
precise = metrics.precision_score(y_pred=pred_label, y_true=test_label)

se = tp/(tp+fn)
sp = tn/(tn+fp)

acc = metrics.accuracy_score(y_pred=pred_label, y_true=test_label)
f1 = metrics.f1_score(y_pred=pred_label, y_true=test_label)
mcc = metrics.matthews_corrcoef(y_pred=pred_label, y_true=test_label)

auc = metrics.roc_auc_score(y_true=test_label, y_score=pred_res)
ap = metrics.average_precision_score(y_score=pred_res, y_true=test_label)


print("tn", tn,flush=True)
print("tp", tp,flush=True)
print("fp", fp,flush=True)
print("fn", fn,flush=True)

print("recall",recall,flush=True)
print("precise",precise,flush=True)
print("se",  se,flush=True)
print("sp",  sp,flush=True)
print("acc", acc,flush=True)
print("f1",  f1,flush=True)
print("mcc", mcc,flush=True)
print("auc", auc,flush=True)
print("ap",  ap,flush=True)