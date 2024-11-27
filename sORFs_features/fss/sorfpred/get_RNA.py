import numpy as np
import csv
import warnings
warnings.filterwarnings("ignore")

from Bio import SeqIO

import numpy as np

import re, os, sys
from collections import Counter
import Bio.SeqIO as Seq

k = 3

def create_mers(k):
    kmers = [[],[]]
    ATCG = 'ATCG'
    m = 0
    if k == 1:
        for i in range(0,4):
            kmers[0].append(ATCG[i])
            kmers[1].append(0)
    elif k == 2:
        for i in range(0,4):
            for j in range(0,4):
                kmers[0].append(ATCG[i]+ATCG[j])
                kmers[1].append(0)
                m += 1
    elif k == 3:
        for i in range(0,4):
            for j in range(0,4):
                for k in range(0,4):
                    kmers[0].append(ATCG[i] + ATCG[j] + ATCG[k])
                    kmers[1].append(0)
                    m += 1
    elif k == 4:
        for i in range(0,4):
            for j in range(0,4):
                for k in range(0,4):
                    for l in range(0,4):
                        kmers[0].append(ATCG[i] + ATCG[j] + ATCG[k] + ATCG[l])
                        kmers[1].append(0)
                        m += 1
    return kmers

def get_kmers_feature(s,k):
    mers = create_mers(k)
    s_len = len(s)
    if k == 1:
        for i in range(0,s_len):
            s_mer = s[i]
            m = 0
            for j in mers[0]:
                if s_mer != j:
                    m += 1
                else:
                    mers[1][m] += 1/(16*s_len)
    if k == 2:
        for i in range(0,s_len - 1):
            s_mer = s[i:i+2]
            m = 0
            for j in mers[0]:
                if s_mer != j:
                    m += 1
                else:
                    mers[1][m] += 1/(4*(s_len-1))
    if k == 3:
        for i in range(0,s_len - 2):
            s_mer = s[i:i+3]
            m = 0
            for j in mers[0]:
                if s_mer != j:
                    m += 1
                else:
                    mers[1][m] += 1/(s_len-2)
    return mers[1]

def get_GC_feature(s):
    s_len = len(s)
    G_num = 0
    C_num = 0
    GC_content = 0
    GC_ratio = 0
    for i in s:
        if i == 'C':
            C_num += 1
        elif i == 'G':
            G_num += 1
    if C_num == 0:
        GC_ratio = 0
    else:
        GC_content = (G_num + C_num)/s_len
        GC_ratio = G_num/C_num
    return GC_content,GC_ratio

def create_SSMname(k):
    ATCG = 'ATCG'
    SSM_feature = [[],[]]
    if k == 1:
        for i in range(0,4):
            for j in range(0,4):
                SSM_feature[0].append(ATCG[i] + '*' + ATCG[j])
                SSM_feature[1].append(0)
    elif k == 2:
        for i in range(0,4):
            for j in range(0,4):
                SSM_feature[0].append(ATCG[i] + '**' + ATCG[j])
                SSM_feature[1].append(0)
    elif k == 3:
        for i in range(0,4):
            for j in range(0,4):
                SSM_feature[0].append(ATCG[i] + '***' + ATCG[j])
                SSM_feature[1].append(0)
    return SSM_feature

def get_SSM_feature(s,k):
    SSM_feature = create_SSMname(k)
    s_len = len(s)
    if k == 1:
        for i in range(0,s_len-2):
            s_ssm1 = s[i]
            s_ssm2 = s[i+2]
            s_ssm = s_ssm1 + '*' + s_ssm2
            m = 0
            for j in SSM_feature[0]:
                if s_ssm != j:
                    m += 1
                else:
                    SSM_feature[1][m] += 1/(s_len-2)
    if k == 2:
        for i in range(0,s_len - 3):
            s_ssm1 = s[i]
            s_ssm2 = s[i + 3]
            s_ssm = s_ssm1 + '**' + s_ssm2
            m = 0
            for j in SSM_feature[0]:
                if s_ssm != j:
                    m += 1
                else:
                    SSM_feature[1][m] += 1/(s_len-3)
    if k == 3:
        for i in range(0,s_len - 4):
            s_ssm1 = s[i]
            s_ssm2 = s[i + 4]
            s_ssm = s_ssm1 + '***' + s_ssm2
            m = 0
            for j in SSM_feature[0]:
                if s_ssm != j:
                    m += 1
                else:
                    SSM_feature[1][m] += 1/(s_len-4)
    return SSM_feature[1]

def SNR_feature(s):
    Ua = []
    Ut = []
    Uc = []
    Ug = []
    for i in s:
        if i == 'A':
            Ua.append(1)
            Ut.append(0)
            Uc.append(0)
            Ug.append(0)
        elif i == 'T':
            Ua.append(0)
            Ut.append(1)
            Uc.append(0)
            Ug.append(0)
        elif i == 'C':
            Ua.append(0)
            Ut.append(0)
            Uc.append(1)
            Ug.append(0)
        else:
            Ua.append(0)
            Ut.append(0)
            Uc.append(0)
            Ug.append(1)
    return Ua, Ut, Uc, Ug

def QR(s):
    XYZa = [0]*3
    XYZt = [0]*3
    XYZc = [0]*3
    XYZg = [0]*3
    for i in range(len(s)):
        if s[i] == 'A':
            if i % 3 == 0:
                XYZa[0] += 1
            elif i % 3 == 1:
                XYZa[1] += 1
            else:
                XYZa[2] += 1
        elif s[i] == 'T':
            if i % 3 == 0:
                XYZt[0] += 1
            elif i % 3 == 1:
                XYZt[1] += 1
            else:
                XYZt[2] += 1
        elif s[i] == 'C':
            if i % 3 == 0:
                XYZc[0] += 1
            elif i % 3 == 1:
                XYZc[1] += 1
            else:
                XYZc[2] += 1
        else:
            if i % 3 == 0:
                XYZg[0] += 1
            elif i % 3 == 1:
                XYZg[1] += 1
            else:
                XYZg[2] += 1

    P_a = XYZa[0] * XYZa[0] + XYZa[1] * XYZa[1] + XYZa[2] * XYZa[2] - XYZa[0] * XYZa[1] - XYZa[0] * XYZa[2] - XYZa[1] * XYZa[2]
    P_t = XYZt[0] * XYZt[0] + XYZt[1] * XYZt[1] + XYZt[2] * XYZt[2] - XYZt[0] * XYZt[1] - XYZt[0] * XYZt[2] - XYZt[1] * XYZt[2]
    P_c = XYZc[0] * XYZc[0] + XYZc[1] * XYZc[1] + XYZc[2] * XYZc[2] - XYZc[0] * XYZc[1] - XYZc[0] * XYZc[2] - XYZc[1] * XYZc[2]
    P_g = XYZg[0] * XYZg[0] + XYZg[1] * XYZg[1] + XYZg[2] * XYZg[2] - XYZg[0] * XYZg[1] - XYZg[0] * XYZg[2] - XYZg[1] * XYZg[2]
    R = (P_a + P_t + P_c + P_g)/len(s)
    return R


def get_RNA_feature(s):
    s = s.rstrip('\n')
    #kmer0-83
    feature_1mer = get_kmers_feature(s, 1)
    feature_2mers = get_kmers_feature(s, 2)
    feature_3mers = get_kmers_feature(s, 3)
    feature1 = feature_1mer + feature_2mers + feature_3mers
    #CG含量84,85
    GC_content, GC_ratio = get_GC_feature(s)
    feature2 = [GC_content, GC_ratio]
    #信噪比86
    feature_SNR = QR(s)
    feature3 = [feature_SNR]
    #短序列模体87-134  总共是48维
    feature_one = get_SSM_feature(s, 1)  # 4*4=16 维
    feature_two = get_SSM_feature(s, 2)   # 4*4=16 维
    feature_three = get_SSM_feature(s, 3)  # 4*4=16 维
    feature4 = feature_one + feature_two + feature_three

    sORF_length = len(s)
    feature5 = [sORF_length]


    # feature = feature1 + feature2 + feature3 + feature4 + feature5

    feature = feature4 + feature5
    return feature

# # 读取fasta文件并提取特征
# def get_features(fasta_file, label):
#     data = []
#     for record in SeqIO.parse(fasta_file, "fasta"):
#         seq = str(record.seq)
#         # length = get_length(seq)
#         fs = get_RNA_feature(seq)
#         # print(fick)
#         # print(np.array(fick).shape)
#
#         # row = [length, fick]
#         data.append(fs)
#     return data
#
#
# def get_data10(pos, neg):
#     from sklearn.model_selection import train_test_split
#
#     train_p_data =get_features(path_p_train,1)
#     # print(np.array(train_p_data).shape)
#     train_n_data = get_features(path_n_train,0)
#     # print(np.array(train_n_data).shape)
#     # exit()
#
#     train_p_data, test_p_data = train_test_split(train_p_data, test_size=0.4, random_state=3)
#     val_p_data, test_p_data = train_test_split(test_p_data, test_size=0.5, random_state=3)
#     train_n_data, test_n_data = train_test_split(train_n_data, test_size=0.4, random_state=3)
#     val_n_data, test_n_data = train_test_split(test_n_data, test_size=0.5, random_state=3)
#     train_p_data = np.array(train_p_data)
#     train_n_data = np.array(train_n_data)
#     val_p_data = np.array(val_p_data)
#     val_n_data = np.array(val_n_data)
#     train_data = np.concatenate([train_p_data, train_n_data], axis=0)
#     val_data = np.concatenate([val_p_data, val_n_data], axis=0)
#     test_data = np.concatenate([test_p_data, test_n_data], axis=0)
#     train_label = [1] * len(train_p_data) + [0] * len(train_n_data)
#     val_label = [1] * len(val_p_data) + [0] * len(val_n_data)
#     test_label = [1] * len(test_p_data) + [0] * len(test_n_data)
#     train_label = np.array(train_label)
#     val_label = np.array(val_label)
#     test_label = np.array(test_label)
#     return train_data, val_data, test_data, train_label, val_label, test_label
# # fastas1 = readFasta.readFasta('train_hum+.txt')
# # fastas = readFasta.readFasta('train_hum-.txt')
# path_p_train = "rorf_pos.txt"
# path_n_train = "rorf_neg.txt"
# train_data, val_data, test_data, train_label, val_label, test_label = get_data10(path_p_train, path_n_train)
# # print(train_data)
# # print(train_data.shape)
# # exit()
# from catboost import CatBoostClassifier
# clf = CatBoostClassifier(learning_rate=0.01,
#                              iterations=100000000,
#                              depth=8,
#                              loss_function="Logloss",
#                              early_stopping_rounds=200,
#                              # eval_metric="AUC",
#                              # eval_metric="MCC",
#                              eval_metric="Accuracy",
#                              thread_count=50,
#                              # od_wait=500,
#                              # task_type="CPU",
#                              task_type="CPU",
#                              # devices='0:1:2:3'
#                              devices='0'
#                              )
# # from sklearn import svm
# # svc = svm.SVC(probability=True, kernel='rbf')
#
# # from sklearn.linear_model import LogisticRegression as LR
# # l = LR(C=1, penalty='l1', solver='liblinear')
#
#
# clf.fit(X=train_data,
#             y=train_label,
#         eval_set=(val_data,
#                   val_label))
#
# pred_res = clf.predict_proba(test_data)[:,1]
#
# pred_label = [0 if x < 0.5 else 1 for x in pred_res]
#
# from sklearn import metrics
# tn, fp, fn, tp = metrics.confusion_matrix(y_true=test_label, y_pred=pred_label).ravel()
#
# recall = metrics.recall_score(y_pred=pred_label, y_true=test_label)
# precise = metrics.precision_score(y_pred=pred_label, y_true=test_label)
#
# se = tp/(tp+fn)
# sp = tn/(tn+fp)
#
# acc = metrics.accuracy_score(y_pred=pred_label, y_true=test_label)
# f1 = metrics.f1_score(y_pred=pred_label, y_true=test_label)
# mcc = metrics.matthews_corrcoef(y_pred=pred_label, y_true=test_label)
#
# auc = metrics.roc_auc_score(y_true=test_label, y_score=pred_res)
# ap = metrics.average_precision_score(y_score=pred_res, y_true=test_label)
#
#
# print("tn", tn,flush=True)
# print("tp", tp,flush=True)
# print("fp", fp,flush=True)
# print("fn", fn,flush=True)
#
# print("recall",recall,flush=True)
# print("precise",precise,flush=True)
# # print("se",  se,flush=True)
# print("sp",  sp,flush=True)
# print("acc", acc,flush=True)
# print("f1",  f1,flush=True)
# print("mcc", mcc,flush=True)
# print("auc", auc,flush=True)
# print("ap",  ap,flush=True)