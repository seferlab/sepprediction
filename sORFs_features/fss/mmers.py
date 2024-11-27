import warnings

warnings.filterwarnings("ignore")
import sys

sys.path.append("/")
import orfipy_core
import math
import random
import Bio
import re
from Bio import SeqIO
from Bio.Seq import Seq
import FrameKmer
from Bio.SeqUtils import ProtParam
import numpy as np
import sys
sys.path.append("./repDNA/")
from repDNA.psenac import PseKNC, PseDNC
top = 3

# 1 float
def get_length(seq):
    return np.log(len(seq) + 1)


def mers4(seq):
    mers = []
    for x in "ATCG":
        for y in "ATCG":
            tmps = []
            for z in "ATCG":
                for k in "ATCG":
                    for l in "ATCG":
                        for m in "ATCG":
                            tmp = x+z+k+l+m+y
                            tmps.append(tmp)
            sum = 0
            for mer in tmps:
                sum = sum +seq.count(mer)
            sum = sum/len(seq)
            mers.append(sum)
    return mers

# 读取fasta文件并提取特征
def get_features(fasta_file, label):
    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        # length = get_length(seq)
        zc = mers4(seq)
        # print(mmers)
        # print(np.array(mmers).shape)

        # row = [length, mmers]
        data.append(zc)
    return data


# # 读取正样本fasta文件
# path_p = 'p.fa'
# p_data = get_features(path_p, 1)
#
# # 读取负样本fasta文件
# path_n = 'n.fa'
# n_data = get_features(path_n, 0)
#
# print(np.array(p_data).shape)
#
# print(np.array(n_data).shape)
#
# exit()

def get_data10(pos, neg):
    from sklearn.model_selection import train_test_split

    train_p_data =get_features(path_p_train,1)
    # print(np.array(train_p_data).shape)
    train_n_data = get_features(path_n_train,0)
    # print(np.array(train_n_data).shape)
    # exit()

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
path_p_train = "/home/nq/orf/horf_pos.txt"
path_n_train = "/home/nq/orf/horf_neg.txt"
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
# svc = svm.SVC(probability=True, kernel='rbf')

# from sklearn.linear_model import LogisticRegression as LR
# l = LR(C=1, penalty='l1', solver='liblinear')


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
# print("se",  se,flush=True)
print("sp",  sp,flush=True)
print("acc", acc,flush=True)
print("f1",  f1,flush=True)
print("mcc", mcc,flush=True)
print("auc", auc,flush=True)
print("ap",  ap,flush=True)