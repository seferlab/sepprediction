# from gensim.test.utils import common_texts
# from gensim.models.doc2vec import Doc2Vec
# from Bio import SeqIO
# import numpy as np
#
# def read_data(path):
#     seqs = []
#     rx  = SeqIO.parse(path,format="fasta")
#     for x in list(rx):
#         seq = str(x.seq)
#         seqs.append(seq)
#     return seqs
#
# print("load train")
# train_path1 = "horf_pos12616.txt"
# train_path2 = "horf_neg12616.txt"
#
# paths = [train_path1, train_path2]
#
# seqs = []
# for path in paths:
#     seq = read_data(path)
#     seqs.extend(seq)
#
# seqs = [list(seq) for seq in seqs]
#
# print("begin train")
# model = Doc2Vec( vector_size=100, window=5, min_count=1, workers=40, epochs=128)
# model.save("d2c.model")
#
# print("finish")

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import warnings
warnings.filterwarnings("ignore")


from Bio import SeqIO
import numpy as np
from Bio.Seq import Seq


d2c = Doc2Vec.load("d2v2.model")
A_d2c = d2c.dv["A"]
T_d2c = d2c.dv["T"]
C_d2c = d2c.dv["C"]
G_d2c = d2c.dv["G"]


def pystx(seq):
    std = {
        "A":A_d2c,
        "T":T_d2c,
        "C":C_d2c,
        "G":G_d2c
    }

    from pyts.image import GramianAngularField
    gaf = GramianAngularField(image_size=10)
    vec = np.array([std[x] for x in seq])
    vec = vec.transpose([1,0])
    mat = gaf.fit_transform(vec)
    mat = mat.flatten().tolist()
    return mat

def proteink1(seq):
    std = "ACDEFGHIKLMNPQRSTVWY"
    messenger_rna = Seq(seq)
    seq = messenger_rna.translate(to_stop=True)
    seq = str(seq)

    mers = []
    for x in std:
        for y in std:
            tmps = []
            for z in std:
                tmp = x + z + y
                tmps.append(tmp)
            sum = 0
            for mer in tmps:
                sum = sum + seq.count(mer)
            sum = sum / len(seq)
            mers.append(sum)
    return mers


def proteink2(seq):
    std = "ACDEFGHIKLMNPQRSTVWY"
    messenger_rna = Seq(seq)
    seq = messenger_rna.translate(to_stop=True)
    seq = str(seq)

    mers = []
    for x in std:
        for y in std:
            tmps = []
            for z in std:
                for p in std:
                    tmp = x + z + p + y
                    tmps.append(tmp)
            sum = 0
            for mer in tmps:
                sum = sum + seq.count(mer)
            sum = sum / len(seq)
            mers.append(sum)
    return mers

def proteindc(seq):
    std = "ACDEFGHIKLMNPQRSTVWY*"
    messenger_rna = Seq(seq)
    seq = messenger_rna.translate()
    seq = str(seq)
    mers = []
    for x in std:
        for y in std:
            mer = x+y
            sum = seq.count(mer)
            sum = sum / len(seq)
            mers.append(sum)
    return mers


def get_d2feature(seq):
    feature1 = pystx(seq)

    feature2 = proteindc(seq)

    feature3 = proteink1(seq)

    feature4 = proteink2(seq)

    feature = feature1 + feature2 + feature3 + feature4

    return feature1


# 读取fasta文件并提取特征
def get_features(fasta_file, label):
    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        # length = get_length(seq)
        fs = get_d2feature(seq)
        # print(fick)
        # print(np.array(fick).shape)

        # row = [length, fick]
        data.append(fs)
    return data


# 读取正样本fasta文件
path_p = 'p.fa'
p_data = get_features(path_p, 1)

# 读取负样本fasta文件
path_n = 'n.fa'
n_data = get_features(path_n, 0)

print(np.array(p_data).shape)

print(np.array(n_data).shape)

exit()

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
path_p_train = "horf_pos12616.txt"
path_n_train = "horf_neg12616.txt"
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