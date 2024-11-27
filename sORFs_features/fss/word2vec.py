# from gensim.test.utils import common_texts
# from gensim.models import Word2Vec
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
# train_path1 = "Orf_NEW_P_Rat.fasta"
# train_path2 = "Orf_NEW_N_Rat.fasta"
#
# paths = [train_path1, train_path2]
#
# seqs = []
# for path in paths:
#     seq  = read_data(path)
#     seqs.extend(seq)
#
# seqs = [list(seq) for seq in seqs]
#
# print("begin train")
# model = Word2Vec(sentences=seqs, vector_size=10, window=5, min_count=1, workers=40,epochs=128)
# model.save("NEW_Ro_w2c.model")
#
# print("finish")
#
# exit()

import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from Bio import SeqIO
import numpy as np
from Bio.Seq import Seq


from gensim.models import Word2Vec
w2c = Word2Vec.load("NEW_Ro_w2c.model")
A_w2c = w2c.wv["A"]
T_w2c = w2c.wv["T"]
C_w2c = w2c.wv["C"]
G_w2c = w2c.wv["G"]


def pystx(seq):

    '''WORD2VEC:
    HUM:
recall 0.73
precise 0.7242063492063492
se 0.73
sp 0.722
acc 0.726
f1 0.7270916334661355
mcc 0.452014464694309
auc 0.790019111111111
ap 0.7772411629546377

    MOU:
recall 0.6407692307692308
precise 0.6574585635359116
se 0.6407692307692308
sp 0.6661538461538462
acc 0.6534615384615384
f1 0.6490066225165563
mcc 0.3070220120858102
auc 0.7061349112426035
ap 0.6943116010937243

RAT:
recall 0.7
precise 0.6496519721577726
se 0.7
sp 0.6225
acc 0.66125
f1 0.6738868832731647
mcc 0.3234728925897716
auc 0.72050625
ap 0.7120946087192642

    :param seq:
    :return:
    '''

    std = {
        "A":A_w2c,
        "T":T_w2c,
        "C":C_w2c,
        "G":G_w2c,
    }

    from pyts.image import GramianAngularField
    gaf = GramianAngularField(image_size=10)
    vec = np.array([std[x] for x in seq])
    vec = vec.transpose([1,0])
    mat = gaf.fit_transform(vec)
    mat = mat.flatten().tolist()
    return mat

# def mer3(seq):
#     '''recall 0.7127575277337559
#     precise 0.7977827050997782
#     sp 0.8194774346793349
#     acc 0.7661386138613862
#     f1 0.752877170956267
#     mcc 0.5353040991470839
#     auc 0.8393057683451848
#     ap 0.852381773603433'''
#     mers = []
#     for x in "ATCGN":
#         for y in "ATCGN":
#             for z in "ATCGN":
#                 tmp = x+y+z
#                 mers.append(tmp)
#
#     res = []
#     for mer in mers:
#         c = seq.count(mer)/len(seq)
#         res.append(c)
#     return res
#
# def mer4(seq):
#     '''recall 0.7262282091917591
#     precise 0.8096289752650176
#     sp 0.8293745051464766
#     acc 0.7778217821782178
#     f1 0.7656641604010025
#     mcc 0.5585948660592065
#     auc 0.8473976821719725
#     ap 0.8579595021548059'''
#     mers = []
#     for x in "ATCGN":
#         for y in "ATCGN":
#             for z in "ATCGN":
#                 for k in "ATCGN":
#                     tmp = x+y+z+k
#                     mers.append(tmp)
#
#     res = []
#     for mer in mers:
#         c = seq.count(mer)/len(seq)
#         res.append(c)
#     return res

def proteink1(seq):

    '''
    HUM:
recall 0.71
precise 0.876904075751338
sp 0.9003333333333333
acc 0.8051666666666667
f1 0.784674894087309
mcc 0.6216982742148247
auc 0.8798072222222222
ap 0.9051005108856391
MOU:
recall 0.69
precise 0.8551000953288846
sp 0.8830769230769231
acc 0.7865384615384615
f1 0.7637292464878671
mcc 0.5840669470252315
auc 0.8512165680473374
ap 0.8824969465651042

RAT:
recall 0.6875
precise 0.7834757834757835
sp 0.81
acc 0.74875
f1 0.7323568575233023
mcc 0.5012753485936747
auc 0.82733125
ap 0.8545942207848368


    :param seq:
    :return:
    '''

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
    '''HUM:
recall 0.7356666666666667
precise 0.8775347912524851
sp 0.8973333333333333
acc 0.8165
f1 0.800362647325476
mcc 0.6414378435054332
auc 0.8834405555555556
ap 0.9087332861751861

MOU:
recall 0.703076923076923
precise 0.8638941398865785
sp 0.8892307692307693
acc 0.7961538461538461
f1 0.7752332485156912
mcc 0.602845057250573
auc 0.861478698224852
ap 0.8896375400557323

RAT:
recall 0.65
precise 0.8024691358024691
sp 0.84
acc 0.745
f1 0.7182320441988951
mcc 0.4990914039375817
auc 0.8062125
ap 0.8379466798059066
    '''


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

    '''
HUM:
recall 0.7426666666666667
precise 0.8696330991412958
sp 0.8886666666666667
acc 0.8156666666666667
f1 0.8011506652283352
mcc 0.6381716037212631
auc 0.8850654444444443
ap 0.9104193311546196
MOU:
recall 0.7015384615384616
precise 0.8611898016997167
sp 0.8869230769230769
acc 0.7942307692307692
f1 0.7732089868588385
mcc 0.5988418394993419
auc 0.858491124260355
ap 0.8885273556786694
RAT:
recall 0.6975
precise 0.7948717948717948
sp 0.82
acc 0.75875
f1 0.7430093209054593
mcc 0.5214271214014606
auc 0.8309124999999999
ap 0.8574078344877065

    :param seq:
    :return:
    '''


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


def get_w2feature(seq):
    feature1 = pystx(seq)

    # feature2 = mer3(seq)

    # feature3 = mer4(seq)

    feature2 = proteindc(seq)

    # feature3 = proteink1(seq)

    feature4 = proteink2(seq)
    #
    # feature =  feature2 + feature4

    return feature1


# 读取fasta文件并提取特征
def get_features(fasta_file, label):
    data = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        # length = get_length(seq)
        fs = get_w2feature(seq)
        # print(fick)
        # print(np.array(fick).shape)

        # row = [length, fick]
        data.append(fs)
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

def get_data10(pos, neg):
    from sklearn.model_selection import train_test_split

    train_p_data =get_features(path_p_train,1)
    print(np.array(train_p_data).shape)
    train_n_data = get_features(path_n_train,0)
    # print(np.array(train_n_data).shape)
    # exit()

    # # 合并所有数据的特征
    # all_data = np.concatenate([train_p_data, train_n_data], axis=0)
    #
    # # 保存总特征为CSV文件
    # all_features = pd.DataFrame(all_data)
    # all_features.to_csv("HUM_word_kmer_features.csv", index=False, header=False)
    # print("csv文件已生成")

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
print("se",  se,flush=True)
print("sp",  sp,flush=True)
print("acc", acc,flush=True)
print("f1",  f1,flush=True)
print("mcc", mcc,flush=True)
print("auc", auc,flush=True)
print("ap",  ap,flush=True)