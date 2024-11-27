import os,sys
import CTD
import numpy
import Bio.SeqIO as Seq
import fickett
import FrameKmer
import csv

import warnings
warnings.filterwarnings("ignore")

from Bio import SeqIO
import numpy as np
import pandas as pd


def coding_nocoding_potential(input_file):
	coding={}
	noncoding={}
	for line in open(input_file).readlines():
		fields = line.split()
		if fields[0] == 'hexamer':continue
		coding[fields[0]] = float(fields[1])
		noncoding[fields[0]] = float(fields[2])
	return coding,noncoding


def get_fhCTD_feature(seq):
	hex_file='NEW_HUM_hexamer.tsv'  #hexamer 的频率表要根据数据集的不同而更改
	coding,noncoding = coding_nocoding_potential(hex_file)
	fickett_fe = fickett.fickett_value(seq)
	hexamer = FrameKmer.kmer_ratio(seq,6,3,coding,noncoding)
	A,T,G,C,AT,AG,AC,TG,TC,GC,A0,A1,A2,A3,A4,T0,T1,T2,T3,T4,G0,G1,G2,G3,G4,C0,C1,C2,C3,C4 = CTD.CTD(seq)

	# feature=[fickett_fe,float(A),float(T),float(G),float(C),float(AT),float(AG),float(AC),float(TG),float(TC),float(GC),
	# 		 float(A0),float(A1),float(A2),float(A3),float(A4),float(T0),float(T1),float(T2),float(T3),float(T4),
	# 		 float(G0),float(G1),float(G2),float(G3),float(G4),float(C0),float(C1),float(C2),float(C3),float(C4)]
	feature = [hexamer]
	return feature
'''hexamer:
HUM:
recall 0.6663333333333333
precise 0.7351967635159985
se 0.6663333333333333
sp 0.76
acc 0.7131666666666666
f1 0.6990732645567407
mcc 0.42821593602013985
auc 0.7804260555555556
ap 0.7881681875880875

MOU:
recall 0.6976923076923077
precise 0.6902587519025876
se 0.6976923076923077
sp 0.686923076923077
acc 0.6923076923076923
f1 0.693955623565417
mcc 0.3846376896961958
auc 0.754137278106509
ap 0.7435560467828736

RAT:
recall 0.7575
precise 0.7046511627906977
se 0.7575
sp 0.6825
acc 0.72
f1 0.7301204819277108
mcc 0.44124274529623214
auc 0.78791875
ap 0.7485539653763671

'''

# 读取fasta文件并提取特征
def get_features(fasta_file, label):
	data = []
	for record in SeqIO.parse(fasta_file, "fasta"):
		seq = str(record.seq)
		# length = get_length(seq)
		fs = get_fhCTD_feature(seq)
		# print(fick)
		# print(np.array(fick).shape)

		# row = [length, fick]
		data.append(fs)
	return data


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
	# all_features.to_csv("TEST_HEX11.csv", index=False, header=False)
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
path_p_train = "Orf_NEW_P_Hum.fasta"
path_n_train = "Orf_NEW_N_Hum.fasta"
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