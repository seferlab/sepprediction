import argparse
import sys
import time
import pandas as pd
import numpy as np

from features_code import *
import sys
import os
import re
from features_code import CKSAAGP
from features_code import GTPC
from features_code import NMBroto
from features_code import KSCTriad
from features_code import APAAC
from features_code import PAAC
from features_code import Moran
from features_code import SOCNumber
from features_code import QSOrder
from features_code import Geary
from features_code import BINARY
from features_code import EGAAC
from features_code import ZSCALE
from features_code import BLOSUM62
from features_code import EAAC
from features_code import AAINDEX
from features_code import TPC
import readFasta

# 生成标签
def target_blu(changdu,dizhi):
    a = []
    for i in range(0, 12642):
        a.append(0)
    for i in range(0, 12643):
        a.append(1)
    target = np.array(a)
    target=pd.DataFrame(target)
    target.to_csv(dizhi, index=False, header=False)

# target_blu(573,'./features/human_label.csv')



def todataframe(encoding):
    index_list = []
    new_list = []
    for i in encoding[1:]:
        new_list.append(i[1:])
        index_list.append(i[0])
    return new_list, index_list




# 生成等长序列
def equal_length(fasta,length):

    index = []
    encoding = []

    for name, sequence in fasta:
        index.append(name)
        if len(sequence) > length:
            encoding.append(sequence[:length])

        elif len(sequence) < length:
            seq = sequence
            for k in range(length - len(sequence)):
                seq = seq + '-'
            encoding.append(seq)
        else:
            encoding.append(sequence)

    he = []
    for i in range(len(index)):
        he.append([index[i], encoding[i]])
    return he
'''
合并多个特征时候用
'''
def XGB_inputGenterator_duo(features,fasta,length,nlag, **kw,):
    equal_length_fea = ['BINARY', 'EGAAC', 'ZSCALE', 'BLOSUM62', 'EAAC', 'AAINDEX']
    need_nlag = ['QSOrder',]
    need_lambdaValue=['PAAC','APAAC']
    feature_dict = {}
    feature_list = []
    for i in features:
        print('当前特征',i)
        # 生成等长序列
        if i in equal_length_fea:
            fasta=equal_length(fasta,length)

        if i in need_nlag:
            cmd = i + '.' + i + '(fasta,nlag='+str(nlag)+', **kw)'
        elif i in need_lambdaValue:
            cmd = i + '.' + i + '(fasta,lambdaValue='+str(nlag)+', **kw)'
        else:
            cmd = i + '.' + i + '(fasta, **kw)'
        encoding = eval(cmd)
        content, index = todataframe(encoding)

        feature_dict[i] = pd.DataFrame(content, columns=encoding[0][1:])
        feature_list.append(pd.DataFrame(content, columns=encoding[0][1:]))


    df = pd.concat(feature_list, axis=1)
    return df

'''
合并单个特征
'''

def XGB_inputGenterator(features,fasta,length,nlag, **kw,):
    equal_length_fea=['BINARY','EGAAC','ZSCALE','BLOSUM62','EAAC','AAINDEX']
    need_nlag = ['QSOrder', 'Geary','SOCNumber','Moran','NMBroto']
    need_lambdaValue = ['PAAC', 'APAAC']
    # 只存储数据
    feature_list = []
    # 生成等长序列
    if features in equal_length_fea:
        fasta=equal_length(fasta,length)

    if i in need_nlag:
        cmd = i + '.' + i + '(fasta,nlag=' + str(nlag) + ', **kw)'
    elif i in need_lambdaValue:
        cmd = i + '.' + i + '(fasta,lambdaValue=' + str(nlag) + ', **kw)'
    else:
        cmd = i + '.' + i + '(fasta, **kw)'

    encoding = eval(cmd)

    # 讲数据名称和 数据分开, 比如 index: [postive14.... ]   content: [0.211,0........]
    content, index = todataframe(encoding)
    feature_list.append(pd.DataFrame(content, columns=encoding[0][1:]))
    df = pd.concat(feature_list, axis=1)
    return df

def generator(feature,pos,nec,save,length,nlag,PSSM_file=None, fasta_file=None,features_duo_or_shao=False):
    if not PSSM_file and not fasta_file:
        PSSM_file = "./features/pse_pssm_test.csv"
        fasta_N = readFasta.readfasta(nec)
        fasta_P = readFasta.readfasta(pos)
        fasta = fasta_N + fasta_P

    kw = {'order': 'ACDEFGHIKLMNPQRSTVWY', 'PSSM': PSSM_file}

    if features_duo_or_shao==False:
        df_XGB = XGB_inputGenterator(feature,fasta,length ,nlag,**kw,)
    else:
        df_XGB = XGB_inputGenterator_duo(feature, fasta,length,nlag, **kw,)
    print(df_XGB.shape)

    df_XGB.to_csv(save, index=False, header=False)



if __name__ == "__main__":
    # 完整
    # features = ["AAC", "CKSAAP", "CTDC", "CTDT", "CTDD", "CTriad", "DDE", "GAAC", "GDPC", "DPC","Geary","QSOrder","SOCNumber",
    # "Moran","PAAC","APAAC","KSCTriad","NMBroto","GTPC","CKSAAGP","BINARY","EGAAC","ZSCALE","BLOSUM62","EAAC","AAINDEX",'TPC'
    #             ]

    # features = ["AAC", "CKSAAP", "CTDC", "CTDT", "CTDD", "CTriad", "DDE", "GAAC", "GDPC", "DPC", "Geary", "QSOrder",
    #             "SOCNumber",
    #             "Moran", "PAAC", "APAAC", "KSCTriad", "NMBroto", "GTPC", "CKSAAGP", "BINARY", "EGAAC", "ZSCALE",
    #             "BLOSUM62", "EAAC",]
    # features = ["AAC", "APAAC", "QSOrder" ,"PAAC", ]
    # features = ["AAC", "CTDC", "QSOrder", "PAAC", "APAAC", 'EAAC','CKSAAP','DDE']
    # features = ["AAC", "CTDC", "QSOrder", "PAAC", "APAAC", 'EAAC', 'CKSAAP', 'DDE','CTDT','CKSAAGP']
    # features = ["AAC", "CTDC", "QSOrder", "PAAC", "APAAC", 'EAAC', 'CKSAAP', 'DDE', 'CTDT', 'CKSAAGP','DPC','GDPC','CTDD']
    # features = ["AAC", "APAAC", "QSOrder", "PAAC", "EAAC", "DPC", "CKSAAP", "DDE", "BLOSUM62", "CTDC", "CTDD", "BINARY", "ZSCALE"]
    # features = ["AAC", "APAAC", "QSOrder", "PAAC", "DPC", "CKSAAP"]
    # features = [ "EAAC", "CKSAAP", "DDE", "BLOSUM62", "CTDD"]
    features = ["AAC", "QSOrder", "EAAC", "BLOSUM62", ]
   #
    # 数据集地址
    pos = './NEW_HUM/Pep_NEW_P_Hum.txt'
    nec = './NEW_HUM/Pep_NEW_N_Hum.txt'
    # 是否选择合成特征
    features_duo_or_shao=False
    # 特征需要等长序列,长度的设置
    length=30
    nlag=3

    if features_duo_or_shao==True:
        for i in features:
            print('\n---------------------------------------------\n')
            start = time.time()

            save='./NEW_HUM/'+i+'.csv'
            generator(i,pos,nec,save,length,nlag)

            end = time.time()
            print('{}  提取成功'.format(i))
            print('运行时间花费了:%s'%(end-start))
    else:
        # 合并多个特征时候用
        name = '+'.join(features)

        generator(features, pos, nec, './NEW_HUM/' + name + '.csv', length, nlag, features_duo_or_shao=True)







