import asyncore

from codes import KNN
from codes import SVM
from codes import random_forest
from codes import logistic_regression
from sklearn.ensemble import ExtraTreesClassifier
from codes import decision_tree
from codes import Adaboost
from codes import XGBoost
from codes import LightGBM
from codes import GDBT
from codes import Adaboost
from codes import Catboost
from statistics import mean
from statistics import median
import numpy as np
import pandas as pd
import time


# 完整的
# features = ["AAC", "CKSAAP", "CTDC", "CTDT", "CTDD", "CTriad", "DDE", "GAAC", "GDPC", "DPC","Geary","QSOrder","SOCNumber",
#     "Moran","PAAC","APAAC","KSCTriad","NMBroto","GTPC","CKSAAGP","BINARY","EGAAC","ZSCALE","BLOSUM62","EAAC","AAINDEX",'TPC'
#                 ]
# code=['KNN','SVM','decision_tree','logistic_regression','random_forest','Adaboost','XGBoost','LightGBM','GDBT','Catboost']




def me(paiming_ceshi):
    pingjun=[]
    for i in range(8):
        tong = []
        for j in range(len(paiming_ceshi)):
            tong.append(paiming_ceshi[j][i])
        if i==0:
            me_dian = median(tong)
            max_acc=max(tong)
            min_acc = min(tong)

        tong = mean(tong)


        pingjun.append(tong)

    return round(pingjun[0],3),round(pingjun[1],3),round(pingjun[2],3),round(pingjun[3],3),round(pingjun[4],3),round(pingjun[5],3),round(pingjun[6],3),round(pingjun[7],3),round(me_dian,3),round(max_acc,3),round(min_acc,3)

if __name__ =="__main__":



    # features = ["AAC", "CKSAAP", "CTDC", "CTDT", "CTDD", "CTriad", "DDE", "GAAC", "GDPC", "DPC", "Geary", "QSOrder",
    #             "SOCNumber",
    #             "Moran", "PAAC", "APAAC", "KSCTriad", "NMBroto", "GTPC", "CKSAAGP", "BINARY", "EGAAC", "ZSCALE",
    #             "BLOSUM62", "EAAC"
    #             ]
    # features = ["AAC+APAAC+QSOrder+PAAC"]
    # features = ["AAC", "CKSAAP", "CTDC", "CTDT", "CTDD", "CTriad", "DDE", "GAAC", "GDPC", "DPC", "Geary", "QSOrder",
    #             "SOCNumber",
    #             "Moran", "PAAC", "APAAC", "KSCTriad", "NMBroto", "GTPC", "CKSAAGP", "BINARY", "EGAAC", "ZSCALE",
    #             "BLOSUM62", "EAAC", ]

    features = ["RAT_SEPs_sORFs_Fused Finally features"]

    # code = ['SVM', ' XGBoost', 'random_forest', 'LightGBM', 'GDBT', 'Catboost','logistic_regression','decision_tree']
    code = ['Catboost']
    # 读取 label




    columns = ['模型', '特征', 'ACC','Recall','Precise','AUC','MCC','F1','Se','Sp','消耗时间','ACC中位数','ACC最大值','ACC最小值']

    df = pd.DataFrame(data=[['-','-','-','-','-','-','-','-','-','-','-','-','-','-']], columns=columns)

    #  {'AAC': 0, 'CKSAAP': 0, 'CTDC': 0, 'CTDT': 0, 'CTDD':
    select_d= {}
    for i in features:
        select_d[i]=0


    # 读取标签
    target = pd.read_csv('./features/rat_label.csv', header=None)
    target = target.values.ravel()


    # 循环几次取平均
    xunhuan=1

    index=1

    for i in code:

        paiming = []
        for j in features:

            paiming_ceshi = []
            time_all = 0

            for s in range(xunhuan):
                start = time.time()
                print('\n---------------------\n')
                print('\n', "模型:{}  特征:{}  当前循环{}次,一共循环{}次".format(i,j,s+1,xunhuan,))
                # 模型训练
                cmd = i + '.tudou' + "(\'./huatu/" + j + ".csv\',target,xianshi=False)"

                # 模型训练
                [ACC,Recall,Precise,AUC,MCC,F1,Se,Sp]= eval(cmd)


                paiming_ceshi.append([ACC,Recall,Precise,AUC,MCC,F1,Se,Sp])

                #计算消耗时间
                end = time.time()
                time_all = time_all + end - start

            # 选出平均值保存
            [ACC,Recall,Precise,AUC,MCC,F1,Se,Sp,me_dian,max_value,min_value]=me(paiming_ceshi)
            paiming.append([i, j,ACC,Recall,Precise,AUC,MCC,F1,Se,Sp,round(time_all/xunhuan,2),me_dian,max_value,min_value])
        # 以准确率为基准排序
        paiming = sorted(paiming, key=lambda x: x[2], reverse=True)

        # # 统计特征
        for i in paiming:

            select_d[i[1]]=select_d[i[1]]+i[2]

        print(paiming)
        for i in range(len(paiming)):
            df.loc[str(index)] = paiming[i]
            index += 1
        df.loc[str(index)] = ['  ', '  ', '  ', '  ', '  ','  ','  ','  ','  ','  ',' ',' ',' ',' ']
        index += 1
        df.loc[str(index)] =  ['模型', '特征', 'ACC','Recall','Precise','AUC','MCC','F1','Se','Sp','消耗时间','ACC中位数','ACC最大值','ACC最小值']
        index += 1

    # 特征效果排序
    s = {k: v for k, v in sorted(select_d.items(), key=lambda item: item[1], reverse=True)}
    key = list(s.keys())
    values = list(s.values())
    # with open('NEW_RAT/排行NEW_RAT.txt', 'w') as files:
    #     for i in range(len(key)):
    #         files.write(str(key[i]) + ':')
    #         files.write(str(values[i]))
    #         files.write('\n')


    # with pd.ExcelWriter('data.xlsx', mode='a') as writer:
    #     df.to_excel(writer, sheet_name='Sheet1', index=False, header=False)
    # df.to_excel('./NEW_RAT/RAT_SEPs_sORFs_features.xlsx', sheet_name="Sheet1",header='模型', engine='openpyxl')
    # df.to_excel('./NEW_HUM/HUM_model-FINALLY.xlsx', sheet_name="Sheet1", header='模型', engine='openpyxl')
    # df.to_excel('./First_kind/RAT/biaoge/TESTCatboost_RAT_SEPs_sORFs_Finally_features.xlsx', sheet_name="Sheet1", header='模型', engine='openpyxl')






















    # for i in code:
    #     paiming=[]
    #     for j in features:
    #         cmd= i +'.tudou'+"(\'./features/" + j + ".csv\',target,xianshi=False)"
    #         print('\n---------------------\n')
    #         print('\n',"模型:",i,"   特征:",j )
    #         [ACC,Recall,Precise,AUC,MCC,F1,Se,Sp]= eval(cmd)
    #         paiming.append([i, j,ACC,Recall,Precise,AUC,MCC,F1,Se,Sp])
    #
    #
    #     paiming = sorted(paiming, key=lambda x: x[2], reverse=True)
    #
    #     for i in range(len(paiming)):
    #         df.loc[str(index)] = paiming[i]
    #         index += 1
    #     df.loc[str(index)] = ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ', ]
    #     index += 1
    #     # print(df)
    # df.to_excel('评估2.xls', header='模型'