import copy
import random
from Bio import SeqIO
import sys
from xgboost import XGBClassifier
from sklearn import metrics
import numpy as np
import time
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from train_Stank import train_my
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from statistics import median,mean
from main11 import data_d
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#TensorFlow按需分配显存
config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
import joblib
def me(paiming_ceshi):
    pingjun=[]
    for i in range(9):
        tong = []
        for j in range(len(paiming_ceshi)):
            tong.append(paiming_ceshi[j][i])
        if i==0:
            me_dian = median(tong)
            max_acc=max(tong)
            min_acc = min(tong)

        tong = mean(tong)


        pingjun.append(tong)

    return round(pingjun[0],3),round(pingjun[1],3),round(pingjun[2],3),round(pingjun[3],3),round(pingjun[4],3),round(pingjun[5],3),round(pingjun[6],3),round(pingjun[7],3), round(pingjun[8],3),round(me_dian,3),round(max_acc,3),round(min_acc,3)


def get_xgboost():
    xgb_model = XGBClassifier()
    return xgb_model

def get_rf():
    rf_clf = RandomForestClassifier(n_estimators=500)
    return rf_clf

def get_catboost():
    rf_clf = CatBoostClassifier(iterations=500,verbose=False)
    return rf_clf

def get_lightgbm():
    lgb_model = LGBMClassifier(n_estimators=500,max_depth=60,learning_rate=0.2,num_leavel=30)
    return lgb_model

def get_GDBT():
    kn_ = GradientBoostingClassifier()
    return kn_
def get_svm():
    clf = SVC(probability=True)
    return clf
def get_Lgst():
    lg = LogisticRegression()
    return lg
def get_de_tree():
    tree = DecisionTreeClassifier()
    return tree
def get_knn():
    kn_ = KNeighborsClassifier()
    return kn_

def main(features,target,models,code):




    global clf, test_allss
    std = StandardScaler()
    features = std.fit_transform(features)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20,random_state=30)
    deep_features='esm_Pep_NEW_Rat'
    dirs = ['TextCNN', 'TextRNN', 'TextLSTM', 'Attention', 'MultiHeadAtten', 'Attention_BiLSTM', 'BiGRU_Attention',
            'CNN_LSTM', 'CNN', 'RNN', 'lstm', 'BiLSTM', 'gru', 'MLP', ]
    threshold = 0.5
    # 循环次数
    model_num = 1

    hidden_size = 30

    embedding_dimension = 100
    pool_size = 5
    fully_dimension = 128

    drop_out = 0.2
    learning_rate = 0.001
    ephch = 100

    batch_size = 32

    k_ = 10
    KF = KFold(n_splits=k_)
    acc_all = 0

    K_f= False # 设置为True的时候,每次test 数据集在第一阶段都输入进去, 第二阶段相加取平均
    jishu=1

    # 保存ACC
    yz_acc = []
    yz_acc_ju = []
    for i in range(len(code)):
        yz_acc.append(0)
        yz_acc_ju.append([])

    '''
    读取深度特征
    
    '''

    features = np.load('./NEW_RAT/esm_Pep_NEW_Rat.npy', allow_pickle=True, ).astype(np.float32)
    # 标准化
    std, avgs = \
            features.std(axis=0), features.sum(axis=0) / features.shape[0]

    for i in range(features.shape[1]):  # 43 是维度, 第二维
            # print(maximums[i], minimums[i], avgs[i])
        features[:, i] = (features[:, i] - avgs[i]) / std[i]
    tr_data, te_data, tr_label, te_label = train_test_split(features, target, test_size=0.20,random_state=30)
    result=[]
    acc_all, se_all, sp_all, mcc_all, auc_all = 0, 0, 0, 0, 0
    acc_test, se_test, sp_test, mcc_test, auc_test = 0, 0, 0, 0, 0
    yan_li = []
    val_prob = []
    val_true = []
    test_prob = []
    test_true = []

    for train_index, test_index in KF.split(x_train, y_train):

        deep_X_train, deep_X_test = tr_data[train_index], tr_data[test_index]
        deep_Y_train, deep_Y_test = tr_label[train_index], tr_label[test_index]

        X_train, X_test = x_train[train_index], x_train[test_index]
        Y_train, Y_test = y_train[train_index], y_train[test_index]
        train_layer2_input = []
        test_layer2_input = []
        test_all = []
        for i in models:

            if i in dirs:

                train = [np.array(deep_X_train), np.array(deep_Y_train)]
                test = [np.array(deep_X_test), np.array(deep_Y_test)]
                test_last=[np.array(te_data), np.array(te_label)]
                test.append(0.5)
                train_xgboost_res1,test_xgboost_res1, model=train_my(train, test,test_last, model_num, i, ephch, batch_size,

                         hidden_size,True)
                train_xgboost_res1 = train_xgboost_res1.reshape(len(train_xgboost_res1), 1)
                test_xgboost_res1 = test_xgboost_res1.reshape(len(test_xgboost_res1), 1)
                train_layer2_input.append(train_xgboost_res1)
                test_layer2_input.append(test_xgboost_res1)


                '''
                交叉验证最后一次,放入测试集
                '''
                if K_f==True or jishu==10:
                    test_a = [np.array(te_data), np.array(te_label)]
                    if test_a[0].ndim != 3 :
                        length = test_a[0].shape[1]
                        test_a[0] = test_a[0].reshape(-1, 1, length)
                    pred_a = model.predict(test_a[0])[:, 1]
                    pred_a = pred_a.reshape(len(pred_a), 1)
                    test_all .append(pred_a )


            if i =="XGBoost":
                xgboost_f1  = get_xgboost().fit(X_train, Y_train)
                train_xgboost_res1 = xgboost_f1.predict_proba(X_train)[:,1]
                train_xgboost_res1= train_xgboost_res1.reshape(len(train_xgboost_res1), 1)
                test_xgboost_res1 = xgboost_f1.predict_proba(X_test)[:,1]
                test_xgboost_res1 = test_xgboost_res1.reshape(len(test_xgboost_res1), 1)
                train_layer2_input.append(train_xgboost_res1)
                test_layer2_input.append(test_xgboost_res1)

                if K_f == True or jishu==10:

                    test_aa1 = xgboost_f1.predict_proba(x_test)[:, 1]
                    test_aa1 = test_aa1.reshape(len(test_aa1), 1)
                    test_all.append(test_aa1)
            if i =='logistic_regression':
                lgst_f1 = get_Lgst().fit(X_train, Y_train)
                train_lgst_res1 = lgst_f1.predict_proba(X_train)[:, 1]
                train_lgst_res1 = train_lgst_res1.reshape(len(train_lgst_res1), 1)
                test_lgst_res1 = lgst_f1.predict_proba(X_test)[:, 1]
                test_lgst_res1 = test_lgst_res1.reshape(len(test_lgst_res1), 1)
                train_layer2_input.append(train_lgst_res1)
                test_layer2_input.append(test_lgst_res1)
                if K_f == True or jishu==10:
                    test_aa2 = lgst_f1.predict_proba(x_test)[:, 1]
                    test_aa2 = test_aa2.reshape(len(test_aa2), 1)
                    test_all.append(test_aa2)
            if i =="random_forest":
                rf_f1  = get_rf().fit(X_train, Y_train)
                train_rf_res1 = rf_f1.predict_proba(X_train)[:,1]
                test_rf_res1 = rf_f1.predict_proba(X_test)[:,1]
                train_rf_res1 = train_rf_res1.reshape(len(train_rf_res1), 1)
                test_rf_res1  = test_rf_res1 .reshape(len(test_rf_res1 ), 1)
                train_layer2_input.append(train_rf_res1)
                test_layer2_input.append(test_rf_res1)
                if K_f == True or jishu==10:
                    test_aa3= rf_f1.predict_proba(x_test)[:, 1]
                    test_aa3 = test_aa3.reshape(len(test_aa3), 1)
                    test_all.append(test_aa3)
            if i =='Catboost':
                catboost_f1  = get_catboost().fit(X_train, Y_train)
                train_catboost_res1 = catboost_f1.predict_proba(X_train)[:,1]
                test_catboost_res1 = catboost_f1.predict_proba(X_test)[:,1]
                train_catboost_res1 = train_catboost_res1.reshape(len(train_catboost_res1), 1)
                test_catboost_res1 = test_catboost_res1.reshape(len(test_catboost_res1), 1)
                train_layer2_input.append(train_catboost_res1)
                test_layer2_input.append(test_catboost_res1)
                #
                if K_f == True or jishu==10:
                    test_aa4 = catboost_f1.predict_proba(x_test)[:, 1]

                    test_aa4 = test_aa4.reshape(len(test_aa4), 1)
                    test_all.append(test_aa4)
            if i == 'LightGBM':
                lightgbm_f1 = get_lightgbm().fit(X_train, Y_train)
                train_lightgbm_res1 = lightgbm_f1.predict_proba(X_train)[:,1]
                test_lightgbm_res1 = lightgbm_f1.predict_proba(X_test)[:,1]
                train_lightgbm_res1 = train_lightgbm_res1.reshape(len(train_lightgbm_res1), 1)
                test_lightgbm_res1 = test_lightgbm_res1.reshape(len(test_lightgbm_res1), 1)
                train_layer2_input.append(train_lightgbm_res1)
                test_layer2_input.append(test_lightgbm_res1)
                if K_f == True or jishu==10:
                    test_aa5 = lightgbm_f1.predict_proba(x_test)[:, 1]
                    test_aa5 = test_aa5.reshape(len(test_aa5), 1)
                    test_all.append(test_aa5)
            if i =='GDBT':
                GDBT_f1 = get_GDBT().fit(X_train, Y_train)
                train_GDBT_f1_res1 = GDBT_f1.predict_proba(X_train)[:,1]
                test_GDBT_f1_res1 = GDBT_f1.predict_proba(X_test)[:,1]
                train_GDBT_f1_res1 = train_GDBT_f1_res1.reshape(len(train_GDBT_f1_res1), 1)
                test_GDBT_f1_res1 = test_GDBT_f1_res1.reshape(len(test_GDBT_f1_res1), 1)
                train_layer2_input.append(train_GDBT_f1_res1)
                test_layer2_input.append(test_GDBT_f1_res1)
                if K_f == True or jishu==10:
                    test_aa6 = GDBT_f1.predict_proba(x_test)[:, 1]
                    test_aa6 = test_aa6.reshape(len(test_aa6), 1)
                    test_all.append(test_aa6)
            if i=='SVM':
                svm_f1 = get_svm().fit(X_train, Y_train)
                train_svm_f1_res1 = svm_f1.predict_proba(X_train)[:,1]
                test_svm_f1_res1 = svm_f1.predict_proba(X_test)[:,1]

                train_svm_f1_res1 = train_svm_f1_res1.reshape(len(train_svm_f1_res1), 1)
                test_svm_f1_res1 = test_svm_f1_res1.reshape(len(test_svm_f1_res1), 1)
                train_layer2_input.append(train_svm_f1_res1)
                test_layer2_input.append(test_svm_f1_res1)
                if K_f == True or jishu==10:
                    print('jishu:', jishu)
                    test_aa7= svm_f1.predict_proba(x_test)[:, 1]
                    test_aa7 = test_aa7.reshape(len(test_aa7), 1)
                    test_all.append(test_aa7)
            if i=='decision_tree':
                de_tree_f1 = get_de_tree().fit(X_train, Y_train)
                train_de_tree_f1_f1_res1 = de_tree_f1.predict_proba(X_train)[:,1]
                test_de_tree_f1_f1_res1 = de_tree_f1.predict_proba(X_test)[:,1]
                train_de_tree_f1_f1_res1 = train_de_tree_f1_f1_res1.reshape(len(train_de_tree_f1_f1_res1), 1)
                test_de_tree_f1_f1_res1 = test_de_tree_f1_f1_res1.reshape(len(test_de_tree_f1_f1_res1), 1)
                train_layer2_input.append(train_de_tree_f1_f1_res1)
                test_layer2_input.append(test_de_tree_f1_f1_res1)
                if K_f == True or jishu==10:
                    test_aa8 = de_tree_f1.predict_proba(x_test)[:, 1]
                    test_aa8 = test_aa8.reshape(len(test_aa8), 1)
                    test_all.append(test_aa8)

            if i=='KNN':
                de_knn = get_knn().fit(X_train, Y_train)
                train_de_tree_f1_f1_res1 = de_knn.predict_proba(X_train)[:,1]
                test_de_tree_f1_f1_res1 = de_knn.predict_proba(X_test)[:,1]
                train_de_tree_f1_f1_res1 = train_de_tree_f1_f1_res1.reshape(len(train_de_tree_f1_f1_res1), 1)
                test_de_tree_f1_f1_res1 = test_de_tree_f1_f1_res1.reshape(len(test_de_tree_f1_f1_res1), 1)
                train_layer2_input.append(train_de_tree_f1_f1_res1)
                test_layer2_input.append(test_de_tree_f1_f1_res1)
                if K_f == True or jishu==10:
                    test_aa8 = de_knn.predict_proba(x_test)[:, 1]
                    test_aa8 = test_aa8.reshape(len(test_aa8), 1)
                    test_all.append(test_aa8)




        train_layer2_inputs = np.concatenate([train_layer2_input[0], train_layer2_input[1]], axis=-1)
        for i in range(2,len(train_layer2_input)):
            train_layer2_inputs=np.concatenate([train_layer2_inputs,train_layer2_input[i]],axis=-1)

        test_layer2_inputs = np.concatenate([test_layer2_input[0], test_layer2_input[1]], axis=-1)
        for i in range(2,len(test_layer2_input)):
            test_layer2_inputs=np.concatenate([test_layer2_inputs,test_layer2_input[i]],axis=-1)



        stdd = StandardScaler()
        train_layer2_inputs = stdd.fit_transform(train_layer2_inputs)
        test_layer2_inputs = stdd.transform(test_layer2_inputs)




        # test在第一部分只运行一次的时候
        if K_f!=True and jishu==10:
            test_alls = np.concatenate([test_all[0], test_all[1]], axis=-1)
            for i in range(2, len(test_all)):
                test_alls = np.concatenate([test_alls, test_all[i]], axis=-1)
            test_allss = stdd.transform(test_alls)





        for i,last in enumerate(code):

                # 最后的模型预测
            if last == 'XGBoost':
                clf = get_xgboost().fit(train_layer2_inputs,Y_train)
            elif last=='random_forest':
                clf = get_rf().fit(train_layer2_inputs, Y_train)
            elif last=='Catboost':
                clf = get_catboost().fit(train_layer2_inputs, Y_train)
            elif last=='LightGBM':
                clf = get_lightgbm().fit(train_layer2_inputs, Y_train)
            elif last=='GDBT':
                clf = get_GDBT().fit(train_layer2_inputs, Y_train)
            elif last=='SVM':
                clf = get_svm().fit(train_layer2_inputs,Y_train)
            elif last=='logistic_regression':
                clf = get_Lgst().fit(train_layer2_inputs, Y_train)
            elif last=='decision_tree':
                clf = get_de_tree().fit(train_layer2_inputs,Y_train)
            elif last=='KNN':
                clf = get_knn().fit(train_layer2_inputs,Y_train)




            pred_res = clf.predict_proba(test_layer2_inputs)[:, 1]

            # 验证集的检测
            pred_label = [0 if x < 0.5 else 1 for x in pred_res]
            acc = metrics.accuracy_score(y_true=Y_test, y_pred=pred_label)
            yz_acc[i]+=acc

            yz_acc_ju[i].append(round(acc,3))

            if jishu!=10:

                print('第{}次验证集acc:{},最终为:{},'.format(jishu,acc,last))

            if jishu==10:
                print('\n\n\n\n')
                print('验证集平均acc:{} ,最终模型为{}'.format(yz_acc[i]/10,code[i]))
                pred_res = clf.predict_proba(test_allss)[:, 1]

                test_prob.append(pred_res)
                test_true.append(y_test)


                pred_label = [0 if x < 0.5 else 1 for x in pred_res]

                #

                acc = metrics.accuracy_score(y_true=y_test, y_pred=pred_label)
                f1 = metrics.f1_score(y_true=y_test, y_pred=pred_label)
                mcc = metrics.matthews_corrcoef(y_true=y_test, y_pred=pred_label)

                auc = metrics.roc_auc_score(y_true=y_test, y_score=pred_res)
                ap = metrics.average_precision_score(y_true=y_test, y_score=pred_res)
                tn, fp, fn, tp = metrics.confusion_matrix(y_pred=pred_label, y_true=y_test).ravel()
                se = tp / (tp + fn)
                sp = tn / (tn + fp)

                recall = metrics.recall_score(y_pred=pred_label, y_true=y_test)
                precise = metrics.precision_score(y_pred=pred_label, y_true=y_test)

                acc_test += acc
                se_test += se
                sp_test += sp
                mcc_test += mcc
                auc_test += auc


                test_prob = np.concatenate(test_prob)
                test_true = np.concatenate(test_true)


                name = 'D:/xuexi/ORF/试验/model/DEEP-1'

                #
                # np.save(name + 'test_prob.npy', test_prob)
                # np.save(name + 'test_true.npy', test_true)

                result.append([acc, recall, precise, auc, mcc, f1, se, sp, yz_acc[i]/10,yz_acc_ju[i]])

        jishu += 1

    return result


if __name__=='__main__':
    columns = ['基础模型','最终', '特征', 'ACC', 'Recall', 'Precise', 'AUC', 'MCC', 'F1', 'Se', 'Sp', '验证集平均', '消耗时间','交叉验证集合']

    df = pd.DataFrame(data=[['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-','-','-','-',]], columns=columns)
    target = pd.read_csv('./features/rat_label.csv', header=None)
    target = target.values.ravel()




    # feature = ['AAC+CTDC+QSOrder+PAAC', 'AAC+CTDC+QSOrder+PAAC+APAAC',
    #            'AAC+CTDC+QSOrder+PAAC+APAAC+EAAC', 'AAC+CTDC+QSOrder+PAAC+APAAC+EAAC+CKSAAP'
    #     , 'AAC+CTDC+QSOrder+PAAC+APAAC+EAAC+CKSAAP+DDE', 'AAC+CTDC+QSOrder+PAAC+APAAC+EAAC+CKSAAP+DDE+CTDT','AAC+CTDC+QSOrder+PAAC+APAAC+EAAC+CKSAAP+DDE+CTDT+GDPC',
    #            'AAC+CTDC+QSOrder+PAAC+APAAC+EAAC+CKSAAP+DDE+CTDT+GDPC+CTDD']
    feature=['RAT_SEPs_sORFs_Fused Finally features' ,]
    # feature = ['AAC+CTDC+QSOrder+PAAC', 'AAC+CTDC+QSOrder+PAAC+APAAC',]

    # 基准模型
    dirs = ['RNN', 'lstm', 'BiLSTM', 'gru', 'CNN_LSTM', 'CNN', 'TextCNN', 'Attention', 'MultiHeadAtten',
            'Attention_BiLSTM', 'BiGRU_Attention', 'MPMABP', 'transformer', ]
    jizhun = ['XGBoost', 'random_forest', 'LightGBM','Catboost','GDBT','SVM','logistic_regression','decision_tree','KNN']




    # jizhuns = [['TextCNN', 'CNN','SVM','KNN','Catboost','random_forest']]

    jizhuns = [
        ['Catboost', 'Attention',], ]




    # 最终模型
    code=['XGBoost', 'random_forest', 'LightGBM','Catboost','GDBT','SVM','logistic_regression','decision_tree','KNN']


    code = ['logistic_regression',]
    # 循环多少次



    index=1

    # 遍历最后的模型



    for jizhun in jizhuns:

        for j in feature:
            paiming = []
            # 读取数据
            features = pd.read_csv(
                './huatu/'+j+'.csv',
                header=None)

            time_all = 0


            start = time.time()
            print('\n---------------------\n')
            jizhun_name = ' + '.join(jizhun)
            print('基准模型{}'.format(jizhun_name))
            print('\n', " 机器特征:{}  ".format(j,))
            # 模型训练
            cmd = 'main(features,target,jizhun,code)'

            # 模型训练
            result= eval(cmd)
            #计算消耗时间
            end = time.time()
            time_all = time_all + end - start

            # 选出平均值保存
            for s,z in enumerate(code):

                paiming.append([jizhun_name,z, j,round(result[s][0],3),round(result[s][1],3),round(result[s][2],3),round(result[s][3],3),round(result[s][4],3),round(result[s][5],3),round(result[s][6],3),round(result[s][7],3), round(result[s][8],3), round(time_all,2),result[s][9]])
            # 特征之间以准确率为基准排序
            paiming = sorted(paiming, key=lambda x: x[10], reverse=True)
            print(paiming)

            # # 统计特征


            for i in range(len(paiming)):
                df.loc[str(index)] = paiming[i]
                index += 1
            df.loc[str(index)] = ['  ', '  ', '  ', '  ', '  ','  ','  ','  ','  ','  ',' ',' ',' ',' ',]
            index += 1
            df.loc[str(index)] = ['基础模型','最终', '特征', 'ACC', 'Recall', 'Precise', 'AUC', 'MCC', 'F1', 'Se', 'Sp', '验证集平均', '消耗时间','交叉验证集合']
            index += 1

        print(df)
        # name=1
        # while True:
        #     name_zu='./NEW_HUM/TEXTDEEP'
        #     name_zhong=name_zu+str(name)+'.xlsx'
        #     if os.path.exists(name_zhong) ==False:
        #         df.to_excel(name_zhong, engine='openpyxl',sheet_name='Sheet1',header='模型')
        #         break
        #     else:
        #         name+=1
        #
        # df = pd.DataFrame(data=[['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']], columns=columns)