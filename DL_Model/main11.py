# -*- coding: utf-8 -*-
# @Author  : twd
# @FileName: demo.py
# @Software: PyCharm
from train import train_my

import pandas as pd
import os
import time
import numpy as np
from pathlib import Path
from statistics import mean,median
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from test import test_my
# Path(dir).mkdir(exist_ok=True)


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


def data_d(features,target,threshold):
    di='./NEW_HUM/'
    dir =di+features+'.csv'


    if not os.path.exists(dir)  :
        dir = di + features + '.npy'
        features=np.load(dir, allow_pickle=True, ).astype(np.float32)
        #
        # maximums, minimums, avgs = \
        #                      features.max(axis=0), features.min(axis=0), features.sum(axis=0) / features.shape[0]
        # # 对数据进行归一化处理
        # for i in range(features.shape[1]):  # 43 是维度, 第二维
        #     #print(maximums[i], minimums[i], avgs[i])
        #     features [:, i] = (features [:, i] - minimums[i]) / (maximums[i] - minimums[i])


    # 标准化
        std, avgs = \
            features.std(axis=0), features.sum(axis=0) / features.shape[0]

        for i in range(features.shape[1]):  # 43 是维度, 第二维
            # print(maximums[i], minimums[i], avgs[i])
            features[:, i] = (features[:, i] - avgs[i]) / std[i]


    else:

        features = pd.read_csv(dir,
            header=None)
        std = StandardScaler()
        features = std.fit_transform(features)
    tr_data, te_data, tr_label, te_label = train_test_split(features, target, test_size=0.20, random_state=30)
    # tr_data, te_data, tr_label, te_label = train_test_split(features, target, test_size=0.20, random_state=30)
    train = [np.array(tr_data), np.array(tr_label)]
    test = [np.array(te_data), np.array(te_label)]
    test.append(threshold)
    return train,test



def main():
    columns = ['模型', '特征', 'ACC', 'Recall', 'Precise', 'AUC', 'MCC', 'F1', 'Se', 'Sp','测试集ACC', '消耗时间', '测试集ACC中位数',
               '测试集ACC最大值', '测试集ACC最小值']

    df = pd.DataFrame(data=[['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-']], columns=columns)

    # 读取标签
    # target = pd.read_csv('./features/human_label.csv', header=None)
    target = pd.read_csv('./features/human_label.csv', header=None)
    target = target.values.ravel()




    features = ['HUM_SEPs_sORFs_Fused Finally features']
    # features = ['AAC']

    dirs = ['TextCNN','TextRNN','TextLSTM','Attention', 'MultiHeadAtten','Attention_BiLSTM', 'BiGRU_Attention',
        'CNN_LSTM','CNN','RNN','lstm','BiLSTM','gru','MLP','PositionalEmbedding_Transformer']
    dirs =['TextCNN','TextRNN','TextLSTM','Attention', 'MultiHeadAtten','Attention_BiLSTM', 'BiGRU_Attention',
        'CNN_LSTM','CNN','RNN','lstm','BiLSTM','gru','MLP',]
    dirs = ['TextCNN','CNN','RNN','lstm','BiLSTM','gru','MLP']
    # dirs=['CNN_LSTM','TextRNN','TextLSTM','Attention', 'MultiHeadAtten','Attention_BiLSTM', 'BiGRU_Attention']
    dirs = ['RNN', 'gru', 'BiLSTM', 'MPMABP', 'MLP', 'TextCNN', 'CNN_LSTM', 'CNN']
    dirs = ['RNN', 'gru', 'BiLSTM']
    dirs = [ 'Attention_BiLSTM', 'BiGRU_Attention']
    dirs = [ 'MultiHeadAtten', 'Attention',]
    dirs = ['MLP', 'TextCNN', ]
    # dirs = ['CNN', 'CNN_LSTM', 'lstm']
    dirs = ['lstm']


    # threshold = 0.5
    # # 循环次数
    # model_num = 1
    #
    # hidden_size = 256
    #
    # embedding_dimension = 100
    # pool_size = 5
    # fully_dimension = 128
    #
    # drop_out = 0.1
    # learning_rate = 0.001
    # ephch = 128
    # batch_size = 128
    # # 循环几次取平均
    # xunhuan = 1

    threshold = 0.5
    # 循环次数
    model_num = 1

    hidden_size = 256

    embedding_dimension = 100
    pool_size = 5
    fully_dimension = 128

    drop_out = 0.1
    learning_rate = 0.001
    ephch = 128
    batch_size = 32
    # 循环几次取平均
    xunhuan = 1

    index = 1
    select_d = {}
    for i in features:
        select_d[i] = 0
    for i in dirs:

        paiming = []
        for j in features:

            paiming_ceshi = []
            time_all = 0

            for s in range(xunhuan):
                start = time.time()
                print('\n---------------------\n')
                print('\n', "模型:{}  特征:{}  当前循环{}次,一共循环{}次".format(i, j, s + 1, xunhuan, ))
                # 读取数据
                train, test = data_d(j, target,threshold)
                # 模型训练
                [ACC, Recall, Precise, AUC, MCC, F1, Se, Sp,ceshi] = train_my(train,test,  i, ephch, batch_size,

                                                                        hidden_size,j)

                print('验证集{},\n\n\n\n\n\n\n:'.format(ceshi))

                paiming_ceshi.append([ACC, Recall, Precise, AUC, MCC, F1, Se, Sp,ceshi])

                # 计算消耗时间
                end = time.time()
                time_all = time_all + end - start

            # 选出平均值保存
            [ACC, Recall, Precise, AUC, MCC, F1, Se, Sp, ceshi, me_dian, max_value, min_value] = me(paiming_ceshi)
            paiming.append(
                [i, j, ACC, Recall, Precise, AUC, MCC, F1, Se,Sp, ceshi, round(time_all / xunhuan, 2), me_dian, max_value,
                 min_value])
        # 以准确率为基准排序
        paiming = sorted(paiming, key=lambda x: x[9], reverse=True)

        # # 统计特征
        for i in paiming:
            select_d[i[1]] = select_d[i[1]] + i[2]

        print(paiming)
        for i in range(len(paiming)):
            df.loc[str(index)] = paiming[i]
            index += 1
        df.loc[str(index)] = ['  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ', '  ', ' ', ' ', ' ', ' ', ' ']
        index += 1
        # df.loc[str(index)] =  ['模型', '特征', 'ACC','Recall','Precise','AUC','MCC','F1','Se','Sp','消耗时间','ACC中位数','ACC最大值','ACC最小值']
        # index += 1

    # 特征效果排序
    # s = {k: v for k, v in sorted(select_d.items(), key=lambda item: item[1], reverse=True)}
    # key = list(s.keys())
    # values = list(s.values())
    # with open('排行4.txt', 'w') as files:
    #     for i in range(len(key)):
    #         files.write(str(key[i]) + ':')
    #         files.write(str(values[i]))
    #         files.write('\n')

    # with pd.ExcelWriter('data.xlsx', mode='a') as writer:
    #     df.to_excel(writer, sheet_name='Sheet1', index=False, header=False)

    name=1
    while True:
        name_zu='./NEW_HUM/TEST/'
        name_zhong=name_zu+str(name)+'.xlsx'
        if os.path.exists(name_zhong) ==False:
            df.to_excel(name_zhong, engine='openpyxl',sheet_name='Sheet1',header='模型')
            break
        else:
            name+=1
#
    # test_my(test, para, model_num, dir)

if __name__ == '__main__':
    # executing the main function
    main()

