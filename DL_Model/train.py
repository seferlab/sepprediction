
import os
import tensorflow as tf
import keras
from sklearn import metrics
from keras.backend import set_session
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))
tf.compat.v1.disable_eager_execution()
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
np.random.seed(101)
from pathlib import Path


from model import base, BiGRU_base,lstm,MPMABP,transformer,gru,RNN,MLP,BiLSTM,TextLSTM,seq2seq,TextCNN,TextRNN,CNN_LSTM,CNN, Attention,MultiHeadAtten,Attention_BiLSTM,BiGRU_Attention,PositionalEmbedding_Transformer,transformer1


def train_my(train,test ,model_path,epoch,batch_si,hidden_size,j):
    jiaocha=True
    yan=1
    yan_li=[]
    if jiaocha:
        k_ = 10
        KF = KFold(n_splits=k_)
        x_train, y_train = train[0], train[1]
        # acc_all=0
        acc_all, se_all, sp_all, mcc_all, auc_all = 0, 0, 0, 0, 0
        acc_test, se_test, sp_test, mcc_test, auc_test = 0, 0, 0, 0, 0
        yan_li = []

        test_prob = []
        test_true = []

        for train_index, test_index in KF.split(x_train, y_train):
            X_train, X_test = x_train[train_index], x_train[test_index]
            Y_train, Y_test = y_train[train_index], y_train[test_index]
            Y_train = keras.utils.to_categorical(Y_train)
            Y_test1 = keras.utils.to_categorical(Y_test)
            if X_train.ndim != 3 and model_path != 'PositionalEmbedding_Transformer' :
                X_train = X_train.reshape(-1,  1, X_train.shape[1],)
                X_test = X_test.reshape(-1,  1, X_test.shape[1],)
            length = X_train.shape[1]
            out_length = 2
            if model_path != 'PositionalEmbedding_Transformer':
                length3 = X_train.shape[2]

            # 获取模型
            if model_path == 'base':
                model = base(hidden_size, out_length, )
            elif model_path == 'BiGRU_base':
                model = BiGRU_base(length,length3, out_length, )
            elif model_path == 'lstm':
                model = lstm(hidden_size,length,length3, out_length, )
            elif model_path == 'TextLSTM':
                model = TextLSTM(hidden_size, out_length, )
            elif model_path == 'MPMABP':
                model = MPMABP(length, out_length, )
            elif model_path == 'transformer':
                model = transformer(length, out_length, )
            elif model_path == 'transformer1':
                model = transformer1(length,length3, out_length, )
            elif model_path == 'gru':
                model = gru(hidden_size, out_length, )
            elif model_path == 'RNN':
                model = RNN(hidden_size, out_length, )
            elif model_path == 'MLP':
                model = MLP(hidden_size, out_length, )
            elif model_path == 'TextCNN':
                model = TextCNN(length, length3,hidden_size, out_length, )
            elif model_path=='seq2seq':
                model = seq2seq(length, length3,out_length,)
            elif model_path == 'BiLSTM':
                model = BiLSTM(hidden_size, out_length, )
            elif model_path =='TextRNN':
                model = TextRNN(hidden_size, out_length, )
            elif model_path == 'CNN_LSTM':
                model = CNN_LSTM(hidden_size, out_length, )
            elif model_path == 'CNN':
                model = CNN(hidden_size, out_length, )
            elif model_path == 'Attention':
                model = Attention(length, length3,hidden_size, out_length, )
            elif model_path == 'MultiHeadAtten':
                model = MultiHeadAtten(length, length3,hidden_size, out_length, )
            elif model_path == 'Attention_BiLSTM':
                model = Attention_BiLSTM(length, length3,hidden_size, 32, out_length)
            elif model_path == 'BiGRU_Attention':
                model = BiGRU_Attention(length,length3, 32, out_length)
            elif model_path == 'PositionalEmbedding_Transformer':
                model = PositionalEmbedding_Transformer(length, 32, 6000, out_length)
            else:
                print('no model')

            es = EarlyStopping(patience=6)
            optimizer = Adam(learning_rate=0.001)
            # model.compile(loss='binary_crossentropy', optimizer=optimizer , metrics=['accuracy'])
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, Y_train, nb_epoch=epoch, batch_size=batch_si,verbose=2,validation_data=(X_test, Y_test1), callbacks=[es]) #  callbacks=[es]
            pred_res=model.predict(X_test)[:, 1]



            pred_label = [0 if x < 0.5 else 1 for x in pred_res]
            acc = metrics.accuracy_score(y_true=Y_test, y_pred=pred_label)
            yan_li.append(acc)
            print('模型: {} ,特征: {}:第{}次验证集准确率为{}'.format(model_path,j,yan,acc))
            yan+=1
            acc_all+=acc
        print(yan_li)
        if test[0].ndim != 3 and model_path != 'PositionalEmbedding_Transformer':
            length = test[0].shape[1]
            test[0] = test[0].reshape(-1, 1,length,)
        pred_res = model.predict(test[0])[:, 1]

        y_pre = [0 if x < 0.5 else 1 for x in pred_res]

        test_prob.append(pred_res)
        test_true.append(test[1])

        tn, fp, fn, tp = metrics.confusion_matrix(y_pred=y_pre, y_true=test[1]).ravel()
        se = tp / (tp + fn)
        sp = tn / (tn + fp)
        recall = metrics.recall_score(y_pred=y_pre, y_true=test[1])
        precise = metrics.precision_score(y_pred=y_pre, y_true=test[1])
        auc = metrics.roc_auc_score(y_score=pred_res, y_true=test[1])
        acc1 = metrics.accuracy_score(y_pred=y_pre, y_true=test[1])
        f1 = metrics.f1_score(y_pred=y_pre, y_true=test[1])
        mcc = metrics.matthews_corrcoef(y_pred=y_pre, y_true=test[1])
        ap = metrics.average_precision_score(y_score=pred_res, y_true=test[1])
        acc_test += acc
        se_test += se
        sp_test += sp
        mcc_test += mcc
        auc_test += auc





        # pattern = re.compile(r'features/(.*?).csv')
        # match1 = re.search(pattern, CSV)
        # part1 = match1.group(1)
        # name = 'D:/xuexi/ORF/试验/second_kind/HUM/RNN_HUM_SEPs_sORFs_Finally_features'
        name = 'D:/xuexi/ORF/试验/model/RAT_DeepCPP'


        np.save(name + 'test_prob.npy', test_prob)
        np.save(name + 'test_true.npy', test_true)

        return  acc1,recall,precise,auc,mcc,f1,se,sp,acc_all/k_

    else:

        Path(model_path).mkdir(exist_ok=True)

        # data get

        X_train, y_train = train[0], train[1]
        y_train = keras.utils.to_categorical(y_train)
        # train
        length = X_train.shape[1]
        length3 = X_train.shape[2]
        out_length = 2

        if model_path == 'base':
            model = base(hidden_size, out_length, )
        elif model_path == 'BiGRU_base':
            model = BiGRU_base(length, length3, out_length, )
        elif model_path == 'lstm':
            model = lstm(hidden_size, length, length3, out_length, )
        elif model_path == 'TextLSTM':
            model = TextLSTM(hidden_size, out_length, )
        elif model_path == 'MPMABP':
            model = MPMABP(length, out_length, )
        elif model_path == 'transformer':
            model = transformer(length, out_length, )
        elif model_path == 'transformer1':
            model = transformer1(length, length3, out_length, )
        elif model_path == 'gru':
            model = gru(hidden_size, out_length, )
        elif model_path == 'RNN':
            model = RNN(hidden_size, out_length, )
        elif model_path == 'MLP':
            model = MLP(hidden_size, out_length, )
        elif model_path == 'TextCNN':
            model = TextCNN(length, length3, hidden_size, out_length, )
        elif model_path == 'seq2seq':
            model = seq2seq(length, length3, out_length, )
        elif model_path == 'BiLSTM':
            model = BiLSTM(hidden_size, out_length, )
        elif model_path == 'TextRNN':
            model = TextRNN(hidden_size, out_length, )
        elif model_path == 'CNN_LSTM':
            model = CNN_LSTM(hidden_size, out_length, )
        elif model_path == 'CNN':
            model = CNN(hidden_size, out_length, )
        elif model_path == 'Attention':
            model = Attention(length, length3, hidden_size, out_length, )
        elif model_path == 'MultiHeadAtten':
            model = MultiHeadAtten(length, length3, hidden_size, out_length, )
        elif model_path == 'Attention_BiLSTM':
            model = Attention_BiLSTM(length, length3, hidden_size, 32, out_length)
        elif model_path == 'BiGRU_Attention':
            model = BiGRU_Attention(length, length3, 32, out_length)
        elif model_path == 'PositionalEmbedding_Transformer':
            model = PositionalEmbedding_Transformer(length, 32, 6000, out_length)
        else:
            print('no model')

        if X_train.ndim!=3 and model_path!='PositionalEmbedding_Transformer' :
            X_train= X_train.reshape(-1,  length,1, )
        es = EarlyStopping(patience=8)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, nb_epoch=epoch, batch_size=batch_si, verbose=2,callbacks=[es])
        pred_train = model.predict(X_train)[:, 1]



            #
            # each_model = os.path.join(model_path, 'model' + str(counter) + '.h5')
            # model.save(each_model)

        # test[1] = keras.utils.to_categorical(test[1])




        if test[0].ndim != 3 and model_path!='PositionalEmbedding_Transformer':
            length=test[0].shape[1]
            test[0]= test[0].reshape(-1, 1, length)

        pred_res = model.predict(test[0])[:, 1]
        y_pre = [0 if x < 0.5 else 1 for x in pred_res]

        tn, fp, fn, tp = metrics.confusion_matrix(y_pred=y_pre, y_true=test[1]).ravel()
        se = tp / (tp + fn)
        sp = tn / (tn + fp)

        recall = metrics.recall_score(y_pred=y_pre, y_true=test[1])
        precise = metrics.precision_score(y_pred=y_pre, y_true=test[1])
        auc = metrics.roc_auc_score(y_score=pred_res, y_true=test[1])
        acc = metrics.accuracy_score(y_pred=y_pre, y_true=test[1])
        f1 = metrics.f1_score(y_pred=y_pre, y_true=test[1])
        mcc = metrics.matthews_corrcoef(y_pred=y_pre, y_true=test[1])
        ap = metrics.average_precision_score(y_score=pred_res, y_true=test[1])

                # print("当前模型为:{} ,训练集上的准确率为:{}：".format(model_path,acc))
                # print("recall:", recall, flush=True)
                # print("precise:", precise, flush=True)
                # print("se:", se, flush=True)
                # print("sp:", sp, flush=True)
                # print("f1:", f1, flush=True)
                # print("mcc:", mcc, flush=True)
                # print("auc:", auc, flush=True)
                # print("ap:", ap, flush=True)



        print(model_path, 'ACC: ', acc)
        return  acc,recall,precise,auc,mcc,f1,se,sp,model


