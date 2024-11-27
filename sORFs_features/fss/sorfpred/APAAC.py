import sys, os,re,platform
import numpy as np
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import readFasta
fastasFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\hump_pos12616.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/hump_pos12616.txt'
fastas = readFasta.readFasta(fastasFile)
fastasFile1 = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\hump_neg12616.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/hump_neg12616.txt'
fastas1 = readFasta.readFasta(fastasFile1)
import math
def APAAC(fastas, lambdaValue=3, w=0.05, **kw):
    dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\data\PAAC.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/data/PAAC.txt'
    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records) - 1):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
        AAProperty1.append([(j - meanI) / fenmu for j in i])

    encodings = []
    # header = ['#']
    # for i in AA:
    # 	header.append('Pc1.' + i)
    # for j in range(1, lambdaValue + 1):
    # 	for i in AAPropertyNames:
    # 		header.append('Pc2.' + i + '.' + str(j))
    # encodings.append(header)
    for i in fastas:
        sequence = re.sub('-' , '' , i[1])
        # a = sequence.ljust(100, '-')
        # b=re.sub( '-' , '' , a)
        # print(b)
        #
        # s= a[0:40]

        code = []
        theta = []
        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)):
                theta.append(sum([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]] for k in
                                  range(len(sequence) - n)]) / (len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)

        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [w * value / (1 + w * sum(theta)) for value in theta]
        encodings.append(code)
    return encodings
# res = APAAC(fastas)
# print(res)

if __name__=='__main__':
    train_p_data = APAAC(fastas)
    train_n_data = APAAC(fastas1)


    from sklearn.model_selection import train_test_split

    train_p_data, val_p_data = train_test_split(train_p_data,test_size=0.4,random_state=1,shuffle=True)
    val_p_data, test_p_data = train_test_split(val_p_data,test_size=0.5,random_state=1,shuffle=True)

    train_n_data, val_n_data = train_test_split(train_n_data,test_size=0.4,random_state=1,shuffle=True)
    val_n_data, test_n_data = train_test_split(val_n_data,test_size=0.5,random_state=1,shuffle=True)

    train_p_data = np.array(train_p_data)
    train_n_data = np.array(train_n_data)
    val_p_data = np.array(val_p_data)
    val_n_data = np.array(val_n_data)

    train_data = np.concatenate([train_p_data,train_n_data],axis=0)
    val_data = np.concatenate([val_p_data,val_n_data],axis=0)
    test_data = np.concatenate([test_p_data,test_n_data],axis=0)

    train_label = [1]*len(train_p_data) + [0]*len(train_n_data)
    val_label = [1]*len(val_p_data) + [0]*len(val_n_data)
    test_label = [1]*len(test_p_data) + [0]*len(test_n_data)

    train_label = np.array(train_label)
    val_label = np.array(val_label)
    test_label = np.array(test_label)
    x_train,x_val,x_test,y_train,y_val,y_test = train_data, val_data, test_data, train_label, val_label, test_label

    print("train pos num",train_p_data.shape)
    print("train neg num",train_n_data.shape)

    print("train data",train_data.shape)
    print("val data",val_data.shape)
    print("test data",test_data.shape)

    print("train label",train_label.shape)
    print("val label",val_label.shape)
    print("test label",test_label.shape)


    print("begin train ###")
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold


    def constructCATBoost8():
        from keras.layers import Dropout, MaxPooling1D, Flatten, Embedding, Dense, LSTM, Bidirectional, Conv1D, \
            GlobalMaxPooling1D, Activation
        from keras.models import Sequential
        from tensorflow import keras
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=5,
            verbose=1,
            mode="auto",
            baseline=None
        )
        callbacks = [early_stopping]
        model = Sequential()
        model.add(Embedding(input_dim=100, output_dim=64, input_length=40))
        model.add(Dropout(0.2))
        model.add(LSTM(64, return_sequences=True, input_shape=(40, 64)))
        model.add(Activation(activation='relu'))
        model.add(Dropout(0.2))
        model.add(LSTM(32, return_sequences=True, input_shape=(40, 32)))
        model.add(Activation(activation='relu'))
        model.add(Dropout(0.2))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=100, callbacks=callbacks,
        #           verbose=1)

        return model


    def constructCATBoost9():
        from keras.layers import Dropout, MaxPooling1D, Flatten, Embedding, Dense, LSTM, Bidirectional, Conv1D, \
            GlobalMaxPooling1D, Activation
        from keras.models import Sequential
        from tensorflow import keras
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=5,
            verbose=1,
            mode="auto",
            baseline=None
        )
        callbacks = [early_stopping]
        model2 = Sequential()
        model2.add(Embedding(input_dim=100, output_dim=64, input_length=40, mask_zero=False))
        model2.add(Dropout(0.2))
        model2.add(Conv1D(filters=128, kernel_size=3, strides=1, padding='valid', activation='relu'))
        model2.add(Activation(activation='relu'))
        model2.add(Dropout(0.2))
        model2.add(Conv1D(filters=64, kernel_size=3, strides=1, padding='valid', activation='relu'))
        model2.add(Activation(activation='relu'))
        model2.add(Dropout(0.2))
        model2.add(GlobalMaxPooling1D())
        model2.add(Dense(128))
        model2.add(Dropout(0.2))
        model2.add(Dense(1, activation='sigmoid'))
        model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # model2.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=100, callbacks=callbacks,
        #            verbose=1)

        return model2


    from tensorflow import keras
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    import matplotlib.pyplot as plt

    from tensorflow import keras
    import numpy as np
    import pandas as pd

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        verbose=1,
        mode="auto",
        baseline=None
    )
    callbacks = [early_stopping]
    model1 = constructCATBoost8()
    model1.fit(x_train, y_train, batch_size=20, validation_data=(x_val, y_val), epochs=100, callbacks=callbacks)
    model1.evaluate(x_test, y_test)
    y_pred_keras1 = model1.predict(x_test)
    fpr_keras1, tpr_keras1, thresholds_keras1 = roc_curve(y_test, y_pred_keras1)
    auc_keras1 = auc(fpr_keras1, tpr_keras1)

    model = constructCATBoost9()
    model.fit(x_train, y_train, batch_size=64, validation_data=(x_val, y_val), epochs=100, callbacks=callbacks)
    model.evaluate(x_test, y_test)
    y_pred_keras = model.predict(x_test)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)


    def constructCATBoost6():
        from sklearn import svm
        svc = svm.SVC(probability=True, kernel='rbf')
        return svc


    def constructCATBoost1():
        from catboost import CatBoostClassifier
        CAT = CatBoostClassifier(
            iterations=1000, learning_rate=0.03, depth=8, subsample=0.66,
            rsm=1

        )
        return CAT


    def constructCATBoost2():
        from lightgbm import LGBMClassifier
        lgb = LGBMClassifier(n_estimators=3835, learning_rate=0.0925472963334393, max_depth=13,
                             cat_smooth=30.61541228449972, bagging_fraction=0.6582040629836386, bagging_freq=9,
                             feature_fraction=0.6887307635842295, lambda_l1=0.9911958420007507,
                             lambda_l2=34.290859455476834, )
        return lgb


    def constructCATBoost3():
        from sklearn.linear_model import LogisticRegression as LR
        l = LR(C=1, penalty='l2', solver='liblinear')

        return l


    def constructCATBoost4():
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier(loss='deviance', n_estimators=2000, learning_rate=0.028673784262499755,
                                        max_depth=11, min_samples_leaf=59, min_samples_split=19,
                                        subsample=0.6447933070462003, max_features=0.0001)
        return gb


    def constructCATBoost5():
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=3789, max_depth=19, min_samples_leaf=82, min_samples_split=7,
                                    bootstrap=True, max_features='auto', n_jobs=1, random_state=3, verbose=100,
                                    oob_score=True,

                                    )

        return rf


    def constructCATBoost7():
        from xgboost import XGBClassifier
        xg = XGBClassifier(n_estimators=4906, colsample_bytree=0.961263998009275, learning_rate=0.004264475482427766,
                           max_depth=13, min_child_weight=45.48826116566274, subsample=0.8556594922541692)

        return xg


    clf=constructCATBoost6()
    clf.fit(x_train,y_train)



    # pred_res = clf.predict(test_data)
    pred_res = clf.predict_proba(test_data)[:, 1]

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