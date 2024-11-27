from keras.layers import Dropout, MaxPooling1D, Flatten, Embedding, Dense, LSTM, Bidirectional, Conv1D, \
        GlobalMaxPooling1D, Activation
from keras.models import Sequential
from tensorflow import keras
from tensorflow import keras
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt

from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def constructCATBoost8():

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
    model.add(Embedding(input_dim=100, output_dim=64, input_length=184))
    # model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=True, input_shape=(40, 64)))
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=True, input_shape=(40, 32)))
    model.add(Activation(activation='relu'))
    # model.add(Dropout(0.2))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128))
    # model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=100, callbacks=callbacks,
    #           verbose=1)

    return model

early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.001,
        patience=5,
        verbose=1,
        mode="auto",
        baseline=None
    )
features = '../features/AAC+CTDC+QSOrder+PAAC+APAAC.csv'
feature = pd.read_csv(features,header=None)
target = pd.read_csv('../features/my.csv',header=None)
target = target.values.ravel()

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=0.20)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20)

callbacks = [early_stopping]
model1 = constructCATBoost8()
model1.fit(x_train, y_train, batch_size=20, validation_data=(x_val, y_val), epochs=100, callbacks=callbacks)
print(model1.evaluate(x_test, y_test))
y_pred_keras1 = model1.predict(x_test)
fpr_keras1, tpr_keras1, thresholds_keras1 = roc_curve(y_test, y_pred_keras1)
auc_keras1 = auc(fpr_keras1, tpr_keras1)

# model = constructCATBoost9()
# model.fit(x_train, y_train, batch_size=64, validation_data=(x_val, y_val), epochs=100, callbacks=callbacks)
# model.evaluate(x_test, y_test)
# y_pred_keras = model.predict(x_test)
# fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
# auc_keras = auc(fpr_keras, tpr_keras)