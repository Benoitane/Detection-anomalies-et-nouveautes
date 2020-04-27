# -*- coding: utf-8 -*-
"""

"""

#import os
#os.chdir("C:/Users/Philo/Documents/3A -- ENSAE/apprentissage avancé/détection d'anomalies/Detection-anomalies-et-nouveautes")
#import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from utils.evaluate import plot_var_in_out, evaluation_detection, deep_predict, evaluate
from utils.preprocessing import upload_data, split_data, prepro_data, simul_gaussian_data, simul_uniform_data

def CustomAutoencoder(input_dim, encoding_dim=10):
    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="tanh",activity_regularizer=regularizers.l1(10e-5))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
    encoder = Dense(int(2), activation="tanh")(encoder)
    decoder = Dense(int(encoding_dim/ 2), activation='tanh')(encoder)
    decoder = Dense(int(encoding_dim), activation='tanh')(decoder)
    decoder = Dense(input_dim, activation='tanh')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    print(autoencoder.summary())
    return autoencoder


def AlgoTrainPredict(name, algorithm, algo_particuliers, X_train, X_test, y_test, config, scale="MinMax", var1=10, var2=20):
    if (name == "KNN") | (name == "ABOD") | (name == "HBOS"):
        algorithm.fit(X_train)
        y_pred = algorithm.predict(X_test)
        y_pred[y_pred == 1] = 1 #outlier
        y_pred[y_pred == 0] = -1 #normal
        print('---------'+name+'-----------')
        eval_=evaluate(y_test,y_pred)
        print(eval_)
        evaluation_detection(X_test, y_test,y_pred, var1, var2)
    if name == "Local Outlier Factor":
        algorithm.fit(X_train)
        y_pred = -algorithm.fit_predict(X_test) #outlier = 1
        print('---------'+name+'-----------')
        eval_=evaluate(y_test,y_pred)
        print(eval_)
        evaluation_detection(X_test, y_test,y_pred, var1, var2)
    if name == "Deep MLP":
        # scale data here
        if scale == "StandardScaler":
            scaler = StandardScaler()
        else :
            scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)
        
        algorithm.fit(X_train_scaled, X_train_scaled, epochs=config["nb_epoch"],
                      batch_size=config["batch_size"],shuffle=True,validation_split=0.33, verbose=0)
        y_pred = -deep_predict(algorithm,X_test_scaled,config["outlier_prop"],y_test)#outlier = 1
        print('---------'+name+'-----------')
        eval_=evaluate(y_test,y_pred)
        print(eval_)
        evaluation_detection(X_test_scaled, y_test,y_pred, var1, var2)
  
    if name == "Robust covariance" :
        algorithm.fit(X_train)
        y_pred = algorithm.predict(X_test) #outlier = 1
        print('---------'+name+'-----------')
        eval_=evaluate(y_test,y_pred)
        print(eval_)
        evaluation_detection(X_test, y_test,y_pred, var1, var2)

    if name not in algo_particuliers:
        algorithm.fit(X_train)
        y_pred = -algorithm.predict(X_test) #outlier = 1
        print('---------'+name+'-----------')
        eval_=evaluate(y_test,y_pred)
        print(eval_)
        evaluation_detection(X_test, y_test,y_pred, var1, var2)
        
    return y_pred, eval_

