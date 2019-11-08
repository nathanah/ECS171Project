
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.svm import OneClassSVM
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras import optimizers


#import data
file = "processed.data" #?????????????????
names = ["word_freq_make",
        "word_freq_address",
        "word_freq_all",
        "word_freq_3d",
        "word_freq_our",
        "word_freq_over",
        "word_freq_remove",
        "word_freq_internet",
        "word_freq_order",
        "word_freq_mail",
        "word_freq_receive",
        "word_freq_will",
        "word_freq_people",
        "word_freq_report",
        "word_freq_addresses",
        "word_freq_free",
        "word_freq_business",
        "word_freq_email",
        "word_freq_you",
        "word_freq_credit",
        "word_freq_your",
        "word_freq_font",
        "word_freq_000",
        "word_freq_money",
        "word_freq_hp",
        "word_freq_hpl",
        "word_freq_george",
        "word_freq_650",
        "word_freq_lab",
        "word_freq_labs",
        "word_freq_telnet",
        "word_freq_857",
        "word_freq_data",
        "word_freq_415",
        "word_freq_85",
        "word_freq_technology",
        "word_freq_1999",
        "word_freq_parts",
        "word_freq_pm",
        "word_freq_direct",
        "word_freq_cs",
        "word_freq_meeting",
        "word_freq_original",
        "word_freq_project",
        "word_freq_re",
        "word_freq_edu",
        "word_freq_table",
        "word_freq_conference",
        "char_freq_;",
        "char_freq_(",
        "char_freq_[",
        "char_freq_!",
        "char_freq_$",
        "char_freq_#",
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
        ]
data = pd.read_csv(file, delim=",", names=names)






#grid search
trainingAccuracy = np.zeros((3,4))
testingAccuracy = np.zeros((3,4))
gridNodes = [3,6,9,12]
for layers in range(1,4):
    for nodes in range(0,len(gridNodes)):
        #set up ann
        ann = Sequential()
        ann.add(Dense(units=gridNodes[nodes], input_shape=(57,), activation="relu")) #hidden layers
        for l in range(1,layers):
            ann.add(Dense(units=gridNodes[nodes], activation="relu"))
        ann.add(Dense(units=1, activation="softmax")) #output layer


        earlyStop = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=100, mode='min', baseline=None, restore_best_weights=False)

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
        ann.compile(loss="categorical_crossentropy",
                    optimizer=sgd,
                    metrics=["accuracy"])


        info = ann.fit(x,yCat,
                        validation_split=.34,
                        shuffle=False,
                        epochs=10000,
                        batch_size=32,
                        callbacks=[earlyStop])
