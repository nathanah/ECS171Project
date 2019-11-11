
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras import optimizers


#import data
file = "spambase.data"
names = [
        "word_freq_make",
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
        "spamClassification"
        ]
data = pd.read_csv(file, delimiter=",", names=names)

x = data.iloc[:,data.columns != "spamClassification"]
y = data.loc[:,"spamClassification"]




#grid search
maxLayers = 3
gridNodes = [3,6,10,15,30]
trainingAccuracy = np.zeros((maxLayers,len(gridNodes)))
testingAccuracy = np.zeros((maxLayers,len(gridNodes)))

for layers in range(1,maxLayers+1):
    print("Started Layer: %s" % layers)
    for nodes in range(0,len(gridNodes)):
        print("Started Nodes: %s" % gridNodes[nodes])
        #set up ann
        ann = Sequential()
        ann.add(Dense(units=gridNodes[nodes], input_shape=(57,), activation="sigmoid")) #hidden layers
        for l in range(1,layers):
            ann.add(Dense(units=gridNodes[nodes], activation="sigmoid"))
        ann.add(Dense(units=1, activation="sigmoid")) #output layer


        earlyStop = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=100, mode='min', baseline=None, restore_best_weights=False)

        sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
        ann.compile(loss="mean_squared_error",
                    optimizer=sgd,
                    metrics=["binary_accuracy"])


        info = ann.fit(x,y,
                        validation_split=.34,
                        shuffle=True,
                        epochs=1000,
                        batch_size=32,
                        verbose=0,
                        callbacks=[earlyStop])
        trainingAccuracy[layers-1][nodes] = info.history['binary_accuracy'][len(info.history)-1]
        testingAccuracy[layers-1][nodes] = info.history['val_binary_accuracy'][len(info.history)-1]




print("Training Accuracy")
print(trainingAccuracy)
print("Testing Accuracy")
print(testingAccuracy)
