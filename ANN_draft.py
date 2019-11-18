#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import callbacks
#from tensorflow.keras.utils import np_utils
from tensorflow.keras import optimizers
from sklearn.preprocessing import MinMaxScaler



def load_data():
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
    #shuffle data
    data = data.sample(frac=1).reset_index(drop=True)


    scaler = MinMaxScaler()
    scaler.fit(data)
    data.iloc[:] = scaler.transform(data)
    return data

def construct_model(num_nodes, num_layers, learning_rate, loss_fn, activation_fn, output_fn):
    ann = Sequential()
    ann.add(Dense(units = num_nodes), input_shape=(57,), activation = activation_fn)
    for l in range(1,num_layers):
        ann.add(Dense(units = num_nodes), activation = activation_fn) #hidden layers
    ann.add(Dense(units =1),activation = output_fn) #output layer
    
    earlyStop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=100, mode='min', baseline=None, restore_best_weights=False)
    
    sgd = optimizers.SGD(lr= learning_rate, decay=1e-6, momentum=0.9, nesterov=False)
           
    ann.compile(loss= loss_fn,
                    optimizer= sgd ,
                    metrics=["binary_accuracy"])
    return ann
    

def Main():
    data = load_data()  #returns min-max scaled data
    


# In[ ]:




