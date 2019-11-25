import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import callbacks, optimizers
from sklearn.preprocessing import MinMaxScaler

from matplotlib import axes
from matplotlib import pyplot as plot
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc


def error_plot(X, y, model, numEpochs = 1000):


    #callback to collect testing and training accuracy at end of each epoch
    class accuracyHistory(callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.accuracies = []
            self.val_accuracies = []
        def on_epoch_end(self, epoch, logs={}):
            self.accuracies.append(logs.get('binary_accuracy'))
            self.val_accuracies.append(logs.get('val_binary_accuracy'))

    acc_Callback = accuracyHistory()
    acc_matrix = np.zeros([10, numEpochs])
    val_acc_matrix = np.zeros([10, numEpochs])

    kf = StratifiedKFold(n_splits=10)
    idx = 0
    for train_index, test_index in kf.split(X = X, y = y):
        training_x = np.asarray(X)[train_index,:]
        testing_x = np.asarray(X)[test_index,:]
        training_y = np.asarray(y)[train_index]
        testing_y = np.asarray(y)[test_index]
        #Fit model with callback
        info = model.fit(training_x, training_y, batch_size=32, validation_data = [testing_x,testing_y], epochs = numEpochs, callbacks=[acc_Callback], verbose = 0)
        #Make plot of training and testing error
        acc_matrix[idx,:] = np.asarray(acc_Callback.accuracies).astype(float)
        val_acc_matrix[idx,:] = np.asarray(acc_Callback.val_accuracies).astype(float)
        idx = idx + 1

    gen_acc = acc_matrix.mean(axis = 0)
    gen_val_acc = val_acc_matrix.mean(axis = 0)

    epochs = np.arange(numEpochs) + 1
    plot.figure(1)
    plot.plot(epochs, (1 - gen_acc), 'r')
    plot.plot(epochs, (1 - gen_val_acc), 'b')
    plot.legend({'Training','Testing'},fontsize = 12)
    plot.xlabel('Epochs',fontsize = 16), plot.ylabel('Misclassification Rate',fontsize = 16)
    plot.xlim([1,numEpochs]), plot.ylim([0, (np.amax(1 - gen_val_acc) + 0.1)])
    plot.title('Generalized Error',fontsize = 20)

def ROC_and_PR_plots(X,y, model, numEpochs = 1000):
    earlyStop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=100, mode='min', baseline=None, restore_best_weights=False)

    kf = KFold(n_splits=10)
    idx = 0
    training_ROC_AUC = []
    testing_ROC_AUC = []
    training_PR_AUC = []
    testing_PR_AUC = []

    for train_index, test_index in kf.split(X = X, y = y):
        training_x = np.asarray(X)[train_index,:]
        testing_x = np.asarray(X)[test_index,:]
        training_y = np.asarray(y)[train_index]
        testing_y = np.asarray(y)[test_index]
        val_data = [testing_x, testing_y]
        model.fit(training_x, training_y,batch_size=32, validation_data = val_data, epochs = numEpochs, verbose = 0,callbacks=[earlyStop])

        #False/true positive rates for ROC
        training_y_pred = model.predict(training_x).ravel()
        testing_y_pred = model.predict(testing_x).ravel()

        training_false_pos_rate, training_true_pos_rate, training_thresholds = roc_curve(training_y, training_y_pred)
        testing_false_pos_rate, testing_true_pos_rate, testing_thresholds = roc_curve(testing_y, testing_y_pred)
        training_ROC_AUC.append(roc_auc_score(training_y, training_y_pred))
        testing_ROC_AUC.append(roc_auc_score(testing_y, testing_y_pred))

        #Precision and recall
        training_y_proba = model.predict_proba(training_x).ravel()
        testing_y_proba = model.predict_proba(testing_x).ravel()

        training_precision, training_recall, training_thresholds_PR = precision_recall_curve(training_y, training_y_proba)
        testing_precision, testing_recall, training_thresholds_PR = precision_recall_curve(testing_y, testing_y_proba)

        training_PR_AUC.append(auc(training_recall, training_precision))
        testing_PR_AUC.append(auc(testing_recall, testing_precision))

        #plot labels
        labelTr = "_nolegend_"
        labelTe = "_nolegend_"
        if idx == 0:
            labelTr = "Training"
            labelTe = "Testing"
        idx = idx + 1

        #Precision-Recall Plot
        plot.figure(3)
        plot.plot(training_recall, training_precision, 'r', label = labelTr, linewidth = 0.4)
        plot.plot(testing_recall,testing_precision, 'b', label = labelTe,linewidth = 0.4)
        plot.ylabel('precision',fontsize = 16), plot.xlabel('recall',fontsize = 16)
        plot.title('PR curve',fontsize = 20)
        plot.legend(fontsize = 12, loc='best')
        plot.xlim([0,1]), plot.ylim([0,1])

        #ROC plot
        plot.figure(2)
        plot.plot([0,1],[0,1],'k--',label= '_nolegend_')
        plot.plot(training_false_pos_rate, training_true_pos_rate, 'r',label = labelTr, linewidth = 0.4)
        plot.plot(testing_false_pos_rate, testing_true_pos_rate, 'b',label = labelTe ,linewidth = 0.4)
        plot.xlabel('false positive rate',fontsize = 16), plot.ylabel('true positive rate',fontsize = 16)
        plot.title('ROC curve',fontsize = 20)
        plot.legend(fontsize = 12, loc='best')
        plot.xlim([0,1]), plot.ylim([0,1])


    plot.figure(2)
    ROC_val_Tr = str(np.round(np.asarray(training_ROC_AUC).mean(0),3))
    AUC_text_Tr = 'Training AUC: ' + ROC_val_Tr
    plot.text(0.6, 0.45, AUC_text_Tr ,fontsize = 12)

    ROC_val_Te = str(np.round(np.asarray(testing_ROC_AUC).mean(0),3))
    AUC_text_Te = 'Testing AUC: ' + ROC_val_Te
    plot.text(0.6, 0.35, AUC_text_Te ,fontsize = 12)

    plot.figure(3)
    PR_val_Tr = str(np.round(np.asarray(training_PR_AUC).mean(0),3))
    AUC_text_Tr = 'Training AUC: ' + PR_val_Tr
    plot.text(0.05, 0.45, AUC_text_Tr ,fontsize = 12)

    PR_val_Te = str(np.round(np.asarray(testing_PR_AUC).mean(0),3))
    AUC_text_Te = 'Testing AUC: ' + PR_val_Te
    plot.text(0.05, 0.35, AUC_text_Te ,fontsize = 12)

'''
Loads the data from ./spambase.data, shuffles it to eliminate any ordering that 
may exist, and returns the min-max normalized version of it
'''
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

'''
Compiles and returns a model with the hyperparameters passed
@num_nodes: the number of nodes per hidden layer
@num_layers: the number of hidden layers
@learning_rate: learning rate to use for model
@loss_fn: loss function to use for model
@activation_fn: activation function to use for the hidden layers. Can be a string or a predefined function
@output_fn: activation function to use for the final layer. Can be a string or a predifined function
'''
def construct_model(num_nodes, num_layers, learning_rate, loss_fn, activation_fn, output_fn):
    ann = Sequential()
    ann.add(Dense(units=num_nodes, input_shape=(57,), activation=activation_fn))

    for l in range(1,num_layers):
        ann.add(Dense(units=num_nodes, activation=activation_fn)) #hidden layers

    ann.add(Dense(units=1, activation=output_fn)) #output layer

    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=False)

    ann.compile(loss=loss_fn,
                optimizer= sgd,
                metrics=["binary_accuracy"])
    return ann

'''
Performs a grid search over the number of hidden nodes per layer and number of hidden layers
for a model with a given activation function, learning rate, and output function.
Returns a training error matrix and a testing error matrix with the misclassification 
errors for each combination of hidden layers and nodes per layer.
@training_data: data to train the model with
@testing_data: data to test the model with
@max_hidden_layers: maximum number of hidden layers to test. The function will test values
from 1 to max_hidden_layers
@nodes_per_layer: list containing different values of nodes per layer to test
@learning_rate: learning rate the model should use
@loss_fn: loss function the model should use
@activation_fn: activation function the model should use for the hidden layers
@output_fn: activation function the model should use for the output layer
'''
def grid_search(training_data, testing_data, max_hidden_layers, nodes_per_layer, learning_rate, loss_fn, activation_fn, output_fn):
    training_error = np.zeros((max_hidden_layers, len(nodes_per_layer)))
    testing_error = np.zeros((max_hidden_layers,len(nodes_per_layer)))
    earlyStop = callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=180, mode='min', baseline=None, restore_best_weights=False)
    training_x = training_data[:, :-1]
    testing_x = testing_data[:, :-1]
    training_y = training_data[:, -1]
    testing_y = testing_data[:, -1]

    for layer in range(max_hidden_layers):
        for i, nodes in enumerate(nodes_per_layer):
            model = construct_model(nodes, layer+1, learning_rate, loss_fn, activation_fn, output_fn)
            logs = model.fit(training_x, training_y,
                            validation_data=(testing_x, testing_y),
                            shuffle=True,
                            batch_size=32,
                            verbose=0,
                            epochs=1500,
                            callbacks=[earlyStop])
            training_error[layer][i] = 1-logs.history['binary_accuracy'][-1]
            testing_error[layer][i] = 1-logs.history['val_binary_accuracy'][-1]

    return (training_error, testing_error)

'''
Returns the training and testing sets for a given number of folds given which iteration
of testing we're on.
@data: the whole dataset
@test_index: index of the fold that should be used for testing. Must be between 0 and k
@k: the number of folds
'''
def k_fold(data, test_index, k):
    fold_size = len(data)//k
    testing_data = np.asarray(data[test_index * fold_size:(test_index + 1) * fold_size])
    training_data = np.concatenate([data[:test_index * fold_size], data[(test_index + 1) * fold_size:]])

    return (training_data, testing_data)

def main():
    data = load_data()  #returns min-max scaled data
    max_hidden_layers = 3
    nodes_per_layer = [3, 6, 10, 15, 30, 50, 100]
    activation_fn = "sigmoid"
    output_fn = "sigmoid"
    loss_fn = "mean_squared_error"
    lr = 0.005
    k = 10

    training_errors = np.zeros((k, max_hidden_layers, len(nodes_per_layer)))
    testing_errors = np.zeros((k, max_hidden_layers, len(nodes_per_layer)))

    for i in range(k):
        (training_data, testing_data) = k_fold(data, i, k)
        (training_error, testing_error) = grid_search(training_data, testing_data, max_hidden_layers, nodes_per_layer, lr, loss_fn, activation_fn, output_fn)
        training_errors[i] = training_error
        testing_errors[i] = testing_error
        print("\nTraining error for k=", i, ":\n", training_error)
        print("\nTesting error for k=", i, ":\n", testing_error)
        print("------------------------------------------------------\n")

    print("\nGeneralized training errors:\n", training_errors.mean(0))
    print("\nGeneralized testing errors:\n", testing_errors.mean(0))

def testing():
    data = load_data()  #returns min-max scaled data
    hidden_layers = 3
    nodes_per_layer = 10
    activation_fn = "sigmoid"
    output_fn = "sigmoid"
    loss_fn = "mean_squared_error"
    lr = 0.005
    k = 10

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    model = construct_model(nodes_per_layer, hidden_layers, lr, loss_fn, activation_fn, output_fn)
    error_plot(x, y, model)
    ROC_and_PR_plots(x, y, model)

#main()
testing()
