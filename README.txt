Preprocess Code: spam_outlier_detection.py
Usage:
  In main() function, to graph outliers vs non-outliers, set variable outlier to True, False to see spam vs
  non-spam data. When running with outlier=False, the data set will save as a new Data Frame with outliers
  removed: lof.data for Local Outlier Factor results and iso.data for Isolation Forest results. Setting
  n-components will change how many dimensions the graphs will have, either 2 for a 2-D graph or 3 for a
  3-D graph.

  To get training and testing graphs:
  
    For ann.py, comment out main() and uncomment testing() at the bottom of the code. Then, in load_data() function,
    replace file = 'spambase.data' with file = 'lof.data' for Local Outlier Factor outliers removed, and
    file = 'iso.data' for Isolation Forest outliers removed. in testing() function, you can change the activation
    function (activation_fn variable) to 'sigmoid', 'linear', 'leaky reLU', and 'tanh', where the output function
    (output_fn variable) should be 'sigmoid'.


ANN Code: ann.py
Requires:
  keras
  tensorflow
  pandas
  sklearn
  matplotlib

Usage:
  To perform a grid search over the number of nodes per layer and number of hidden layers, change max_hidden_layers in main() to one of the options described below. Set the learning rate by changing the lr variable in main and set the value of k for k-fold validation by changing k (again, options below). Setting the activation function, output layer activation function, and loss function is achieved by changing activation_fn, output_fn, and loss_fn respectively. Comment out the call to testing() at the end of the file and call main(). At the end of each fold, the training and testing misclassification error matrices will be printed, where the row corresponds to the number of hidden layers and the column corresponds to the number of nodes per layer.
  
  To generate the generalized error, ROC, and PR plots, comment out the call to main() at the end of the file and call testing(). In testing(), set the number of hidden layers and nodes per layer by changing hidden_layers and nodes_per_layer, change the learning rate by setting lr, and set the activation, output and loss functions by changing activation_fn, output_fn, and loss_fn.
  

  Options for activation/output function:
  "sigmoid"
  "linear"
  layers.LeakyReLU(alpha=0.01)
  "tanh"

  Options for number of hidden layers:
    Any nonzero positive integer

  Options for number of nodes per layer:
    Any nonzero positive integer (or list of positive integers if setting in main)

  Options for learning rate:
    Any nonzero positive number

   Options for loss function:
   "binary cross-entropy"
   "mse"


Bayes Code: bayes.py
Requires:
  numpy

Usage:
  python bayes.py filepath method transformation

  Options for method:
  0: assume gaussian distribution
  1: discrete variable calculation

  Options for transformation:
  <=0: no transformation of data
  1: transform data using a mean-bucketed method
  >=2: transform data using specified number of quantiles

  ex:
  "python bayes.py spambase.data 1 3"
  will use 3 buckets and calculate likelihood using a discrete function

Output:
  Prints TP/FP/TN/FN counts and misclassification for each batch of training and testing over 10 folds
  Also prints the average training and testing misclassification at the end of training as well as min testing misclassification.

Bayes graphing: bayes_graphing.py
Requires:
  matplotlib

Usages:
  python bayes_graphing.py filepath
