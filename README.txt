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
Usage:
  put usage here

Evaluation:
  The misclassification rate per epoch, Receiver Operating Characteristic (ROC) curve, and Precision-Recall (PR) curve give     insight to a model’s accuracy and efficiency. These three plots are created for any model with the error_plot function.       Figures ___ evaluate models     with parameters determined by grid search.
  Enter the following model parameters in the Training method:

  Options for activation function:
  "sigmoid"
  "linear"
  "leaky reLU"
  "tanh"

  Options for output function:
  "sigmoid"
  "linear"

  Options for number of hidden layers:
    Any nonzero positive integer

  Options for number of nodes per layer:
    Any nonzero positive integer

  Options for learning rate:
    Any nonzero positive number

   Options for loss function:
   "binary cross-entropy"
   "mse"


Bayes Code: bayes.py
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
  Prints TP/FP/TN/FN counts and accuracy for each batch of training and testing over 10 folds
  Also prints the average training and testing accuracy at the end of training as well as max testing accuracy.
