Preprocess Code: spam_outlier_detection.py
Usage:


ANN Code: ann.py
Requires:
  keras
  tensorflow
  pandas
  sklearn
  matplotlib

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
