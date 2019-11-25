Preprocess Code: spam_outlier_detection.py
Usage:


ANN Code: ann.py
Usage:
  put usage here


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
  Also prints the average testing accuracy at the end of training as well as max testing accuracy.
