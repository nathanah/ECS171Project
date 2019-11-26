# Preprocessing Code: spam_outlier_detection.py
## Requires:
  * numpy
  * pandas
  * sklearn
  * matplotlib
  
## Usage:
### Option to produce .data files without outliers:
Run the code with 2 as a commandline argument:
```
python spam_outlier_detection.py 2
```

> Output: spambase_iso.data, spambase_lof.data
  
### Option to produce tSNE graphs
Run the code with {0,1} {0,1} {2,3} as the commandline arguments:
```
python spam_outlier_detection.py (1) (2) (3)
```

(1): Options for the outlier method:
  * 0: Isolation Forest 
  * 1: Local Outlier Factor

(2): Options for graph content:
  * 0: graph spam vs non-spam
  * 1: graph outliers vs non-outliers
  
(3): Options for the tSNE dimension:
  * 2: two-dimensional tSNE graph
  * 3: tree-dimensional tSNE graph

> Output: 2-d or 3-d tSNE graph.
  

# ANN Code: ann.py
## Requires:
  * keras
  * tensorflow
  * pandas
  * sklearn
  * matplotlib

## Set up:
  To perform a grid search over the number of nodes per layer and number of hidden layers, change max_hidden_layers in
  main() to one of the options described below. Set the learning rate by changing the lr variable in main and set the 
  value of k for k-fold validation by changing k (again, options below). Setting the activation function, output layer 
  activation function, and loss function is achieved by changing activation_fn, output_fn, and loss_fn respectively. 
  Comment out the call to testing() at the end of the file and call main(). At the end of each fold, the training and 
  testing misclassification error matrices will be printed, where the row corresponds to the number of hidden layers 
  and the column corresponds to the number of nodes per layer.
  
  To generate the generalized error, ROC, and PR plots, comment out the call to main() at the end of the file and call
  testing(). In testing(), set the number of hidden layers and nodes per layer by changing hidden_layers and nodes_per_layer, 
  change the learning rate by setting lr, and set the activation, output and loss functions by changing activation_fn, 
  output_fn, and loss_fn.
  
 ## Usage: python3 ann.py

  ### Options for activation/output function:
  * "sigmoid"
  * "linear"
  * layers.LeakyReLU(alpha=0.01)
  * "tanh"

  ### Options for number of hidden layers:
   * Any nonzero positive integer

  ### Options for number of nodes per layer:
   * Any nonzero positive integer (or list of positive integers if setting in main)

  ### Options for learning rate:
   * Any nonzero positive number

  ### Options for loss function:
   * "binary cross-entropy"
   * "mse"


# Bayes Code: bayes.py
## Requires:
  * numpy

## Usage:
  python bayes.py filepath method transformation

  ### Options for method:
   * 0: assume gaussian distribution
   * 1: discrete variable calculation

  ### Options for transformation:
   * <=0: no transformation of data
   * 1: transform data using a mean-bucketed method
   * \>=2: transform data using specified number of quantiles

  ### ex:
  "python bayes.py spambase.data 1 3"
  will use 3 buckets and calculate likelihood using a discrete function

## Output:
  Prints TP/FP/TN/FN counts and misclassification for each batch of training and testing over 10 folds
  Also prints the average training and testing misclassification at the end of training as well as min testing misclassification.

# Bayes graphing: bayes_graphing.py
## Requires:
   * matplotlib

## Usages:
  python bayes_graphing.py filepath
