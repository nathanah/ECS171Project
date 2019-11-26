import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd  # For data handling

import sys

from sklearn.manifold import TSNE
from numpy import dot
from numpy.linalg import norm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from mpl_toolkits.mplot3d import Axes3D as ax


'''
Isolation Forest equations
Anomaly Score: s(x, n) = 2^-[E(h(x)) / c(n)]
Calculates c(n), the average path length of UNSUCCESSFUL search in a Binary Search Tree
n = number of nodes
c(n) = 2H(n-1) - (2(n-1)/n)
H(i) = ln(i) + E -- E = Euler's Constant = 0.5772156649

Isolation Forest -- Randomly splits the data in between max and min value of the dataset and
counts the number of splits it takes to isolate each data point. The data points
that require a number of splits that is under a certain threshold is an outlier.
Takes in the dataset with only the numeric features (8)
Returns an array with 1's normal data points and -1 for outliers
'''
def isolation_forest(df):

    clf = IsolationForest(behaviour='new', max_samples='auto', random_state=42, contamination='auto')
    clf.fit(df)

    outliers = clf.predict(df)

    return outliers


'''
Local Outlier Factor -- Looks at the density of neighbors for each data point, low density of neighbors
                          below a certain threshold means it is an outlier.
Takes in the dataset with only the numeric features
Returns an array with 1's normal data points and -1 for outliers
'''
def LOF(df):

    lof = LocalOutlierFactor(contamination = 'auto')
    outliers = lof.fit_predict(df)

    return outliers


def remove_outliers_lof(df, lof):

    lof_indices = []
    lof_count = 0
    print(df)

    for i in range(len(lof)):
        if lof[i] == -1:
            lof_count += 1
            lof_indices.append(i)

    for outlier in lof_indices:
        df = df.drop([outlier], axis=0)

    df.reset_index(inplace=True, drop=True)

    return df


def remove_outliers_iso(df, iso):

    iso_indices = []
    iso_count = 0

    for i in range(len(iso)):
        if iso[i] == -1:
            iso_count += 1
            iso_indices.append(i)

    for outlier in iso_indices:
        df = df.drop([outlier], axis=0)

    df.reset_index(inplace=True, drop=True)

    return df


def tSNE_spam(df, n_components):

    df_spam = df.loc[df[57] == 1].reset_index(drop=True).drop(columns=[57])
    df_not_spam = df.loc[df[57] == 0].reset_index(drop=True).drop(columns=[57])


    tsne = TSNE(n_components=n_components, random_state=42) # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    np.set_printoptions(suppress=True) # Supress -- doesn't use scientific notation, writes out full float value

    # fit_transform(self, X[, y]) Fit X into an embedded space and return that transformed output.
    Y_spam = tsne.fit_transform(df_spam) # Takes in vector with multiple values, calculates it into an X and Y value -- blackbox
    Y_not = tsne.fit_transform(df_not_spam)

    spam_coords = []
    not_spam_coords = []

    for i in range(n_components):
        spam_coords.append(Y_spam[:, i])
        not_spam_coords.append(Y_not[:, i])

    return spam_coords, not_spam_coords


def tSNE_outliers(df, n_components):
    # Sort outliers and none outliers, then discard the outlier column
    df_outliers     = df.loc[df['58'] == -1].reset_index(drop=True).drop(columns=['58'])
    df_not_outliers = df.loc[df['58'] == 1].reset_index(drop=True).drop(columns=['58'])

    # Make the tSNE model
    tsne = TSNE(n_components=n_components, random_state=42) # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    np.set_printoptions(suppress=True) # Supress -- doesn't use scientific notation, writes out full float value

    # fit_transform(self, X[, y]) Fit X into an embedded space and return that transformed output.
    Y_outliers = tsne.fit_transform(df_outliers) # Takes in vector with multiple values, calculates it into an X and Y value -- blackbox
    Y_not = tsne.fit_transform(df_not_outliers)

    # Initialize empty arrays to store sorted TSNE vectors
    outliers_coords = []
    not_outliers_coords = []

    for i in range(n_components):
        outliers_coords.append(Y_outliers[:, i])
        not_outliers_coords.append(Y_not[:, i])

    return outliers_coords, not_outliers_coords
    
def graph(n_components, target, coords, not_coords, name):
    # Create and plot figures for provided values

    # 3 dimension graph
    if n_components == 3:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coords[0], coords[1], coords[2], s=4, color='red', label=target)
        ax.scatter(not_coords[0], not_coords[1], not_coords[2], s=4, color='blue', label='not '+target)
        ax.legend(loc='upper right')
        ax.set_xlabel('axis 1', fontsize=16)
        ax.set_ylabel('axis 2', fontsize=16)
        ax.set_zlabel('axis 3', fontsize=16)
        # ax.set_title('Method: '+ name)
    # 2 dimension graph
    elif n_components == 2:
        plt.scatter(coords[0], coords[1], s=4, color='red', label = target)
        plt.scatter(not_coords[0], not_coords[1], s=4, color='blue', label='not '+target)
        plt.legend(loc='upper right')
        plt.xlim(coords[0].min()-75, coords[0].max()+75)
        plt.ylim(coords[1].min()-75, coords[1].max()+75)
        # plt.title('Method: '+ name)

    plt.show()

def graph_outliers(df, n_components, lof_bool):
    
    if lof_bool:
        # Classify outliers using Local Outlier Factor
        method = LOF(df.iloc[:, :-1])
        name = 'Local Outlier Factor'

    else:
        # Classify outliers using Isolation Forest
        method = isolation_forest(df.iloc[:, :-1])
        name = 'Isolation Forest'
    
    df_method = pd.DataFrame({'58': method})
    df1 = pd.concat([df, df_method], axis=1)
    df1 = df1.reset_index(drop=True).drop(columns=[57])

    outlier_coords, not_outlier_coords = tSNE_outliers(df1, n_components)
    graph(n_components, 'outliers', outlier_coords, not_outlier_coords, name)


def graph_spam(df, n_components, lof_bool):
    
    # Select outlier method
    if lof_bool:    # Classify outliers using Local Outlier Factor
        lof = LOF(df.iloc[:, :-1])
        df1 = remove_outliers_lof(df, lof)
        df1.to_csv('lof.data', header=False, index=False, float_format='%g')
        name = 'Local Outlier Factor'
    else:           # Classify outliers using Isolation Forest
        iso = isolation_forest(df.iloc[:, :-1])
        df1 = remove_outliers_iso(df, iso)
        df1.to_csv('iso.data', header=False, index=False, float_format='%g')
        name = 'Isolation Forest'

    # get tSNE spam vectors and graph them
    spam_coords, not_spam_coords = tSNE_spam(df1, n_components)
    graph(n_components, 'spam', spam_coords, not_spam_coords, name)


def strip_outliers(df):
    lof = LOF(df.iloc[:, :-1])
    df_lof = remove_outliers_lof(df, lof)
    df_lof.to_csv('spambase_lof.data', header=None, index=False)

    iso = isolation_forest(df.iloc[:, :-1])
    df_iso = remove_outliers_iso(df, iso)
    df_iso.to_csv('spambase_iso.data', header=None, index=False)


def main():

    if (len(sys.argv) == 2 and int(sys.argv[1]) == 2):
        df = pd.read_csv('spambase.data', header=None, delimiter=',')
        strip_outliers(df)

        return


    #Wants arguments in form: spam_outlier_detection.py method graph_type tSNE_dimension
    if(len(sys.argv) != 4):

        print("Parse data set for LOF and ISO outliers: ")
        print("\"python spam_outlier_detection.py 2\"")
        print()
        print()
        print("Graph outlier methods onto tSNE reduced graphs: ")
        print("Options for method: ")
        print("  0: Isolation Forest")
        print("  1: Local Outlier Factor")
        print()
        print("Options for type of graph:")
        print("  0: graph spam vs non-spam")
        print("  1: graph outliers vs non-outliers")
        print()
        print("Options for tSNE dimension: ")
        print("  2, 3")
        print()
        print("ex:  \"python spam_outlier_detection.py 0 1 3\"\n  will use Isolation Forest to graph outliers in 3 dimensions")

        return

    df = pd.read_csv('spambase.data', header=None, delimiter=',')

    if (int(sys.argv[2])):
        graph_outliers(df, int(sys.argv[3]), int(sys.argv[1]))
    else:
        graph_spam(df, int(sys.argv[3]), int(sys.argv[1]))


if __name__ == '__main__':
    main()
