import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd  # For data handling

from sklearn.manifold import TSNE
from numpy import dot
from numpy.linalg import norm
from sklearn.neighbors import LocalOutlierFactor
# from sklearn.ensemble import IsolationForest

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
    clf.fit(x)

    outliers = clf.predict(x)

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

    for i in range(len(lof)):
        if lof[i] == -1:
            lof_count += 1
            lof_indices.append(i)

    for outlier in lof_indices:
        df = df.drop([outlier], axis=0)

    df.reset_index(inplace=True, drop=True)

    # print("num outliers: " + str(lof_count))

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

    # print("num outliers: " + str(iso_count))

    return df

def main():
    df = pd.read_csv('spambase.data', header=None, delimiter=',')

    # df_lof = df.drop(columns=[57])

    lof = LOF(df.iloc[:, :-1])
    iso = isolation_forest(df.iloc[:, :-1])

    df_lof = remove_outliers_lof(df, lof)
    df_iso = remove_outliers_iso(df, lof)

    # print(df_lof.head())

    df_spam = df_lof.loc[df[57] == 1].reset_index(drop=True).drop(columns=[57])
    df_not_spam = df_lof.loc[df[57] == 0].reset_index(drop=True).drop(columns=[57])

    # print(df_not_spam.head(5))

    tsne = TSNE(n_components=3, random_state=0) # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    np.set_printoptions(suppress=True) # Supress -- doesn't use scientific notation, writes out full float value
    # fit_transform(self, X[, y]) Fit X into an embedded space and return that transformed output.
    Y_spam = tsne.fit_transform(df_spam) # Takes in vector with multiple values, calculates it into an X and Y value -- blackbox
    Y_not = tsne.fit_transform(df_not_spam)

    x_coords_spam = Y_spam[:, 0]
    y_coords_spam = Y_spam[:, 1]
    z_coords_spam = Y_spam[:, 2]

    x_coords_not = Y_not[:, 0]
    y_coords_not = Y_not[:, 1]
    z_coords_not = Y_not[:, 2]

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_coords_spam, y_coords_spam, z_coords_spam)
    ax.scatter(x_coords_not, y_coords_not, z_coords_not)

    # plt.scatter(x_coords_spam, y_coords_spam, color='red')
    # plt.scatter(x_coords_not, y_coords_not, color='green')

    # plt.xlim(x_coords_spam.min()+50, x_coords_spam.max()+50)
    # plt.ylim(y_coords_spam.min()+50, y_coords_spam.max()+50)
    plt.show()


if __name__ == '__main__':
    main()




## Graph predictions vs data points with tsne for visuals
