import matplotlib.pyplot as plt
# %matplotlib notebook
import numpy as np
# import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency

import spacy  # For preprocessing
from gensim.models import Word2Vec
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
from sklearn.manifold import TSNE
from numpy import dot
from numpy.linalg import norm
from sklearn.neighbors import LocalOutlierFactor



from mpl_toolkits.mplot3d import Axes3D as ax

def LOF(df):
    lof = LocalOutlierFactor(contamination = 0.1)
    outliers = lof.fit_predict(df)

    return outliers

def remove_outliers(df, lof):

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

df = pd.read_csv('spambase.data', header=None, delimiter=',')

# df1 = df.drop(columns=[57])

lof = LOF(df.iloc[:, :-1])

df1 = remove_outliers(df, lof)

# print(df1.head())

df_spam = df1.loc[df[57] == 1].reset_index(drop=True).drop(columns=[57])
df_not_spam = df1.loc[df[57] == 0].reset_index(drop=True).drop(columns=[57])

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


# print(df.head(5))


## Graph predictions vs data points with tsne for visuals
