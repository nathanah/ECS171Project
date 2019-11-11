import pandas as pd
import numpy as np
import random


def readFile():
    #import data
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
    data = np.array(data)

    #Bucket data:
    bucket(2, data)

    #Shuffle data:
    random.shuffle(data)

    x = data[:,:57]
    y = data[:,57]

    return [x, y]


"""
    Bayes nets usually work on discrete values, not continuous variables.
    This function transforms the continuous values of our data into discrete buckets with labels.

    Params:
        buckets: the number of buckets we are dividing the data into (quantiles)
        data: the entire set of data before it has been split into X and Y

    Side-effect:
        data is transformed and no longer the same as before it was passed in
"""
def bucket(buckets, data):

    #Go through each column except for the label
    for i in range(57):

        #Sort the data on the given column
        data = data[np.argsort(data[:, i])]

        #Variable to store the lower bound of the current quantile
        lower = 0

        #Go through each quantile labeling it with the proper bucket
        for j in range(buckets):

            #Get the upper cutoff of the current quantile.
            upper = (j+1) * (len(data) / buckets)
            upper = int(upper)

            #Label all samples in range with current bucket
            data[lower:upper, i] = j

            #Upper cutoff becomes lower bound.
            lower = upper

"""
    Prior is defined as the probability of any sample being in the class.
    We can calculate this by just counting up the number of samples that are spam.
    P(not spam) = (1 - P(spam))

    Params:
        y: the labels for all of our data

    Returns:
        Probability of any sample being spam
"""
def prior(y):

    spam = 0

    for i in range(len(y)):
        if y[i] == 1:
            spam += 1

    return spam / len(y)

"""
    Likelihood is the probability P(xj | spam/not-spam)
    We can calculate this by looking looking at the number of elements in that class that have the certain feature we are looking for.

    That is: (count of v in c) / (total number of samples in c)

    Params:
        x: our set of all samples X
        y: our set of all labels Y
        j: the column we are looking at
        v: the value we are looking for in the column
        c: what class we are looking at

    Returns:
        P(xj | spam/not-spam)
"""
def likelihood(x, y, j, v, c):

    #Count of occurences of that value
    count = 0

    #Total count of that class
    total = 0

    #Iterate through each sample
    for i in range(len(x)):

        #If the classes are the same update total number of that class
        if y[i] == c :
            total +=1

            #If the values match update our count of value
            if(x[i][j] == v):
                count += 1
    return count / total


"""
    Posterior is the probability P(Spam | x). We calculate this with Bayes theroem. And by doing a trick of P(Spam | X) / P(!Spam | X) and seeing if it is >= 1.

    Params:
        x: our set of all samples X
        y: our set of all labels Y
        i: the sample we are looking at
"""

def posterior(x, y, i):

    #Get the probability that it is spam
    numerator = 1
    #Go through each feature and calculate likelihood, multiplying it
    for j in range(57):
        v = x[i][j]
        numerator *= likelihood(x, y, j, v, 1)

    #Get the probability that it is not spam
    denominator = 1
    #Go through each feature and calculate likelihood, multiplying it
    for j in range(57):
        v = x[i][j]
        denominator *= likelihood(x, y, v, 0)

    return numerator / denominator


def main():

    #Read in data and preprocess it
    data = readFile()
    x = data[0]
    y = data[1]

    #Portion of samples to be used for training:
    split = .7
    cutoff = int(split * len(x))

    #Split training and Testing sets
    trainingX = x[:cutoff]
    trainingY = y[:cutoff]
    testingX = x[cutoff:]
    testingY = y[cutoff:]

main()
