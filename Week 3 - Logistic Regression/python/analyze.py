import pandas as pd
#import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt


## Convert a set of two features into a much larger set using
#   a higher order polynomial function that can create a non-linear
#   decision boundary
#
def mapFunction(x1, x2):
    # Define the degree of the polynomial
    degree = 6

    s = x1.shape    # creates an array of dimensions

    ## The first column of the new matrix is all 1's, so start with
    #   that
    out = np.ones(s[0])

    ## Now we need to add on a new column of values for each term
    #   in the polynomial
    #
    for i in range(1,degree+1):
        for j in range(i+1):
            ## Create a new vector that contains the values
            #
            v = np.power(x1, i-j) * np.power(x2, j)
            print v

    return out



df = pd.read_csv("ex2data2.txt", header=None);

## Pull in original data set
#
df.columns = ["test1", "test2", "pass"]

#groups = df.groupby('pass')

## Now build up the plot
#fig, ax = plt.subplots()
#ax.margins(0.05)
#for name, group in groups:
#    ax.plot(group['test1'], group['test2'], marker='o', linestyle='', ms=12, label = name)

#ax.legend()

#plt.show()

## Map it into a new data set that incorporates the higher order polynomial
#   we want to use
#
X = np.array(df.as_matrix())

## Call our map function by slicing out the first two columns of the 
#   feature array and passing them as vectors
#
out = mapFunction(X[:,0], X[:,1])

#train_cols = df.columns[[0, 1]]

#print train_cols

#logit = sm.Logit(df["pass"], df[train_cols])

#result = logit.fit()

#print result.summary()



