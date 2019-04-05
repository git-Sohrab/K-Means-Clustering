# Import Dependecies
from array import *
from numpy import *
import math as mth
import numpy as np
import matplotlib.pyplot as plt
import random as rand
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Step 1 -- Plot data
# read the features and labels from file
input_data = open("input_data", "r")
output_data = open("output_data", "r")
# parse the input stream by newline characters
X = input_data.read().split('\n') 
Y = output_data.read().split('\n')
X.remove('')
Y.remove('')
# Our parsed and ready data
n_X = [float(i) for i in X]
n_Y = [float(i) for i in Y]

#plt.scatter(n_X, n_Y)
#plt.xlabel('Xi')
#plt.ylabel('Ti')
#plt.show()

# Step 2 -- ERM
#for i in range(1, 21):
#    coeffs = polyfit(n_X, n_Y, i)
#    ffit = poly1d(coeffs)
    # calculate new x's and y's
#    y_new = ffit(n_X)
#    s = 0
#    for j in range(1, len(n_X)):
#        s += (y_new[j] - n_Y[j])**2
#    s = s/2
#    plt.scatter(i, s)

#plt.xlabel("W ( Degree of polynomial )")
#plt.ylabel("L ( Empirical Loss )")
#plt.show()

# Step 3 -- RLM
#coeffs = polyfit(n_X, n_Y, 20)
#ffit = poly1d(coeffs)
# calculate new x's and y's
#y_new = ffit(n_X)

#euc_sum = 0
#for i in n_X:
#    euc_sum += i

#for lamb in range(1, 21+1):
#    s = 0
#    for j in range(1, len(n_X)):
#        s += (y_new[j] - n_Y[j])**2 + mth.exp(-1*lamb)*(euc_sum)**2
#    s = s/2
#    plt.scatter(lamb, s)

#plt.xlabel("i")
#lt.ylabel("L ( Regularized Loss of Polynomial of degree 20 )")
#plt.show()

# Step 4 -- Crossvalidation

data = zip(n_X, n_Y)
rand.shuffle(data)
data = zip(*[iter(data)]*10)
test_set = data[9]
test_X = zip(*test_set)[0]
test_Y = zip(*test_set)[1]
train_set = data[:9]
train_X = []
train_Y = []

for i in train_set:
        train_X.append(zip(*i)[0])
        train_Y.append(zip(*i)[1])
train_X = list(train_X)
train_Y = list(train_Y)

for a ,b in zip(train_X, train_Y):
    s = 0
    for i in range(1, 10):
        coeffs = polyfit(list(a), list(b), i)
        ffit = poly1d(coeffs)
        # calculate new x's and y's
        y_new = ffit(list(a))
        for j in range(0, 10):
            s += (y_new[j] - list(test_Y)[j])**2
        s = s/2


