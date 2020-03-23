import numpy as np
import csv
import math
import matplotlib.pyplot as plt

def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))

def cost_function(theta, X, y):
    m = len(y)
    z = np.dot(X, theta)
    temp1 = np.multiply(y, np.log(sigmoid_function(z)))
    temp2 = np.multiply(1-y, np.log(1-sigmoid_function(z)))
    return np.sum(temp1 + temp2) / (-m)

def gradientDescent(theta, X, y, lr):
    m = len(y)
    temp = sigmoid_function(np.dot(X, theta.T)) - y.T
    temp = np.dot(temp, X) / m 
    theta = theta - temp * lr
    return theta

def vectorToMatrix(vector):
    matrix = np.zeros((28,28), dtype = float)
    for i in range(0,28):
            for j in range(0,28):
                        matrix[i][j] = vector[i*28 + j]
    return matrix

labels = np.zeros((8817), dtype = int)
X = np.zeros((8817, 784), dtype = float)
j = 0
with open('train.csv') as csvfile:
    readCSV = csv.reader(csvfile)
    i = 0
    for row in readCSV:
        if i > 0 and (row[0] == '0' or row[0] == '1'):
            labels[j] = row[0]
            X[j] = row[1:]
            j = j + 1
        i = i + 1
(m,n) = X.shape
#theta = np.zeros((10,n))
#k = 10
y = labels

for i in range(0,8817):
    X[i] = X[i] / 255.0

## splitting data in train and test (70% train, 30% testing)
training_data = math.floor(0.9 * 8817)
testing_data = 8817 - training_data

print("Total training data: ", training_data)
print("Total test data: ", testing_data)

X_training = np.zeros((training_data, 784), dtype = float)
X_testing = np.zeros((testing_data, 784), dtype = float)
y_training = np.zeros((training_data), dtype = int)
y_testing = np.zeros((testing_data), dtype = int)

zeros = 0
ones = 1
for i in range(8817):
    if labels[i] == 0:
        zeros += 1
    else:
        ones += 1

print("Total 0's: ", zeros)
print("Total 1's: ", ones)

for i in range(8817):
    if i < training_data:
        X_training[i] = X[i]
        y_training[i] = labels[i]
    else:
        X_testing[i - training_data] = X[i]
        y_testing[i - training_data] = labels[i]
        
zeros = 0
ones = 1
for i in range(training_data):
    if y_training[i] == 0:
        zeros += 1
    else:
        ones += 1

print("\nTotal 0's in trainig data: ", zeros)
print("Total 1's in trainig data: ", ones)

zeros = 0
ones = 1
for i in range(testing_data):
    if y_testing[i] == 0:
        zeros += 1
    else:
        ones += 1

print("\nTotal 0's in testing data: ", zeros)
print("Total 1's in testing data: ", ones)

## normalization of the data
for i in range(0,8817):
    if i < training_data:
        X_training[i] /= 255.0
    else:
        X_testing[i - training_data] /= 255.0
        
theta = np.zeros((n), dtype = float)
lr = 0.9
#for i in range(0,k):
    #print("class ", i)
    #yy = tranform(y, i)
for p in range(0, 100):
    print("iteration ", p)
     
    print("###cost = ", cost_function(theta, X, y))
    theta = gradientDescent(theta, X, y, lr)
predictions = sigmoid_function(np.dot(X, theta.T))

# predictions vector
for i in range(X.shape[0]):
    if predictions[i] >= 0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0
        
true = 0
for i in range(X.shape[0]):
    if predictions[i] == y[i]:
        true += 1

print("Accuracy Train: ", true/8817)
