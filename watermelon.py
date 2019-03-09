import numpy as np
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn import metrics

def likelihood(X,y):
    m,n = np.shape(X)
    llh = np.c_[X,np.ones(m)]
    beta = np.array([0,0,0])
    max_times = 5000
    for i in range(max_times):
        beta_deriv1 = [0,0,0]
        beta_deriv2 = 0
        for j in range(m):
            temp = np.exp(np.dot(np.transpose(beta),llh[j]))
            p1 = temp/(1+temp)
            beta_deriv1 += - np.dot(llh[j],y[j] - p1)
            beta_deriv2 += np.dot(llh[j],np.transpose(llh[j]))*(1 - p1)
        beta = beta - np.dot((1 / beta_deriv2),beta_deriv1)
    return beta

def predict(beta,X_test):
    m,n = np.shape(X_test)
    X_test = np.c_[X_test,np.ones(m)]
    prediction = []
    for i in  range(m):
        y = 1 / (1 + (np.exp(-np.dot(np.transpose(beta),X_test[i]))))
        if y > 0.5:
            y = 1
        else:
            y = 0
        prediction.append(y)
    return prediction

#load data
melon_data = np.loadtxt('../watermelon.csv',delimiter = ",")

X = melon_data[:,1:3]
y = melon_data[:,3]
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.5,random_state = 0)
beta = likelihood(X_train,y_train)
y_pred = predict(beta,X_test)
print("y_test:")
print(np.array(np.transpose(y_test)))
print("y_pred:")
print(np.array(y_pred))
print("confusion matrics:")
print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.classification_report(y_test,y_pred))





