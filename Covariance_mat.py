from __future__ import division
import numpy  as np
import random
import matplotlib.pyplot as plt

def Cov(N_Realiz, Samps, mean, sigma):
    np.random.seed(1)
    X = np.random.normal(mean,sigma,size=(N_Realiz, Samps))
    X_T = np.transpose(X)
    Mean = np.mean(X_T, axis = 1)
    Sigma = np.zeros((Samps,Samps))
    for i in range(Samps):
        for j in range(Samps):
            if i<=j:
                Sigma[i][j] = np.sum((X_T[i]-Mean[i])*(X_T[j]-Mean[j]), axis = 0)/N_Realiz
            else:
                Sigma[i][j] = Sigma[j][i]
    return Sigma

import timeit

start = timeit.default_timer()

Covar = Cov(10000,500,0.0,1.0)
Cov_inv = np.linalg.inv(Covar) 
        
stop = timeit.default_timer()

print('Time: ', stop - start)





