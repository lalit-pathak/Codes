from __future__ import division
import numpy  as np
import random
import matplotlib.pyplot as plt

def Sin(omega,t):
    return np.sin(omega*t)

def Cov(N_Realiz, Samps):
    np.random.seed(1)
    X = np.random.normal(0,1.0,size=(N_Realiz, Samps))
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

def Atp2sin(A,omega,t):
    return A*t**2*np.sin(omega*t)

def Atcos(A,omega,t):
    return A*t*np.cos(omega*t)

def Asin(A,omega,t):
    return A*np.sin(omega*t)

def sig(A,omega,samps,t):
    np.random.seed(1)
    d = A*np.sin(omega*t) + np.random.normal(0,1,samps)
    return d

samp, N_realiz = 500, 100000
Cov_mat = Cov(N_realiz,samp)
Cov_inv = np.linalg.inv(Cov_mat)

F_1 = np.zeros((2,2))
A, omega = 0.45, 0.27
t = np.linspace(0,20,samp)
V = Sin(omega,t)
V_1 = Atp2sin(A,omega,t)
V_2 = sig(A_true,omega_true,samp,t) - Asin(A,omega,t)
V_3 = Atcos(A,omega,t)

for i in range(2):
    for j in range(2):
        if i == 0 and j == 0:
            F_1[i][j] = np.matmul(np.matmul(np.transpose(V),Cov_inv), V)/samp
        if i == 0 and j == 1:
            F_1[i][j] = (np.matmul(np.matmul(np.transpose(V_3),Cov_inv), V))/samp
        if i == 1 and j == 0:
            F_1[i][j] = (np.matmul(np.matmul(np.transpose(V_3),Cov_inv), V))/samp
        if i == 1 and j == 1:
            F_1[i][j]= (np.matmul(np.matmul(np.transpose(V_3),Cov_inv), V_3))/samp

            
Cov = np.linalg.inv(F_1)
F_1

F_1 = np.zeros((2,2))
A, omega = 0.1, 0.2
t = np.linspace(0,20,samp)
V = Sin(omega,t)
V_1 = Atp2sin(A,omega,t)
V_2 = sig(A_true,omega_true,samp,t) - Asin(A,omega,t)
V_3 = Atcos(A,omega,t)

for i in range(2):
    for j in range(2):
        if i == 0 and j == 0:
            F_1[i][j] = np.matmul(np.matmul(np.transpose(V),Cov_inv), V)/samp
        if i == 0 and j == 1:
            F_1[i][j] = (np.matmul(np.matmul(np.transpose(V_3),Cov_inv), V))/samp
        if i == 1 and j == 0:
            F_1[i][j] = (np.matmul(np.matmul(np.transpose(V_3),Cov_inv), V))/samp
        if i == 1 and j == 1:
            F_1[i][j]= (np.matmul(np.matmul(np.transpose(V_3),Cov_inv), V_3))/samp

            
Cov = np.linalg.inv(F_1)
F_1

F_1 = np.zeros((2,2))
A, omega = 1000, 100
t = np.linspace(0,20,samp)
V = Sin(omega,t)
V_1 = Atp2sin(A,omega,t)
V_2 = sig(A_true,omega_true,samp,t) - Asin(A,omega,t)
V_3 = Atcos(A,omega,t)

for i in range(2):
    for j in range(2):
        if i == 0 and j == 0:
            F_1[i][j] = np.matmul(np.matmul(np.transpose(V),Cov_inv), V)/samp
        if i == 0 and j == 1:
            F_1[i][j] = (np.matmul(np.matmul(np.transpose(V_3),Cov_inv), V))/samp
        if i == 1 and j == 0:
            F_1[i][j] = (np.matmul(np.matmul(np.transpose(V_3),Cov_inv), V))/samp
        if i == 1 and j == 1:
            F_1[i][j]= (np.matmul(np.matmul(np.transpose(V_3),Cov_inv), V_3))/samp

            
Cov = np.linalg.inv(F_1)
F_1

num = 100
A, omega = 1.5, 0.5
t = np.linspace(0,20,num)
S = sum((Asin(A,omega,t)**2))/num
np.random.seed(1)
n = np.random.normal(0,1,size = 100)
N = sum(n**2)/num
print 'SNR =',S/N
