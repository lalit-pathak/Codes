#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division
import numpy  as np
import random
import matplotlib.pyplot as plt


# $\mathbf{\text{Parameter estimation of a model }} \\ d(t) = n(t) + A\sin \omega t \\ \text{(a) Estimation of A}$
# 

# In[60]:


def signal(t,A):                        #Signal definition
    omg = 0.3
    sig = A*np.sin(omg*np.pi*t)
    return sig

def signal_1(A): 
    num = 500
    t = np.linspace(0,20,num)
    omg = 0.3
    sig = A*np.sin(omg*np.pi*t)
    return sig

def Likelihood(d,A,Cov_inv): 
    Samps = 500
    std = 0.25
    a_1 = (1/(2*np.pi*std**2))**(int(Samps/2))
    a_2 = a_1*np.exp(-np.matmul(np.matmul(np.transpose(d-signal_1(A)),Cov_inv),d-signal_1(A))/2)
    return a_2

def Cov(N_Realiz, Samps):
    np.random.seed(1)
    X = np.random.normal(0,0.25,size=(N_Realiz, Samps))
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
        
A  = 0.5
omg = 0.3
N_Realiz = 10000
num = 500
t = np.linspace(0,20,num)
np.random.seed(1)
n = np.random.normal(0,0.25,num)
s = signal(t,A)
data =  s + n  


# In[61]:


Cov_inv = np.linalg.inv(Cov(N_Realiz,num)) 

A_init = 0.4                   
A_min,A_max  = -1.5, 2.0
prior = 1/(A_max - A_min)      
                                            
A_accept = [A_init]
A_reject = []

j = 0
itr = [0]
N_1 = 10**5

for i in range(N_1):
    post_0 = prior*Likelihood(data,A_init,Cov_inv)
    A_cand = 2*np.random.rand() - 1   
    post_1 =  prior*Likelihood(data,A_cand,Cov_inv)
    alpha = post_1/post_0
    u = np.random.rand()
    if(alpha > 1):
        A_accept.append(A_cand)
        A_init = A_cand
        j = j + 1
        itr.append(j)
    elif(alpha < 1 and alpha > u):
        A_accept.append(A_cand)
        A_init = A_cand
        j = j + 1
        itr.append(j)
    else:
        A_reject.append(A_cand)
        
plt.hist(A_accept, 100)
plt.show()
np.mean(A_accept), np.std(A_accept)


# $\text{(b) Estimation of }\omega$

# In[62]:


def signal(t,A,omg):                        #Signal definition
    sig = A*np.sin(omg*np.pi*t)
    return sig

def signal_1(omg): 
    num = 500
    t = np.linspace(0,20,num)
    A = 0.5
    sig = A*np.sin(omg*np.pi*t)
    return sig

def Likelihood(d,omg,Cov_inv): 
    Samps = 500
    std = 0.25
    a_1 = (1/(2*np.pi*std**2))**(int(Samps/2))
    a_2 = a_1*np.exp(-np.matmul(np.matmul(np.transpose(d-signal_1(omg)),Cov_inv),d-signal_1(omg))/2)
    return a_2

def Cov(N_Realiz, Samps):
    np.random.seed(1)
    X = np.random.normal(0,0.25,size=(N_Realiz, Samps))
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
        
A  = 0.5
omg = 0.3
N_Realiz = 10000
num = 500
t = np.linspace(0,20,num)
np.random.seed(1)
n = np.random.normal(0,0.25,num)
s = signal(t,A,omg)
data =  s + n  


# In[63]:


Cov_inv = np.linalg.inv(Cov(N_Realiz,num)) 

omega_init = 0.4                   
omega_min,omega_max  = -1.0, 1.0
prior = 1/(omega_max - omega_min)      
                                            
omega_accept = [omega_init]
omega_reject = []

j = 0
itr = [0]
N_1 = 10**5

for i in range(N_1):
    post_0 = prior*Likelihood(data,omega_init,Cov_inv)
    omega_cand = np.random.rand()   
    post_1 =  prior*Likelihood(data,omega_cand,Cov_inv)
    alpha = post_1/post_0
    u = np.random.rand()
    if(alpha > 1):
        omega_accept.append(omega_cand)
        omega_init = omega_cand
        j = j + 1
        itr.append(j)
    elif(alpha < 1 and alpha > u):
        omega_accept.append(omega_cand)
        omega_init = omega_cand
        j = j + 1
        itr.append(j)
    else:
        omega_reject.append(omega_cand)


# In[64]:


plt.hist(omega_accept, 100)
plt.show()
np.mean(omega_accept), np.std(omega_accept)


# In[ ]:





# $\text{(b) Estimation of A and }\ \omega$

# In[33]:


def signal(t,A,omg):                        #Signal definition
    sig = A*np.sin(omg*np.pi*t)
    return sig

def signal_1(A,omg): 
    num = 1000
    t = np.linspace(0,40,num)
    sig = A*np.sin(omg*np.pi*t)
    return sig

def Likelihood(d,A,omg,Cov_inv): 
    Samps = 1000
    std = 0.25
    a_1 = (1/(2*np.pi*std**2))**(int(Samps/2))
    a_2 = a_1*np.exp(-np.matmul(np.matmul(np.transpose(d-signal_1(A,omg)),Cov_inv),d-signal_1(A,omg))/2)
    return a_2

def Cov(N_Realiz, Samps):
    np.random.seed(1)
    X = np.random.normal(0,1,size=(N_Realiz, Samps))
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
        
A  = 0.5
omg = 0.3
N_Realiz = 10000
num = 1000
t = np.linspace(0,40,num)
np.random.seed(1)
n = np.random.normal(0,0.25,num)
s = signal(t,A,omg)
data =  s + n  


# In[34]:


import timeit

start = timeit.default_timer()

Cov_inv = np.linalg.inv(Cov(N_Realiz,num)) 

A_init, omega_init = 0.4, 0.2                   
omega_min,omega_max  = 0, 1.0
A_min, A_max = -1.25, 2.0
prior = (1/(omega_max - omega_min))*(1/(A_max - A_min))      
                                            
A_accept = [A_init]
A_reject = []
        
omega_accept = [omega_init]
omega_reject = []

j = 0
itr = [0]
N_1 = 10**7

for i in range(N_1):
    post_0 = prior*Likelihood(data,A_init,omega_init,Cov_inv)
    A_cand, omega_cand = 2*np.random.rand()-1, np.random.rand()   
    post_1 =  prior*Likelihood(data,A_cand,omega_cand,Cov_inv)
    alpha = post_1/post_0
    u = np.random.rand()
    if(alpha > 1):
        A_accept.append(A_cand)
        A_init = A_cand
        omega_accept.append(omega_cand)
        omega_init = omega_cand
        j = j + 1
        itr.append(j)
    elif(alpha < 1 and alpha > u):
        A_accept.append(A_cand)
        A_init = A_cand
        omega_accept.append(omega_cand)
        omega_init = omega_cand
        j = j + 1
        itr.append(j)
    else:
        A_reject.append(A_cand)
        omega_reject.append(omega_cand)

plt.hist(A_accept, 100)

plt.show()
np.mean(A_accept), np.std(A_accept), np.mean(omega_accept), np.std(omega_accept)

stop = timeit.default_timer()

print('Time: ', stop - start) 


# In[35]:


np.mean(A_accept), np.std(A_accept), np.mean(omega_accept), np.std(omega_accept)


# In[50]:


plt.hist(omega_accept, 50)
plt.show()


# In[59]:


plt.plot(itr, A_accept, label = 'Sampled A')
plt.plot(itr, omega_accept, label = 'Sampled $\omega$')
plt.legend()


# In[37]:


Mean_A = np.mean(A_accept)
St_dev_A = np.std(A_accept)

Mean_omega = np.mean(omega_accept)
St_dev_omega = np.std(omega_accept)

n_A, bins_A, patches_A = plt.hist(A_accept, bins = 100)

plt.xlabel('A')
plt.ylabel('counts')
plt.show()


# In[38]:


n_omega, bins_omega, patches_omega = plt.hist(omega_accept, bins = 100)

plt.xlabel(r'$\omega$')
plt.ylabel('counts')
plt.show()


# In[39]:


np.shape(A_accept), np.shape(omega_accept)


# In[40]:


y_A = (1/(St_dev_A*(2*np.pi)**(1/2)))*np.exp(-(bins_A-Mean_A)**2/(2*St_dev_A**2))
y_omega = (1/(St_dev_omega*(2*np.pi)**(1/2)))*np.exp(-(bins_omega-Mean_omega)**2/(2*St_dev_omega**2))

bins_A, bins_omega = np.meshgrid(bins_A, bins_omega)
z = ((1/(St_dev_A*(2*np.pi)**(1/2)))*np.exp(-(bins_A-Mean_A)**2/(2*St_dev_A**2)))*((1/(St_dev_omega*(2*np.pi)**(1/2)))*np.exp(-(bins_omega-Mean_omega)**2/(2*St_dev_omega**2)))


# In[47]:


plt.contour(bins_A, bins_omega, z, 100, cmap=plt.cm.jet )
plt.colorbar()
plt.plot(0.5,0.3, 'k+')
plt.xlabel('A(Amplitude)', fontsize = '15')
plt.ylabel(r'$\omega$(Angular frequency)', fontsize = '15' )
plt.show()


# In[48]:


plt.pcolormesh(bins_A, bins_omega, z, cmap=plt.cm.jet)
plt.colorbar()
plt.plot(0.5,0.3, 'k+')
plt.xlabel('A(Amplitude)', fontsize = '15')
plt.ylabel(r'$\omega$(Angular frequency)', fontsize = '15' )
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




