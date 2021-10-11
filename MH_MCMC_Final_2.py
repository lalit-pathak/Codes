#!/usr/bin/env python
# coding: utf-8

# In[230]:


from __future__ import division
import numpy  as np
import random
import matplotlib.pyplot as plt


# ### For $\sigma = 1.0$

# In[274]:


def signal(params,t):                                    #Signal definition
    sig = params[0]*np.sin(params[1]*np.pi*t)
    return sig

def signal_1(params,cov_fac,oth_fac): 
    t = np.linspace(0,oth_fac[1],cov_fac[1])
    sig = params[0]*np.sin(params[1]*np.pi*t)
    return sig

def Likelihood(d,params,std,cov_fac,oth_fac,Cov_inv):
    #a_1 = ((1/(2*np.pi))**(int(cov_fac[1]/2)))*Det**(-1.0/2.0)
    a_2 = np.exp(-np.matmul(np.matmul(np.transpose(d-signal_1(params,cov_fac,oth_fac)),Cov_inv),d-signal_1(params,cov_fac,oth_fac))/2)
    return a_2

def Cov(std,cov_fac):
    np.random.seed(1)
    X = np.random.normal(0,std,size=(cov_fac[0], cov_fac[1]))
    X_T = np.transpose(X)
    Mean = np.mean(X_T, axis = 1)
    Sigma = np.zeros((cov_fac[1],cov_fac[1]))
    for i in range(cov_fac[1]):
        for j in range(cov_fac[1]):
            if i<=j:
                Sigma[i][j] = np.sum((X_T[i]-Mean[i])*(X_T[j]-Mean[j]), axis = 0)/cov_fac[0]
            else:
                Sigma[i][j] = Sigma[j][i]
    return Sigma

params = [0.5,0.3]
oth_fac = [10000,40]
cov_fac = [oth_fac[0],1000]
std = 1.0

t = np.linspace(0,oth_fac[1],cov_fac[1])
np.random.seed(1)
n = np.random.normal(0,std,cov_fac[1])
s = signal(params,t)
data =  s + n  


# In[275]:


import timeit

start = timeit.default_timer()

Covar = Cov(std,cov_fac)
Cov_inv = np.linalg.inv(Covar) 

params_init, omega_range, A_range  = [0.4, 0.2], [0, 1.0], [-1.0, 1.0]                  
prior = (1/(omega_range[1] - omega_range[0]))*(1/(A_range[1] - A_range[0]))      
                                            
A_accept = [params_init[0]]
A_reject = []
        
omega_accept = [params_init[1]]
omega_reject = []

j = 0
itr = [0]
N_1 = 10**7

for i in range(N_1):
    post_0 = prior*Likelihood(data,params_init,std,cov_fac,oth_fac,Cov_inv)
    params_cand = [0.05*np.random.rand() - 0.025  + A_accept[j], 0.10*np.random.rand() - 0.05 + omega_accept[j]]   
    post_1 =  prior*Likelihood(data,params_cand,std,cov_fac,oth_fac,Cov_inv)
    alpha = post_1/post_0
    u = np.random.rand()
    if(alpha > 1):
        A_accept.append(params_cand[0])
        params_init[0] = params_cand[0]
        omega_accept.append(params_cand[1])
        params_init[1] = params_cand[1]
        j = j + 1
        itr.append(j)
    elif(alpha < 1 and alpha > u):
        A_accept.append(params_cand[0])
        params_init[0] = params_cand[0]
        omega_accept.append(params_cand[1])
        params_init[1] = params_cand[1]
        j = j + 1
        itr.append(j)
    else:
        A_reject.append(params_cand[0])
        omega_reject.append(params_cand[1])
        
        
stop = timeit.default_timer()

print('Time: ', stop - start) 


# In[276]:


Mean_A = np.mean(A_accept)
St_dev_A = np.std(A_accept)

Mean_omega = np.mean(omega_accept)
St_dev_omega = np.std(omega_accept)

n_A, bins_A, patches_A = plt.hist(A_accept, bins = 100)

plt.xlabel('A')
plt.ylabel('counts')
plt.show()


# In[277]:


n_omega, bins_omega, patches_omega = plt.hist(omega_accept, bins = 100)

plt.xlabel(r'$\omega$')
plt.ylabel('counts')
plt.show()


# In[278]:


np.shape(A_accept), np.shape(omega_accept)


# In[279]:


y_A = (1/(St_dev_A*(2*np.pi)**(1/2)))*np.exp(-(bins_A-Mean_A)**2/(2*St_dev_A**2))
y_omega = (1/(St_dev_omega*(2*np.pi)**(1/2)))*np.exp(-(bins_omega-Mean_omega)**2/(2*St_dev_omega**2))

bins_A, bins_omega = np.meshgrid(bins_A, bins_omega)
z = ((1/(St_dev_A*(2*np.pi)**(1/2)))*np.exp(-(bins_A-Mean_A)**2/(2*St_dev_A**2)))*((1/(St_dev_omega*(2*np.pi)**(1/2)))*np.exp(-(bins_omega-Mean_omega)**2/(2*St_dev_omega**2)))


# In[280]:


plt.contour(bins_A, bins_omega, z, 100, cmap=plt.cm.jet )
plt.colorbar()
plt.plot(0.5,0.3, 'k+')
plt.xlabel('A(Amplitude)', fontsize = '15')
plt.ylabel(r'$\omega$(Angular frequency)', fontsize = '15' )
plt.show()


# In[281]:


plt.pcolormesh(bins_A, bins_omega, z, cmap=plt.cm.jet)
plt.colorbar()
plt.plot(0.5,0.3, 'k+')
plt.xlabel('A(Amplitude)', fontsize = '15')
plt.ylabel(r'$\omega$(Angular frequency)', fontsize = '15' )
plt.show()


# In[282]:


Mean_A,St_dev_A,Mean_omega,St_dev_omega 


# In[ ]:





# ### For $\sigma = 0.25$

# In[307]:


def signal(params,t):                                    #Signal definition
    sig = params[0]*np.sin(params[1]*np.pi*t)
    return sig

def signal_1(params,cov_fac,oth_fac): 
    t = np.linspace(0,oth_fac[1],cov_fac[1])
    sig = params[0]*np.sin(params[1]*np.pi*t)
    return sig

def Likelihood(d,params,std,cov_fac,oth_fac,Cov_inv):
    #a_1 = ((1/(2*np.pi))**(int(cov_fac[1]/2)))*Det**(-1.0/2.0)
    a_2 = np.exp(-np.matmul(np.matmul(np.transpose(d-signal_1(params,cov_fac,oth_fac)),Cov_inv),d-signal_1(params,cov_fac,oth_fac))/2)
    return a_2

def Cov(std,cov_fac):
    np.random.seed(1)
    X = np.random.normal(0,std,size=(cov_fac[0], cov_fac[1]))
    X_T = np.transpose(X)
    Mean = np.mean(X_T, axis = 1)
    Sigma = np.zeros((cov_fac[1],cov_fac[1]))
    for i in range(cov_fac[1]):
        for j in range(cov_fac[1]):
            if i<=j:
                Sigma[i][j] = np.sum((X_T[i]-Mean[i])*(X_T[j]-Mean[j]), axis = 0)/cov_fac[0]
            else:
                Sigma[i][j] = Sigma[j][i]
    return Sigma

params = [0.5,0.3]
oth_fac = [10000,40]
cov_fac = [oth_fac[0],300]
std = 0.25

t = np.linspace(0,oth_fac[1],cov_fac[1])
np.random.seed(1)
n = np.random.normal(0,std,cov_fac[1])
s = signal(params,t)
data =  s + n  


# In[308]:


import timeit

start = timeit.default_timer()

Covar = Cov(std,cov_fac)
Cov_inv = np.linalg.inv(Covar) 

params_init, omega_range, A_range  = [0.4, 0.2], [0, 1.0], [-1.0, 1.0]                  
prior = (1/(omega_range[1] - omega_range[0]))*(1/(A_range[1] - A_range[0]))      
                                            
A_accept = [params_init[0]]
A_reject = []
        
omega_accept = [params_init[1]]
omega_reject = []

j = 0
itr = [0]
N_1 = 10**6

for i in range(N_1):
    post_0 = prior*Likelihood(data,params_init,std,cov_fac,oth_fac,Cov_inv)
    params_cand  = [0.05*np.random.rand() - 0.025  + A_accept[j], 0.10*np.random.rand() - 0.05 + omega_accept[j]]   
    post_1 =  prior*Likelihood(data,params_cand,std,cov_fac,oth_fac,Cov_inv)
    alpha = post_1/post_0
    u = np.random.rand()
    if(alpha > 1):
        A_accept.append(params_cand[0])
        params_init[0] = params_cand[0]
        omega_accept.append(params_cand[1])
        params_init[1] = params_cand[1]
        j = j + 1
        itr.append(j)
    elif(alpha < 1 and alpha > u):
        A_accept.append(params_cand[0])
        params_init[0] = params_cand[0]
        omega_accept.append(params_cand[1])
        params_init[1] = params_cand[1]
        j = j + 1
        itr.append(j)
    else:
        A_reject.append(params_cand[0])
        omega_reject.append(params_cand[1])
        
        
stop = timeit.default_timer()

print('Time: ', stop - start) 


# In[309]:


Mean_A = np.mean(A_accept)
St_dev_A = np.std(A_accept)

Mean_omega = np.mean(omega_accept)
St_dev_omega = np.std(omega_accept)

n_A, bins_A, patches_A = plt.hist(A_accept, bins = 100)

plt.xlabel('A')
plt.ylabel('counts')
plt.show()


# In[310]:


n_omega, bins_omega, patches_omega = plt.hist(omega_accept, bins = 100)

plt.xlabel(r'$\omega$')
plt.ylabel('counts')
plt.show()

y_A = (1/(St_dev_A*(2*np.pi)**(1/2)))*np.exp(-(bins_A-Mean_A)**2/(2*St_dev_A**2))
y_omega = (1/(St_dev_omega*(2*np.pi)**(1/2)))*np.exp(-(bins_omega-Mean_omega)**2/(2*St_dev_omega**2))

bins_A, bins_omega = np.meshgrid(bins_A, bins_omega)
z = ((1/(St_dev_A*(2*np.pi)**(1/2)))*np.exp(-(bins_A-Mean_A)**2/(2*St_dev_A**2)))*((1/(St_dev_omega*(2*np.pi)**(1/2)))*np.exp(-(bins_omega-Mean_omega)**2/(2*St_dev_omega**2)))


# In[311]:


np.shape(A_accept), np.shape(omega_accept)


# In[312]:


plt.pcolormesh(bins_A, bins_omega, z, cmap=plt.cm.jet)
plt.colorbar()
plt.plot(0.5,0.3, 'k+')
plt.xlabel('A(Amplitude)', fontsize = '15')
plt.ylabel(r'$\omega$(Angular frequency)', fontsize = '15' )
plt.show()


# In[313]:


Mean_A,St_dev_A,Mean_omega,St_dev_omega 


# In[ ]:





# In[ ]:




