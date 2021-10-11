#!/usr/bin/env python
# coding: utf-8

# In[15]:


from __future__ import division
import numpy  as np
import random
import matplotlib.pyplot as plt

def summ(x,init,fin):
    i = init
    summ_1 = 0
    while i<fin:
        summ_1 = summ_1 + x[i]
        i = i + 1
    return summ_1


# #### Data generation

# In[25]:


lamb_1, lamb_2, n, N = 3, 9, 26, 50
np.random.seed(1)
x_1 = np.random.poisson(lamb_1, n)
x_2 = np.random.poisson(lamb_2, N-n)
X = np.concatenate((x_1,x_2), axis=0)


# #### Gibbs sampler

# In[26]:


a, b = 2, 1
np.random.seed(1)
lam_1, lam_2, n = np.random.gamma(a,b), np.random.gamma(a,b), int((N-1)*np.random.rand() + 1)

Chain = 5200
Burn = 200

samp_n = np.zeros(Chain-Burn)
samp_lambda_1 = np.zeros(Chain-Burn)
samp_lambda_2 = np.zeros(Chain-Burn)

for i in range(Chain):
    lam_1 = np.random.gamma(a+sum(X[0:n]),n+b)
    lam_2 = np.random.gamma(a+sum(X[n:N]),N-n+b)
    mult_n = np.zeros(N)
    for j in range(N):
        mult_n[j]=sum(X[0:j])*np.log(lam_1)-j*lam_1+sum(X[j:N])*np.log(lam_2)-(N-j)*lam_2
    mult_n = np.exp(mult_n-max(mult_n))
    n = np.where(np.random.multinomial(1,mult_n/sum(mult_n),size=1)==1)[1][0]
    #print np.where(np.random.multinomial(1,mult_n/sum(mult_n),size=1)==1)
    if i>=Burn:
        samp_n[i-Burn] = n
        samp_lambda_1[i-Burn] = lam_1
        samp_lambda_2[i-Burn] = lam_2


# In[23]:


plt.hist(samp_n, 100)
plt.show()
np.mean(samp_n), np.std(samp_n)


# In[21]:


plt.hist(samp_lambda1, 100)
plt.show()
np.mean(samp_lambda1), np.std(samp_lambda1)


# In[94]:


plt.hist(samp_lambda2, 100)
plt.show()
np.mean(samp_lambda2), np.std(samp_lambda2)


# In[ ]:





# In[ ]:




