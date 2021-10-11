from __future__ import division
import numpy  as np
import random
import matplotlib.pyplot as plt

# #### $\text{Model taken from 'Bayesian Inference: Metropolis-Hastings Sampling'} \\
# \text{Ilker Yildirim}\\
# \text{Department of Brain and Cognitive Sciences}\\
# \text{University of Rochester}\\
# \text{Rochester, NY 14627}$

# #### $\mathbf{\text{For covariance}(\rho) = 0.0}$

mean = [0,0]
cov = [[1,0],[0,1]]

np.random.seed(1)
x, y = np.random.multivariate_normal(mean, cov, 200).T

def likelihood(x,y,rho):
    N = 200
    a_1 = 1/(2*np.pi*np.sqrt(1-rho**2))**N
    a_2 = 1
    cov = [[1,rho],[rho,1]]
    for i in range(N):
        X = [x[i], y[i]]
        a_2 = a_2*np.exp(-(1/2)*(np.matmul(np.matmul(np.transpose(X),np.linalg.inv(cov)),X)))
    return a_1*a_2

def prior(rho):
    return 1/(1-rho**2)**(3/2)

rho_init = 0.5
rho_accept = [rho_init]
rho_reject = []

rho_candi = []

j = 0
itr = [0]

N_1 = 10000

for i in range(N_1):
    post_0 = prior(rho_init)*likelihood(x,y,rho_init)
    rho_cand = 0.14*np.random.rand() - 0.07 + rho_accept[j] 
    rho_candi.append(rho_cand)
    post_1 =  prior(rho_cand)*likelihood(x,y,rho_cand)
    alpha = post_1/post_0
    u = np.random.rand()
    if(alpha > 1):
        rho_accept.append(rho_cand)
        rho_init = rho_cand
        j = j + 1
        itr.append(j)
    elif(alpha < 1 and alpha > u):
        rho_accept.append(rho_cand)
        rho_init = rho_cand
        j = j + 1
        itr.append(j)
    else:
        rho_reject.append(rho_cand)
print ('No. of samples accepted by MCMC sampling is %d' %j)

# fig, axes = plt.subplots(nrows=3, ncols=1)
# ax0, ax1, ax2 = axes.flatten()

# ax0.plot(x,y,'o')
# ax0.set_xlabel('x')
# ax0.set_ylabel('y')

# ax1.plot(itr, rho_accept)
# ax1.set_ylabel(r'$\rho$')
# ax1.set_xlabel('Iterations')

# ax2.hist(rho_accept,bins = 100, density = False)
# ax2.set_xlabel(r'$\rho$')
# ax2.set_ylabel('counts')

plt.hist(rho_accept, bins = 100, density = False)
plt.xlabel(r'$\rho$')
plt.ylabel('Iterations')

fig.tight_layout()
plt.show()


Mean = np.mean(rho_accept)
St_dev = np.std(rho_accept)

print 'Mean and Standard deviation of the posterior distribution of rho are %f and %f respectively.'%(Mean,St_dev)


# #### $\mathbf{\text{For covariance}(\rho) = 0.42}$

mean = [0,0]
cov = [[1,0.42],[0.42,1]]


np.random.seed(1)
x, y = np.random.multivariate_normal(mean, cov, 200).T

def likelihood(x,y,rho):
    N = 200
    a_1 = 1/(2*np.pi*np.sqrt(1-rho**2))**N
    a_2 = 1
    cov = [[1,rho],[rho,1]]
    for i in range(N):
        X = [x[i], y[i]]
        a_2 = a_2*np.exp(-(1/2)*(np.matmul(np.matmul(np.transpose(X),np.linalg.inv(cov)),X)))
    return a_1*a_2

def prior(rho):
    return 1/(1-rho**2)**(3/2)

rho_init = 0.8
rho_accept = [rho_init]
rho_reject = []
post = []

rho_candi = []

j = 0
itr = [0]

N_1 = 10000

for i in range(N_1):
    post_0 = prior(rho_init)*likelihood(x,y,rho_init)
    rho_cand = 0.14*np.random.rand() - 0.07 + rho_accept[j] 
    rho_candi.append(rho_cand)
    post_1 =  prior(rho_cand)*likelihood(x,y,rho_cand)
    alpha = post_1/post_0
    u = np.random.rand()
    if(alpha > 1):
        rho_accept.append(rho_cand)
        rho_init = rho_cand
        j = j + 1
        itr.append(j)
    elif(alpha < 1 and alpha > u):
        rho_accept.append(rho_cand)
        rho_init = rho_cand
        j = j + 1
        itr.append(j)
    else:
        rho_reject.append(rho_cand)


# In[5]:


# fig, axes = plt.subplots(nrows=3, ncols=1)
# ax0, ax1, ax2 = axes.flatten()

# ax0.plot(x,y,'o')
# ax0.set_xlabel('x')
# ax0.set_ylabel('y')

# ax1.plot(itr, rho_accept)
# ax1.set_ylabel(r'$\rho$')
# ax1.set_xlabel('Iterations')


# ax2.hist(rho_accept,bins = 100, density = False)
# ax2.set_xlabel(r'$\rho$')
# ax2.set_ylabel('counts')

plt.hist(rho_accept, bins = 100, density = False)
plt.xlabel(r'$\rho$')
plt.ylabel('Iterations')

fig.tight_layout()
plt.show()

Mean = np.mean(rho_accept)
St_dev = np.std(rho_accept)

print 'Mean and Standard deviation of the posterior distribution of rho are %f and %f respectively.'%(Mean,St_dev)

# $\mathbf{\text{Parameter estimation of a model }} \\ d(t) = n(t) + A\sin \omega t \\ \text{(a) Estimation of A}$
# 

def signal(t,A):                        #Signal definition
    omg = 0.5
    sig = A*np.sin(omg*np.pi*t)
    return sig

def signal_1(A): 
    num = 300
    t = np.linspace(0,5,num)
    omg = 0.5
    sig = A*np.sin(omg*np.pi*t)
    return sig

def Likelihood(d,A): 
    N_Realiz = 1000
    Samps = 300
    a_1 = (1/(2*np.pi))**(int(Samps/2))
    a_2 = a_1*np.exp(-np.dot(d-signal_1(A),d-signal_1(A))/2.0)
    return a_2

def Cov(N_Realize, Samps):
    N_Realiz = 1000
    Samps = 300
    Sigma = np.zeros((Samps,Samps))
    np.random.seed(1)
    X = np.random.normal(0,1,size=(N_Realiz, Samps))

    for i in range(Samps):
        for j in range(Samps): 
            for k in range(N_Realiz):
                Sig_temp = Sig_temp + X[k][i]*X[k][j] 
            Sig_temp = Sig_temp/N_Realiz
            Sigma[i][j] = Sig_temp
            Sig_temp = 0
    return Sigma
        
A  = 0.5
omg = 0.5
num = 300
t = np.linspace(0,5,num)
np.random.seed(1)
n = np.random.normal(0,1,num)
s = signal(t,A)
data =  s + n                         #Data generation

A_init = 0.1
A_min, A_max = -1, 1
prior = 1/(A_max - A_min)

A_accept = [0.1]
A_reject = []

j = 0
itr = [0]
N_1 = 10**5

for i in range(N_1):
    post_0 = prior*Likelihood(data,A_init)
    A_cand = 2*np.random.rand() - 1  
    post_1 =  prior*Likelihood(data,A_cand)
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

fig, axes = plt.subplots(nrows=3, ncols=1)
ax0, ax1, ax2 = axes.flatten()

ax0.plot(t,data)
ax0.set_ylabel('d(t)')
ax0.set_xlabel('t')

ax1.plot(itr, A_accept)
ax1.set_ylabel('A')
ax1.set_xlabel('Iterations')

ax2.hist(A_accept,bins = 100, density = False)
ax2.set_xlabel('A')
ax2.set_ylabel('counts')

fig.tight_layout()
plt.show()

Mean = np.mean(A_accept)
St_dev = np.std(A_accept)

print 'Mean and Standard deviation of the posterior distribution of A are %f and %f respectively.'%(Mean,St_dev)


# $\text{(b) Estimation of }\omega$

def signal(t,A):                        #Signal definition
    omg = 0.5
    sig = A*np.sin(omg*np.pi*t)
    return sig

def signal_2(omega): 
    num = 300
    t = np.linspace(0,5,num)
    A = 0.01
    sig = A*np.sin(omega*np.pi*t)
    return sig

def Likelihood_1(d,omega): 
    num = 300
    a_1 = (1/(2*np.pi))**(int(num/2))
    a_2 = a_1*np.exp(-np.dot(d-signal_2(omega),d-signal_2(omega))/2.0)
    return a_2

A  = 0.01
omg = 0.5
num = 300
t = np.linspace(0,5,num)
np.random.seed(1)
n = np.random.normal(0,1,num)
s = signal(t,A)
data = n + s                          #Data generation

omega_init = 0.2
omega_min, omega_max = -1, 1
prior = 1/(omega_max - omega_min)

omega_accept = [omega_init]
omega_reject = []
post = []

j = 0
itr = [0]
N_1 = 10000

for i in range(N_1):
    post_0 = prior*Likelihood_1(data,omega_init)
    omega_cand = 6*np.random.rand() - 3  #+  omega_accept[j]
    post_1 =  prior*Likelihood_1(data,omega_cand)
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

plt.hist(omega_accept, 100)
plt.show()

plt.plot(itr,omega_accept)

def likelihood(x,y,rho):
    N = 200
    a_1 = 1/(2*np.pi*np.sqrt(1-rho**2))**N
    a_2 = a_1*np.exp(-(1/(2*(1-rho**2)))*(np.dot(np.transpose(x),x) + np.dot(np.transpose(y),y) - 2*np.matmul(np.matmul(np.transpose(x),rho*np.identity(N)),y)))
    return a_2

def prior(rho):
    return 1/(1-rho**2)**(3/2)

rho_init = 0.5
rho_accept = [rho_init]
rho_reject = []
post = []

rho_candi = []

j = 0
itr = [0]

N_1 = 10000

for i in range(N_1):
    post_0 = prior(rho_init)*likelihood(x,y,rho_init)
    rho_cand = 0.14*np.random.rand() - 0.07 + rho_accept[j] 
    rho_candi.append(rho_cand)
    post_1 =  prior(rho_cand)*likelihood(x,y,rho_cand)
    alpha = post_1/post_0
    u = np.random.rand()
    if(alpha > 1):
        rho_accept.append(rho_cand)
        rho_init = rho_cand
        j = j + 1
        itr.append(j)
    elif(alpha < 1 and alpha > u):
        rho_accept.append(rho_cand)
        rho_init = rho_cand
        j = j + 1
        itr.append(j)
    else:
        rho_reject.append(rho_cand)


print ('No. of samples accepted by MCMC sampling is %d' %j)
print(len(rho_candi))
plt.hist(rho_candi, 50, label = 'candidate')
plt.hist(rho_accept, 50, label = 'accepted')

plt.legend()
plt.show()

Mean = np.mean(rho_accept)
St_dev = np.std(rho_accept)

print 'Mean and Standard deviation of the posterior distribution of A are %f and %f respectively.'%(Mean,St_dev)

mean = [0,0]
cov = [[1,0.42],[0.42,1]]

N = 200
np.random.seed(1)
x, y = np.random.multivariate_normal(mean, cov, N).T

def likelihood(x,y,rho):
    N = 200
    a_1 = 1/(2*np.pi*np.sqrt(1-rho**2))**N
    a_2 = 1
    cov = [[1,rho],[rho,1]]
    for i in range(N):
        X = [x[i], y[i]]
        a_2 = a_2*np.exp(-(1/2)*(np.matmul(np.matmul(np.transpose(X),np.linalg.inv(cov)),X)))
    return a_1*a_2

def likelihood_1(x,y,rho):
    a_1 = 1/(2*np.pi*np.sqrt(1-rho**2))
    a_2 = 1.0
    for i in range(200):
        a_2 = a_2*a_1*np.exp(-(1/(2*(1-rho**2)))*(x[i]**2 - 2*rho*x[i]*y[i] + y[i]**2))
    return a_2

likelihood(x,y,0.5),likelihood_1(x,y,0.5)


# #### Code to calculate the program run time

import timeit

start = timeit.default_timer()

likelihood(x, y, 0.5)

stop = timeit.default_timer()

print('Time: ', stop - start) 



# ### Test Code

mean = [0,0]
cov = [[1,0.42],[0.42,1]]


np.random.seed(1)
x, y = np.random.multivariate_normal(mean, cov, 200).T

def likelihood(x,y,rho):
    N = 200
    a_1 = 1/(2*np.pi*np.sqrt(1-rho**2))**N
    a_2 = a_1*np.exp(-(1/(2*(1-rho**2)))*(np.dot(np.transpose(x),x) + np.dot(np.transpose(y),y) - 2*np.matmul(np.matmul(np.transpose(x),rho*np.identity(N)),y)))
    return a_2

def prior(rho):
    return 1/(1-rho**2)**(3/2)

rho_init = 0.8
rho_accept = [rho_init]
rho_reject = []
post = []

rho_candi = []

j = 0
itr = [0]

N_1 = 10000

a = 0.14

for i in range(N_1):
    post_0 = prior(rho_init)*likelihood(x,y,rho_init)
    rho_cand = a*np.random.rand() - (a/2) + rho_accept[j] 
    rho_candi.append(rho_cand)
    post_1 =  prior(rho_cand)*likelihood(x,y,rho_cand)
    alpha = post_1/post_0
    u = np.random.rand()
    if(alpha > 1):
        rho_accept.append(rho_cand)
        rho_init = rho_cand
        j = j + 1
        itr.append(j)
    elif(alpha < 1 and alpha > u):
        rho_accept.append(rho_cand)
        rho_init = rho_cand
        j = j + 1
        itr.append(j)
    else:
        rho_reject.append(rho_cand)

fig, axes = plt.subplots(nrows=3, ncols=1)
ax0, ax1, ax2 = axes.flatten()

ax0.plot(x,y,'o')
ax0.set_xlabel('x')
ax0.set_ylabel('y')
#ax0.set_aspect(adjustable='box', aspect = 0.9)

ax1.plot(itr, rho_accept)
ax1.set_ylabel(r'$\rho$')
ax1.set_xlabel('Iterations')


ax2.hist(rho_accept,bins = 100, density = False)
ax2.set_xlabel(r'$\rho$')
ax2.set_ylabel('counts')

fig.tight_layout()
plt.show()

Mean = np.mean(rho_accept)
St_dev = np.std(rho_accept)

print 'Mean and Standard deviation of the posterior distribution of rho are %f and %f respectively.'%(Mean,St_dev)


# ### Proposal distribution: Gaussian

mean = [0,0]
cov = [[1,0.42],[0.42,1]]


np.random.seed(1)
x, y = np.random.multivariate_normal(mean, cov, 200).T

def likelihood(x,y,rho):
    N = 200
    a_1 = 1/(2*np.pi*np.sqrt(1-rho**2))**N
    a_2 = a_1*np.exp(-(1/(2*(1-rho**2)))*(np.dot(np.transpose(x),x) + np.dot(np.transpose(y),y) - 2*np.matmul(np.matmul(np.transpose(x),rho*np.identity(N)),y)))
    return a_2

def prior(rho):
    return 1/(1-rho**2)**(3/2)


#prior = 1.0

rho_init = 0.8
rho_accept = [rho_init]
rho_reject = []
post = []

rho_candi = []

j = 0
itr = [0]

N_1 = 10000

a = 0.14

for i in range(N_1):
    #post_0 = prior*likelihood(x,y,rho_init)
    post_0 = prior(rho_init)*likelihood(x,y,rho_init)
    rho_cand = a*np.random.normal(0,1) - (a/2) + rho_accept[j] 
    rho_candi.append(rho_cand)
    post_1 =  prior(rho_cand)*likelihood(x,y,rho_cand)
    #post_1 =  prior*likelihood(x,y,rho_cand)
    alpha = post_1/post_0
    u = np.random.rand()
    if(alpha > 1):
        rho_accept.append(rho_cand)
        rho_init = rho_cand
        j = j + 1
        itr.append(j)
    elif(alpha < 1 and alpha > u):
        rho_accept.append(rho_cand)
        rho_init = rho_cand
        j = j + 1
        itr.append(j)
    else:
        rho_reject.append(rho_cand)

fig, axes = plt.subplots(nrows=3, ncols=1)
ax0, ax1, ax2 = axes.flatten()

ax0.plot(x,y,'o')
ax0.set_xlabel('x')
ax0.set_ylabel('y')

ax1.plot(itr, rho_accept)
ax1.set_ylabel(r'$\rho$')
ax1.set_xlabel('Iterations')


ax2.hist(rho_accept,bins = 100, density = False)
ax2.set_xlabel(r'$\rho$')
ax2.set_ylabel('counts')

fig.tight_layout()
plt.show()

Mean = np.mean(rho_accept)
St_dev = np.std(rho_accept)

print 'Mean and Standard deviation of the posterior distribution of rho are %f and %f respectively.'%(Mean,St_dev)

np.random.normal(0,0.15)

Noise = np.zeros((100,300))

for i in range(100):
    for j in range(300):
        Noise[i][j] = np.random.normal(0,1)

num = 300
n = np.random.normal(0,1,num)
plt.plot(n)

N_Realiz = 100
Samps = 300
Sigma = np.zeros((Samps,Samps))
np.random.seed(1)
X = np.random.normal(0,1,size=(N_Realiz, Samps))


import timeit

start = timeit.default_timer()

N_Realiz = 100
Samps = 300
Sigma = np.zeros((Samps,Samps))
np.random.seed(1)
X = np.random.normal(0,1,size=(N_Realiz, Samps))

for i in range(Samps):
    for j in range(Samps):
        Sig_temp = 0
        for k in range(N_Realiz):
            Sig_temp = Sig_temp + X[k][i]*X[k][j] 
        Sig_temp = Sig_temp/N_Realiz
        Sigma[i][j] = Sig_temp
         
        
stop = timeit.default_timer()

print('Time: ', stop - start) 

Sigma

import timeit

start = timeit.default_timer()

N_Realiz = 100
Samps = 300
Sigma = np.zeros((Samps,Samps))
np.random.seed(1)
X = np.random.normal(0,1,size=(N_Realiz, Samps))

for i in range(Samps):
    for j in range(Samps): 
        Sig_temp = 0
        if i <= j:
            for k in range(N_Realiz):
                Sig_temp = Sig_temp + X[k][i]*X[k][j] 
            Sig_temp = Sig_temp/N_Realiz
            Sigma[i][j] = Sig_temp
            
        else:
            Sigma[i][j] = Sigma[j][i]

stop = timeit.default_timer()

print('Time: ', stop - start) 

Sigma

import timeit

start = timeit.default_timer()

likelihood(x, y, 0.5)

stop = timeit.default_timer()

print('Time: ', stop - start) 
