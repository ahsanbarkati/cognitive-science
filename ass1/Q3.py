#!/usr/bin/env python
# coding: utf-8

# In[133]:
import matplotlib.pyplot as plt
import numpy as np
alpha = 4
mu = 0.001
sigma = 0.01


# In[134]:


def f(x,y):
    if(x<=0):
        return alpha/(1-x) + y
    if( x > 0 and x < alpha + y):
        return alpha + y
    return -1


# In[135]:


def gety(y,x):
    return (y - mu*(x+1) + mu*sigma)


# In[136]:


N=10000
x=np.zeros(N+1)
y=np.zeros(N+1)
for i in range(N):
    x[i+1] = f(x[i],y[i])
    y[i+1] = gety(y[i],x[i])


# In[137]:


plt.xlabel("n")
plt.ylabel("x_n")
plt.plot(range(N+1),x)
plt.show()

# In[143]:


sigma = -0.01
N=10000
x=np.zeros(N+1)
y=np.zeros(N+1)
for i in range(N):
    x[i+1] = f(x[i],y[i])
    y[i+1] = gety(y[i],x[i])


# In[144]:


plt.xlabel("n")
plt.ylabel("y_n")
plt.plot(range(N+1),x)
plt.show()

# In[146]:


alpha = 6
sigma = 0.01
N=10000
x=np.zeros(N+1)
y=np.zeros(N+1)
for i in range(N):
    x[i+1] = f(x[i],y[i])
    y[i+1] = gety(y[i],x[i])


# In[149]:


plt.xlabel("y_n")
plt.ylabel("x_n")
plt.scatter(y,x,s=1)
plt.show()

# In[ ]:




