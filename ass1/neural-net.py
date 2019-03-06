#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


x=np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
ones=np.array([1 for i in range(8)]).reshape(8,1)
x=np.append(ones,x,axis=1)
y_target=np.array([[1],[0],[0],[1],[1],[1],[0],[1]])

number_hidden=15
alpha=0.001

w1=np.random.randn(x.shape[1],number_hidden)
w2=np.random.randn(number_hidden,1)


# In[3]:


def fun(y):
    if(y>0.5):
        return 1
    else:
        return 0
fun=np.vectorize(fun)
def relu(x):
    if(x>0):
        return x
    else:
        return 0
def sigmoid(x):
    return 1/(1+np.exp(-x))
def grad_relu(x):
    if(x>0):
        return 1
    else:
        return 0
grad_relu=np.vectorize(grad_relu)
sigmoid= np.vectorize(sigmoid)
relu=np.vectorize(relu)


# In[7]:


print("Learning")
for i in range(10000):
    h=np.matmul(x,w1)
    a=relu(h)
    b=np.matmul(a,w2)
    y_pred=sigmoid(b)
    loss=-( (1-y_target)*np.log(1-y_pred) + y_target*np.log(y_pred) )
    grad_yp=-((y_target-1)/(1-y_pred)+y_target/y_pred)
    grad_b=grad_yp*sigmoid(b)*(1-sigmoid(b))
    grad_w2=np.matmul(a.T,grad_b)
    grad_a=np.matmul(grad_b,w2.T)
    grad_h=grad_a*grad_relu(a)
    grad_w1=np.matmul(x.T,grad_h)
    if(i%500==0):
        print("Epoch:", i, end=" ")
        print("Loss:" , (loss**2).sum())
    w1=w1-alpha*grad_w1
    w2=w2-alpha*grad_w2


# In[8]:


# Predicting using the learnt function
h=np.matmul(x,w1)
a=relu(h)
b=np.matmul(a,w2)
y_pred=sigmoid(b)
print(fun(y_pred))


# In[ ]:





# In[ ]:




