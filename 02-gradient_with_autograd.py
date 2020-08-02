#!/usr/bin/env python
# coding: utf-8

# In[1]:


##gradients are essential for model optmization


# In[2]:


import torch


# In[3]:


a = torch.randn(3)
print(a)


# In[4]:


b = torch.rand(3)
print(b)


# rand() returns random values between 0 and 1. The random values would follow a uniform distribution and hence the mean value would be 0.5
# 
# randn() returns random values between -infinity and +inifinity. The random values would follow a normal distribution with a mean value 0 and a standard deviation 1

# In[5]:


x = torch.randn(3, requires_grad=True) #-> default is False
print(x)


# In[6]:


y = x + 2
print(y)


# In[7]:


z_add = a + y
z_add


# In[8]:


z_mul = y*y*2
print(z_mul)


# In[9]:


z_mean = y.mean()
print(z_mean)


# #### Calculate the gradient
# ##### If the requires_grad is set to False then the .backward() shows error
# ##### Behind the scene during the .backward(), creates the vectorjacobian products(chain rule) and get the final gradients.
# #### NB: grad can be implicitly created only for scalar outputs

# In[10]:


z_mean.backward() #dz_mean/dx


# In[11]:


#x now has the .grad attribute that has all the gradients calculated
print(x.grad)


# ### Ways to prevent pytorch from tracking the gradients
#     * call the requires_grad(False)
#     * call the .detach()
#     * wrapping with torch.no_grad():

# In[12]:


print(x)
x.requires_grad_(False)
print(x)


# In[13]:


x = torch.randn(5, requires_grad=True)
print(x)
y = x.detach()
print(y)


# In[14]:


x = torch.randn(4, requires_grad=True)
print(x)
with torch.no_grad():
    y = x +2
    print(y)


# #### Very important -> gradients keep accumulating so keep closer look

# In[15]:


weights = torch.ones(5, requires_grad=True)
print(weights)


# In[16]:


for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)


# ##### As clearly, we see that the gradients keep on accumulating in each epoch, so must be very careful and zero the gradients after each epoch

# In[17]:


for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_()


# In[18]:


#this is how it looks in real trainig in pytorch
#dummy example


# In[19]:


get_ipython().system('jupyter nbconvert --execute 02-gradient_with_autograd.ipynb --to pdf')


# In[ ]:




