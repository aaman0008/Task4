#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df=pd.read_csv("desktop/advertising.csv")


# In[5]:


df.head()


# In[9]:


df.shape


# In[7]:


df.describe()


# In[11]:


sns.pairplot(df,x_vars=['TV','Radio','Newspaper'],y_vars='Sales', kind="scatter")


# In[12]:


df['TV'].plot.hist(bins=10)


# In[13]:


df['Radio'].plot.hist(bins=10, color='green', xlabel='Radio')


# In[14]:


df['Newspaper'].plot.hist(bins=10, color='purple', xlabel='Newspaper')


# In[15]:


sns.heatmap(df.corr(),annot=True)
plt.show


# In[20]:


from sklearn.model_selection import train_test_split
X_train , X_test, Y_train , Y_test=train_test_split(df[['TV']],df[['Sales']],test_size=0.3,random_state=0)


# In[21]:


print(X_train)


# In[22]:


print(Y_train)


# In[23]:


print(X_test)


# In[25]:


print(Y_test)


# In[27]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train,Y_train)


# In[29]:


res=model.predict(X_test)
print(res)


# In[30]:


model.coef_


# In[31]:


model.intercept_


# In[32]:


plt.plot(res)


# In[33]:


plt.scatter(X_test,Y_test)
plt.show()


# In[ ]:




