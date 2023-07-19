#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from sklearn.preprocessing import StandardScaler


# In[2]:


df=pd.read_csv("C:\\Users\\Dipesh Singh\\Downloads\\Loan_approval_Training_Dataset.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df=df.drop(["Gender","Dependents","Self_Employed","Loan_Amount_Term","Credit_History","Loan_ID"],axis=1)


# In[8]:


df


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# In[11]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[12]:


df1=["Married","Education","Property_Area","Loan_Status"]
for col in df1:
    df[col]=le.fit_transform(df[col])
df


# In[13]:


df.isnull().sum()


# In[14]:


df=df.dropna(axis=0)


# In[15]:


df.isnull().sum()


# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


scaler=StandardScaler()


# In[18]:


scaler.fit(df)
df2=scaler.transform(df)


# In[19]:


df2=pd.DataFrame(df2)
df2


# In[20]:


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2,init='k-means++')


# In[21]:


kmeans.fit(df2)
kmeans.inertia_


# In[22]:


sse=[]
for i in range(1,40):
    kmeans=KMeans(n_clusters=i,init='k-means++')
    kmeans.fit(df2)
    sse.append(kmeans.inertia_)
frame = pd.DataFrame({'Cluster':range(1,40), 'SSE':sse})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')


# In[23]:


kmeans=KMeans(n_clusters=12,init='k-means++')
kmeans.fit(df2)


# In[24]:


kmeans.inertia_


# In[25]:


pred=kmeans.predict(df2)
pred


# In[26]:


frame=pd.DataFrame(df2)
frame['cluster']=pred
frame['cluster'].value_counts()


# In[ ]:




