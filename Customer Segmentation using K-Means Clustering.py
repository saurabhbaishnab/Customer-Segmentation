#!/usr/bin/env python
# coding: utf-8
### Importing the Dependencies
# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

### Data Collection & Analysis 
# In[16]:


# loading the data from csv file to a Pandas DataFrame
customer_details = pd.read_csv('/Users/saurabh453/Desktop/Customer Segmentation/customers_mall.csv')


# In[17]:


# first 5 rows in the dataframe
customer_details.head()


# In[18]:


# finding the number of rows and columns
customer_details.shape


# In[19]:


# getting some informations about the dataset
customer_details.info()


# In[20]:


# checking for missing values
customer_details.isnull().sum()

### Choosing the Annual Income column & Spending Score column.
# In[21]:


A = customer_details.iloc[:,[3,4]].values


# In[22]:


print(A)

### choosing the number of clusters

    WCSS -> within clusters sum of squares
# In[23]:


# finding wcss value for different number of clusters

wcss = []

for i in range(1,11):
  kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
  kmeans.fit(A)

  wcss.append(kmeans.inertia_)


# In[24]:


# plot an elbow graph

sns.set()
plt.plot(range(1,11), wcss)
plt.title('Elbow Point Graph')
plt.xlabel('No. of Clusters')
plt.ylabel('WCSS')
plt.show()

### Optimum number of clusters = 5

  Training the K-means clustering model
# In[25]:


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# return a label for each data point based on their cluster
B = kmeans.fit_predict(A)

print(B)

### 5 Clusters - 0, 1, 2, 3, 4

Visualizing all the Clusters
# In[26]:


# plotting all the clusters and their Centroids

plt.figure(figsize=(8,8))
plt.scatter(A[B==0,0], A[B==0,1], s=50, c='green', label='Cluster 1')
plt.scatter(A[B==1,0], A[B==1,1], s=50, c='red', label='Cluster 2')
plt.scatter(A[B==2,0], A[B==2,1], s=50, c='orange', label='Cluster 3')
plt.scatter(A[B==3,0], A[B==3,1], s=50, c='magenta', label='Cluster 4')
plt.scatter(A[B==4,0], A[B==4,1], s=50, c='blue', label='Cluster 5')

# plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, c='black', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()


# In[ ]:




