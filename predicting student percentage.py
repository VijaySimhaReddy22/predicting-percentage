#!/usr/bin/env python
# coding: utf-8

# # Graduate Rotational Internship Programme
# # Author : Kuchuru Vijay Simha Reddy
# # Predict the Percentage of student studying 9.25hrs/day

# In[16]:


# Importing required libraries
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 


# In[17]:


url='http://bit.ly/w-data'
data=pd.read_csv(url)
data.head()


# data.shape

# In[18]:


data.info()


# In[19]:


data.describe()


# In[20]:


data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Scores')
plt.xlabel('Hours')
plt.ylabel('scores')
plt.show()


# In[21]:


x=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# In[22]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)


# In[23]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()


# In[24]:


regressor.fit(xtrain,ytrain)


# In[25]:


Y= regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x,Y)
plt.show()


# In[26]:


print(xtest) #Testing data-In Hours
y_pred=regressor.predict(xtest) # predicting the scores


# In[27]:


print(y_pred)


# In[28]:


#comparing Actual vs predicted
df=pd.DataFrame({'Actual':ytest,'predicted':y_pred})
df


# In[29]:


#Let's predict for our value
Hours=9.25
Pred_value=regressor.predict([[Hours]])
print(Pred_value)


# In[30]:


from sklearn import metrics
ypred=regressor.predict(xtest)
print('Mean Absolute Error:',metrics.mean_absolute_error(ytest,ypred))


# In[31]:


print('Mean Squared Error:',metrics.mean_squared_error(ytest,ypred))

