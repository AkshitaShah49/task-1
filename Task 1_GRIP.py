#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


url="http://bit.ly/w-data"
data=pd.read_csv(url)
print("Data imported")
data.head(25)


# In[3]:


data.describe()


# In[3]:


# To Plot the Distribution of Scores
data.plot(x='Hours',y='Scores',style='+')
plt.title(' Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage score')
plt.show()


# In[4]:


#Displaying Regression plot
sns.regplot(x=data['Hours'],y=data['Scores'])
plt.title('Regression plot',size=15)
plt.ylabel('Marks',size=10)
plt.xlabel('Hours',size=10)
plt.show()
#Defining values
X=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# In[5]:


#Splitting Data into two
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)


# In[6]:


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train,y_train)
y_predict=regression.predict(X_train)
print(" Completion of training ")


# In[7]:


print('Test Score')
print(regression.score(X_test,y_test))
print('Training score')
print(regression.score(X_train,y_train))


# In[8]:



y_test


# In[9]:


y_predict


# In[10]:


y_predict[:5]


# In[11]:


prediction=pd.DataFrame({'Hours':[i[0] for i in X_train],'Predicted marks':[k for k in y_predict ]})
prediction


# In[12]:


data=pd.DataFrame({'Actual':y_test[:5],'Predicted':y_predict[:5]})
data


# In[15]:


mean_squ_error=mean_squared_error(y_test[:5],y_predict[:5])
mean_abs_error=mean_absolute_error(y_test[:5],y_predict[:5])
print(' Mean Square Error ',mean_squ_error)
print(' Mean absolute Error ',mean_abs_error)


# In[16]:


print('Score of student who studied for 9.25 hrs a day',regression.predict([[9.25]]))


# In[ ]:




