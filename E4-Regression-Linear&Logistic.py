#!/usr/bin/env python
# coding: utf-8

# # Exercise 04
# 
# 
# # Part 1 - Linear Regression
# 
# Estimate a regression using the Income data
# 
# 
# ## Forecast of income
# 
# We'll be working with a dataset from US Census indome ([data dictionary](https://archive.ics.uci.edu/ml/datasets/Adult)).
# 
# Many businesses would like to personalize their offer based on customer’s income. High-income customers could be, for instance, exposed to premium products. As a customer’s income is not always explicitly known, predictive model could estimate income of a person based on other information.
# 
# Our goal is to create a predictive model that will be able to output an estimation of a person income.

# In[58]:


import pandas as pd
import numpy as np
import scipy as sc

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

# read the data and set the datetime as the index
income = pd.read_csv('https://github.com/albahnsen/PracticalMachineLearningClass/raw/master/datasets/income.csv.zip', index_col=0)

income.head()


# In[59]:


income.shape


# # Exercise 4.1 
# 
# What is the relation between the age and Income?
# 
# For a one percent increase in the Age how much the income increases?
# 
# Using sklearn estimate a linear regression and predict the income when the Age is 30 and 40 years

# In[60]:


income.plot(x='Age', y='Income', kind='scatter')


# In[61]:


np.corrcoef(income['Age'], income['Income'])


# # Relación
# La edad explica el ingreso en un 29%.
# 
# # Incremento de 1% en Age 
# Generamos la regresión para identificar la afectación de Income, si se incrementa Age en 1%

# In[62]:


# fit a linear regression model
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
feature_cols = ['Age']
X = income[feature_cols]
y = income.Income
linreg.fit(X, y)


# In[63]:


# make predictions for all values of X
income['Income_pred'] = linreg.predict(X)
income.head()


# In[64]:


# put the plots together
plt.scatter(income.Age, income.Income)
plt.plot(income.Age, income.Income_pred, color='red')
plt.xlabel('Income')
plt.ylabel('Age')


# In[111]:


linreg.intercept_ + linreg.coef_ * 1
test = np.array(1)
test = test.reshape(-1,1)
linreg.predict(test)
print(feature_cols, linreg.coef_)


# # Interpretación:
#  Al incrementar 1 unidad en 'Age' se asocia un incremento de 542.16 en 'Income'.

# # Age en 30:
#  La prediccion para una edad de 30 años es:

# In[112]:


linreg.predict([[30]])


# # Age en 40:
#  La prediccion para una edad de 40 años es:

# In[113]:


linreg.predict([[40]])


# # Exercise 4.2
# Evaluate the model using the MSE error cuadratico medio

# In[68]:


income.describe()


# In[97]:


a=np.array(income['Age'])
b=np.array(income['Income'])
plt.scatter(a,b, alpha=0.3)


#añadimos columna de 1s para termino independiente
a=np.array([np.ones(32561),a]).T

#ejecutamos la regresion por minimos
B=np.linalg.inv(a.T @ a) @a.T @ b

plt.plot([20,90],[B[0]+B[1]*20,B[0]+B[1]*90], c="red")
plt.show()


# In[95]:


B


# In[115]:


y_pred = linreg.predict(income['Age'])
mean_squared_error(income['Income'], y_pred)


# 
# # Exercise 4.3
# 
# Run a regression model using as features the Age and Age$^2$ using the OLS equations

# In[ ]:





# # Exercise 4.4
# 
# 
# Estimate a regression using more features.
# 
# How is the performance compared to using only the Age? MSE

# In[ ]:





# # Part 2: Logistic Regression
# 
# ### Customer Churn: 
# losing/attrition of the customers from the company. Especially, the industries that the user acquisition is costly, it is crucially important for one company to reduce and ideally make the customer churn to 0 to sustain their recurring revenue. If you consider customer retention is always cheaper than customer acquisition and generally depends on the data of the user(usage of the service or product), it poses a great/exciting/hard problem for machine learning.
# 
# ### Data
# Dataset is from a telecom service provider where they have the service usage(international plan, voicemail plan, usage in daytime, usage in evenings and nights and so on) and basic demographic information(state and area code) of the user. For labels, I have a single data point whether the customer is churned out or not.
# 

# In[98]:


# Download the dataset
data = pd.read_csv('https://github.com/ghuiber/churn/raw/master/data/churn.csv')


# In[99]:


data.head()


# # Exercise 4.5
# 
# Create Y and X
# 
# What is the distribution of the churners? cual es el % de churn
# 
# Split the data in train (70%) and test (30%)
# 

# In[100]:


#Creando Y y X
data.columns

#convertir Churn en 0  y 1

data['Churn?'] = np.where(data['Churn?']=='False.',0,1) 


# In[101]:


data.head()


# In[102]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(solver='liblinear',C=1e9)
feature_cols = ['Day Mins']
X = data[feature_cols]
y = data['Churn?']
logreg.fit(X, y)
data['Churn_pred_class'] = logreg.predict(X)


# In[103]:


data.head()


# # Exercise 4.6
# 
# Train a Logistic Regression using the training set and apply the algorithm to the testing set.

# In[105]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')

print(data.groupby('Churn?').size())


# In[110]:


data.describe()


# In[108]:


X=np.array(data.drop(['Churn?'],1))
y=np.array(data['C'])
X.shape


# In[ ]:


predictions = model.predict(X)
print(predictions)[0:5]


# In[ ]:


model.score(X,y)


# # Exercise 4.7
# 
# a) Create a confusion matrix using the prediction on the 30% set. sobre la parte test
# 
# b) Estimate the accuracy of the model in the 30% set. sobre la parte test
# 

# In[ ]:


validation_size=0.3
seed = 7
X_train, X_validation, Y_train, Y_validation=model_selection.train_test_split(X,y,test_size=validation size, random_state=seed)


# In[ ]:


name='Logistic Regression'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)


# In[ ]:


predictions = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions))


# In[ ]:


print(confusion_matrix(Y_validation, predictions))

