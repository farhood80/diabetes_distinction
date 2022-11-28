#!/usr/bin/env python
# coding: utf-8

# <b> Importing the dependencies<b>
#     

# In[58]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# <b> Data collection and Analysis<b>
#     
#     PIMA Diabetes Dataset(its only for females)

# In[71]:


#loading the Daiabetes Database into pandas DataFrame

Diabetes_Dataset = pd.read_csv("/home/farhood/Desktop/diabetes.csv")


# In[72]:


pd.read_csv? #if you want to see all the parameters


# In[73]:


# printing the first 5 rows of the Dataset
Diabetes_Dataset.head()


# In[74]:


# numer of rows and columns in this dataset

Diabetes_Dataset.shape


# In[75]:


# getting the statistical measures of the data
Diabetes_Dataset.describe()


# In[76]:


# getting the number of diabetes cases and non diabetes cases
Diabetes_Dataset['Outcome'].value_counts() # also you can use Diabetes_Dataset['Outcome'].value_counts()


# 0 == Non Diabetes
# 
# 1 == Diabetes

# In[78]:


Diabetes_Dataset.groupby('Outcome').mean()


# In[85]:


# separating data and labels
x =Diabetes_Dataset.drop(columns = 'Outcome', axis =1)
y =Diabetes_Dataset['Outcome']
print(x)


# In[86]:


print(y)


# <b> Data Standardization<b>

# In[87]:


scaler = StandardScaler()


# In[90]:


scaler.fit(x)


# In[92]:


standardized_data = scaler.transform(x)


# In[93]:


print(standardized_data)


# In[94]:


x = standardized_data
y = Diabetes_Dataset['Outcome']


# In[95]:


print(x)
print(y)


# <b> Train Test Split<b>

# In[98]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, stratify=y, random_state=2)


# In[99]:


print(x.shape, x_train.shape, x_test.shape)


# <b> Training the Model<b>

# In[100]:


classifier = svm.SVC(kernel = 'linear')


# In[102]:


#training the svm(support vector machine classifier)
classifier.fit(x_train, y_train)


# <b> Model Evaluation <b>
#     
#  <b> Accuracy of the Model<b>   

# In[106]:


#accuracy of the Model on the training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)


# In[107]:


print("the accuracy of the training data is : ",training_data_accuracy)


# In[108]:


#accuracy of the Model on the test data
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)


# In[109]:


print("the accuracy of the test data is : ",test_data_accuracy)


# <b> Creating a Predictive System<b>

# In[115]:


input_data=(7,160,97,0,0,37.6,0.195,0)

#change the entered data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshaping the array as we are predicting 
input_reshaped_data = input_data_as_numpy_array.reshape(1,-1)

# standardized the input data

std_data = scaler.transform(input_reshaped_data)
print(std_data)

prediction = classifier.predict(std_data)
print("the result is: ",prediction)

if (prediction == 1):
    print("paitent has Diabetes!")
else:
    print("paitent doesnt have Diabetes!")


# In[ ]:




