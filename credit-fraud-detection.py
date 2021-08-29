
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')


# In[4]:


data.head()


# In[5]:


data.info()


# In[6]:


data['Class'].value_counts()


# In[9]:


# separating the data for analysis
legit = data[data.Class == 0]
fraud = data[data.Class == 1]


# In[10]:


legit_sample = legit.sample(n=492) # Handlimg imbalanced data


# In[11]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[12]:


new_dataset['Class'].value_counts()


# In[13]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[16]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[17]:


print(X.shape, X_train.shape, X_test.shape)


# In[18]:


model = LogisticRegression()


# In[20]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# In[21]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[22]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[25]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)

