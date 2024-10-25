#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
import warnings
warnings.simplefilter('ignore')


# In[2]:


df=pd.read_csv("C:\\Users\\Masooma\\Downloads\\Churn_Modelling.csv")


# In[3]:


df.head()


# In[4]:


df.drop(columns=['RowNumber','CustomerId','Surname'],inplace=True)
df


# In[7]:


X=pd.get_dummies(df.drop('Exited',axis=1),drop_first=True)
y=df['Exited']


# In[8]:


X


# ### Train|Test split

# In[15]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[16]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[17]:


from keras.models import Sequential
from keras.layers import Dense

ann=Sequential()
ann.add(Dense(input_dim=11,units=21,kernel_initializer='uniform',activation='relu'))
ann.add(Dense(units=21,kernel_initializer='uniform',activation='relu'))
ann.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[19]:


ann.fit(X_train,y_train,batch_size=32,epochs=250)


# ### Prediction and Accuracy
# 

# In[20]:


ypred_train=ann.predict(X_train)
ypred_train=(ypred_train>=0.5)

ypred_test=ann.predict(X_test)
ypred_test=(ypred_test>=0.5)


# In[22]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[23]:


print('Train Accuracy: ',accuracy_score(y_train,ypred_train))
print('Test Accuracy: ',accuracy_score(y_test,ypred_test))


# In[24]:


confusion_matrix(y_test,ypred_test)


# ### CrossValidation

# In[25]:


pip install scikeras


# In[27]:


def cross_build_classifier():
    classifier=Sequential()
    classifier.add(Dense(input_dim=11,units=21,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=21,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

from scikeras.wrappers import KerasClassifier
classifier=KerasClassifier(cross_build_classifier,batch_size=32,epochs=100)


# In[28]:


from sklearn.model_selection import cross_val_score

cross_val_score(classifier,X_train,y_train,cv=5,scoring='accuracy').mean()


# In[ ]:


#trainCV: 0.8547499999999999 ----> Good model

