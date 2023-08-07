#!/usr/bin/env python
# coding: utf-8

# In[123]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[86]:


df = pd.read_csv("Training.csv")


# In[87]:


df.head()


# In[88]:


df1 = pd.read_csv("Testing.csv")


# In[89]:


df1


# In[90]:


df.shape


# In[91]:


df.drop('Unnamed: 133', axis=1, inplace=True)
df.columns


# In[92]:


df['prognosis'].value_counts()


# In[93]:


df['prognosis'].nunique()


# In[94]:


x = df.drop('prognosis', axis = 1)
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state = 42)


# In[95]:


x


# In[96]:


y_test


# In[97]:


x_train


# In[98]:


tree = DecisionTreeClassifier()


# In[99]:


tree.fit(x_train, y_train)


# In[100]:


pred = tree.predict(x_test)


# In[101]:


df.iloc[1550]


# In[102]:


pred


# In[103]:


acc = tree.score(x_test, y_test)
acc


# In[20]:


df.columns


# In[21]:


fi = pd.DataFrame(tree.feature_importances_*100, x_test.columns, columns=['Importance'])
fi.sort_values(by = 'Importance', ascending = False, inplace = True)
fi


# In[27]:


zeros = np.array(fi[fi['Importance'] <= 2.300000].index)
zeros


# In[28]:


training_new = df.drop(columns=zeros, axis=1)
training_new.shape[1]
training_new.columns


# In[37]:


df.shape #Number of rows and columns before using feature importances 


# In[40]:


training_new.shape #Number of rows and columns after using feature importances


# In[104]:


def modelling(df1):
    x_new = df1.drop('prognosis', axis = 1)
    y_new = df1.prognosis
    x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_new, y_new, test_size=0.3, random_state=42) 
    #tree.fit(x_train_new, y_train_new)

    pred_new = tree.predict(x_test_new)
    
    acc_new = tree.score(x_test_new, y_test_new)
#     a = mean_absolute_error(y_test_new, pred_new)
    print("Acurray on test set: {:.2f}%".format(acc_new*100))
#     print("mean_absolute_error of the test set: {:.2f}%".format(a))


# In[111]:


test = pd.read_csv("Testing.csv")
test_new = test.drop(columns=zeros, axis=1)
test_new.shape[1]


# In[112]:


test_new.head()


# In[114]:


modelling(test)


# In[84]:


test.shape


# In[119]:


ran = RandomForestClassifier()
ran.fit(x_train, y_train)
ran.predict(x_test)


# In[121]:


ran.score(x_test, y_test)


# In[124]:


ada = AdaBoostClassifier()
ada.fit(x_train, y_train)
ada.predict(x_test)


# In[125]:


ada.score(x_test, y_test)


# In[ ]:




