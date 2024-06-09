#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import requests
import os
from bs4 import BeautifulSoup


# In[17]:


dirc = "D:/Black Coffer/zdata/data/Balance Sheets/"
path = os.listdir(dirc)
for x in path:
    if x != ".DS_Store":
        print(x)


# In[18]:


# Path to the HTML file
file = "D:/Black Coffer/zdata/data/Balance Sheets/18320959_3.html"

# Read and parse the HTML file to extract tables
tables = pd.read_html(file)
tables[0]["category"]="Balance sheet"
tables[0]


# In[34]:


all_data = []
def storeindirec(file, statement):
    path = os.listdir(file)
    for x in path:
        if x != ".DS_Store":
            a = file+x
            t = pd.read_html(a)
            t[0]['Category']=statement
            all_data.append(t[0])
#file = "D:/Black Coffer/zdata/data/Balance Sheets/"
storeindirec("D:/Black Coffer/zdata/data/Balance Sheets/", "Balance Sheets")
storeindirec("D:/Black Coffer/zdata/data/Cash Flow/", "Cash Flow")
storeindirec("D:/Black Coffer/zdata/data/Income Statement/", "Income Statement")
storeindirec("D:/Black Coffer/zdata/data/Notes/", "Notes")
storeindirec("D:/Black Coffer/zdata/data/Others/", "Others")


# In[35]:


len(all_data)


# In[36]:


combined_data = pd.concat(all_data, ignore_index=True)


# In[64]:


combined_data['Category']


# In[56]:


encoded_data = combined_data


# In[66]:


from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Assuming encoded_data is your DataFrame with categorical columns
X = encoded_data.drop(columns=['Category'])
y = encoded_data['Category']  # 'Category' is the target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoostClassifier
classifier = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, random_seed=42, verbose=False)

# Train the model
classifier.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = classifier.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))


# In[ ]:





# In[ ]:




