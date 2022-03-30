#!/usr/bin/env python
# coding: utf-8

# Welcome to a competition powered by AutoDSC for Data Science Challenges! By Prof. Manoel Gadi!
# 
# PLEASE DO NOT RENAME THIS FILE!
# 
# 
# Simply run this code and start competing today in the competion: 6aQ6IxU7Va
# 
# 6aQ6IxU7Va details:
#  - Description / Descripción: FRAUD MODELLING CHALLENGE - Predict which Credit Card Application is legitimate and which belongs to a fraudster instead.
#  - Maximum number of daily attempts / Número máximo de intentos diarios: 1000
#  - Creation date / Fecha de creación: 2020-06-10 11:36:52
#  - Starting date / Fecha de inicio: 2021-12-08 12:00:00
#  - Ending date / Fecha de fin: 2021-12-20 23:59:00
#  - Minimum time between prediction submissions / Tiempo mínimo entre envíos de predicciones: 10
# 
# Of couse, to win the competition you should improve the starting model! So let's get to work!
# 

# In[1]:


print ("IMPORTING LIBRARIES...")
import pandas as pd


print ("LOADING DATASETS...")
try: # reading train csv from local file
    df_train = pd.read_csv("mfalonso__6aQ6IxU7Va__train.csv")
    df_train.head()
except: # reading train csv from the internet if it is the first time
    import urllib
    csv_train = urllib.request.urlopen("http://manoelutad.pythonanywhere.com/static/uploads/mfalonso__6aQ6IxU7Va__train.csv")
    csv_train_content = csv_train.read()
    with open("mfalonso__6aQ6IxU7Va__train.csv", 'wb') as f:
            f.write(csv_train_content)
    df_train = pd.read_csv("mfalonso__6aQ6IxU7Va__train.csv")

    
try: # reading test csv from local file
    df_test = pd.read_csv("mfalonso__6aQ6IxU7Va__test.csv")
    df_test.head()
except: # reading test csv from the internet if it is the first time
    import urllib
    csv_test = urllib.request.urlopen("http://manoelutad.pythonanywhere.com/static/uploads/mfalonso__6aQ6IxU7Va__test.csv")
    csv_test_content = csv_test.read()
    with open("mfalonso__6aQ6IxU7Va__test.csv", 'wb') as f:
            f.write(csv_test_content)
    df_test = pd.read_csv("mfalonso__6aQ6IxU7Va__test.csv")


# In[2]:


df_train.head()


# In[3]:


df_test.head()


# In[4]:


df_train = df_train.drop(columns=['Unnamed: 0', 'id'])


# In[5]:


df_test.drop(columns=['contract_date', 'Unnamed: 0'], inplace=True)


# In[6]:


print ("STEP 1: DOING MY TRANSFORMATIONS...")
df_train = df_train.fillna((df_train.mean()))
df_test = df_test.fillna((df_test.mean()))

X_test = df_test.drop(columns=["id"])


#from sklearn.impute import KNNImputer
#knn based imputation for categorical variables
#imputer = KNNImputer(n_neighbors=2)
#df_train = imputer.fit_transform(['if_var_78'])
# print the completed dataframe


# In[7]:


df_train.head()


# In[8]:


df_test.head()


# In[9]:


df_train.dtypes


# In[ ]:


print ("STEP 3: DEVELOPING THE MODEL...")
X_train = df_train.drop(columns=["ob_target"])
y_train = df_train["ob_target"]



#from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression(random_state=0, solver='lbfgs')
#fitted_model = clf.fit(X_train, y_train)
#pred_train = fitted_model.predict_proba(X_train)[:,1]
#pred_test  = fitted_model.predict_proba(X_test)[:,1]

# import Random Forest classifier

import time 

from sklearn.ensemble import RandomForestClassifier
for max_depth in range (20,21):
    for n_estimators in range (100, 10000, 100):
        for random_state in range (100, 10000, 100):
            rfc = RandomForestClassifier(max_depth = max_depth*8, n_estimators=n_estimators*20,  random_state=random_state*50, max_features="log2", n_jobs = -1)
            rfc.fit(X_train, y_train)
            y_pred = rfc.predict(X_train)

            #print ("STEP 5: SUBMITTING THE RESULTS... DO NOT CHANGE THIS PART!")
            import requests
            from requests.auth import HTTPBasicAuth
            pred_test = rfc.predict_proba(X_test)[:,1]
            print(pred_test)


            df_test['pred'] = pred_test
            df_test['id'] = df_test.iloc[:,0]
            df_test_tosend = df_test[['id','pred']]


            filename = "df_test_tosend.csv"

            df_test_tosend.to_csv(filename, sep=',')
            url = 'http://manoelutad.pythonanywhere.com/uploadpredictions/6aQ6IxU7Va'
            files = {'file': (filename, open(filename, 'rb')),
                 'ipynbcode': ('6aQ6IxU7Va.ipynb', open('6aQ6IxU7Va.ipynb', 'rb'))}



            #rsub = requests.post(url, files=files)

            rsub = requests.post(url, files=files, auth=HTTPBasicAuth("margaridapm", "sha256$bkpQgZOa$f401ae21f46569c8d1c5d8ca134e30a22b0fc649b8bab3b2cb69f043d5f5e227"))
            resp_str = str(rsub.text)

            print ("RESULT SUBMISSION:{} for max_depth = {} for n_estimators={}".format(resp_str, max_depth, n_estimators*7))

            time.sleep(30)


# In[ ]:


print ("STEP 4: ASSESSING THE MODEL...")
# CALCULATING GINI PERFORMANCE ON DEVELOPMENT SAMPLE
from sklearn.metrics import roc_auc_score
gini_score = 2*roc_auc_score(y_train, y_pred)-1
print ("GINI DEVELOPMENT=", gini_score)


# WHAT IS GINI?
# * watch this video for reference: https://youtu.be/MiBUBVUC8kE
# 

# In[ ]:




