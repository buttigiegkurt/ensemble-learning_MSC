import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
from pandas import ExcelFile
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
import distython
from distython import HEOM
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

df = pd.read_excel(r'D:\offside\dataset_final_unbalanced.xlsx')
y=df.offside
dff=df[[col for col in df.columns if col != 'offside']]
X_train, X_test, y_train, y_test = train_test_split(dff, y, test_size=0.3)
rr=pd.concat([y_train ,X_train],axis=1)
sm = SMOTENC(categorical_features=[0])
X_res, y_res = sm.fit_resample(X_train, y_train)
X_res=pd.DataFrame(data=X_res)
X_res.columns=[col for col in df.columns if col != 'offside']
y_res=pd.DataFrame(data=y_res)
y_res.columns=['offside']
coco=pd.concat([y_res, X_res],axis=1)
coco2=pd.concat([y_test,X_test],axis=1)

coco['split'] = '1'
coco2['split'] = '0'
coco.split = coco.split.astype(int)
coco2.split = coco2.split.astype(int)
df_new = pd.concat([coco, coco2])

df_new.to_excel('dataset_final_balanced.xlsx',index=False)
