import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn as skl

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

PATH= r'C:\Users\jianan\Desktop\ml_data\\'
r=requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
with open(PATH+'iris.data','w') as f:
    f.write(r.text)
os.chdir(PATH)
df=pd.read_csv(PATH+'iris.data',names=['sepal length','sepal width','petal length','petal width','class'])

clf=RandomForestClassifier(max_depth=5,n_estimators=10)

X=df.ix[:,:4]
Y=df.ix[:,4]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.3)

clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)

rf=pd.DataFrame(list(zip(Y_pred,Y_test)),columns=["predicted","actual"])
rf["correct"]=rf.apply(lambda r:1 if r["predicted"]==r["actual"] else 0,axis=1)

print(rf)

print(rf["correct"].sum()/rf["correct"].count())





