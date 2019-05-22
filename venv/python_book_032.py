# 通过支持向量机 构建线性模型，预测分类
import os
import pandas as pd
import requests
import numpy as np

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

PATH= r'C:\Users\jianan\Desktop\ml_data\\'
r=requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
with open(PATH+'iris.data','w') as f:
    f.write(r.text)
os.chdir(PATH)
df=pd.read_csv(PATH+'iris.data',names=['sepal length','sepal width','petal length','petal width','class'])

clf=OneVsRestClassifier(SVC(kernel='linear'))

X=df.ix[:,:4]
y=np.array(df.ix[:,4]).astype(str)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

rf=pd.DataFrame(list(zip(y_pred,y_test)),columns=["predicted","actual"])
rf["correct"]=rf.apply(lambda r:1 if r["predicted"]==r["actual"] else 0,axis=1)

print(rf)

print(rf["correct"].sum()/rf["correct"].count())







