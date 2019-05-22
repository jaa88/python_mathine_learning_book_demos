import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PATH= r'C:\Users\jianan\Desktop\ml_data\\'
r=requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
with open(PATH+'iris.data','w') as f:
    f.write(r.text)
os.chdir(PATH)
df=pd.read_csv(PATH+'iris.data',names=['sepal length','sepal width','petal length','petal width','class'])

print(df.groupby("petal width")["class"].unique().to_frame())

print(df.groupby("class")['petal width'].agg({'delta':lambda x:x.max() - x.min(),'max':np.max,"min":np.min}))

