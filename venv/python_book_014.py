import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# %matplotlib inline

import numpy as np

PATH= r'C:\Users\jianan\Desktop\ml_data\\'
r=requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
with open(PATH+'iris.data','w') as f:
    f.write(r.text)
os.chdir(PATH)
df=pd.read_csv(PATH+'iris.data',names=['sepal length','sepal width','petal length','petal width','class'])

fig , ax =plt.subplots(2,2,figsize=(6,4))
ax[0][0].hist(df['petal width'],color='black')
ax[0][0].set_ylabel("Count",fontsize=12)
ax[0][0].set_xlabel("Width",fontsize=12)
ax[0][0].set_title("aaaa",fontsize=14,y=1.01)

ax[0][1].hist(df['petal length'],color='black')
ax[0][1].set_ylabel("Count",fontsize=12)
ax[0][1].set_xlabel("Width",fontsize=12)
ax[0][1].set_title("aaaa",fontsize=14,y=1.01)

ax[1][0].hist(df['sepal width'],color='black')
ax[1][0].set_ylabel("Count",fontsize=12)
ax[1][0].set_xlabel("Width",fontsize=12)
ax[1][0].set_title("aaaa",fontsize=14,y=1.01)

ax[1][1].hist(df['sepal length'],color='black')
ax[1][1].set_ylabel("Count",fontsize=12)
ax[1][1].set_xlabel("Width",fontsize=12)
ax[1][1].set_title("aaaa",fontsize=14,y=1.01)

plt.legend()
plt.show()