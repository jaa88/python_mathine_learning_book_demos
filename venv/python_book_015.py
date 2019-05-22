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

fig , ax =plt.subplots(figsize=(6,6))

# 散点图
# ax.scatter(df['petal width'],df['petal length'],color='green')
# ax.set_xlabel('Petal Width')
# ax.set_ylabel("Petal Length")
# ax.set_title("jianangh")

# 线图
ax.plot(df['petal length'],color='green')
ax.set_xlabel('Specimen Number')
ax.set_ylabel("Petal Length")
ax.set_title("jianangh")


plt.legend()
plt.show()