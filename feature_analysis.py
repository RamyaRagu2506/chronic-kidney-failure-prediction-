import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# To read the file 
kidney= pd.read_csv('new_model.csv')
print(kidney.shape)
#print(kidney.head)

# To identify null values 
k1 = kidney.isnull().sum()
print(k1)

# heap map creation 
corr_map=sns.heatmap(kidney)
plt.show()

