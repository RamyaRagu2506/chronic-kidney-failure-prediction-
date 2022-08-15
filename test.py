## We will work on the functions here and further update the other python files ###
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# read file 
stroke = pd.read_csv("Data\strokedata_raw.csv")
#print(stroke.head())

#null values identification 
na = stroke.isnull().sum()
#print(na)

# replace null values 
bmi_f = stroke.bmi[stroke.gender=='Female']
bmi_m = stroke.bmi[stroke.gender=='male']
req=stroke.gender.value_counts()
naf=bmi_f.isnull().sum()
nam=bmi_m.isnull().sum()
# print(naf)
# print(nam)

# female data has null values in bmi , so we replacee that with the mean values 
bmi_f1=stroke[stroke.gender=='Female']
mean= np.mean(bmi_f1.bmi)
# print(mean)
stroke.bmi= stroke.bmi.fillna(mean)
check = stroke.bmi.isnull().sum()
# print(check)

#latest data 
# print(stroke.head())
t=stroke.groupby(['smoking_status','work_type'])
print(t.sum())


# heat map 
# x = sns.heatmap(stroke)
# plt.show()

