## We will work on the functions here and further update the other python files ###
from locale import normalize
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,Normalizer
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter 
import pickle
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from xgboost import XGBClassifier
import xgboost as xgb 


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
#print(t.sum())

# string to object 
obj_list= [a for a in stroke.columns if stroke[a].dtype == object ]
num_list = [a for a in stroke.columns if stroke[a].dtype != object ]

le= LabelEncoder()
for col in obj_list:
  stroke[col] = le.fit_transform(stroke[col])
#print(stroke.info())
stroke = stroke.drop('id',axis = 1)
#print(stroke.head())
#plt.figure(figsize=(16,6))

# heat map 
x = stroke.corr()
# sns.heatmap(x,vmin=-1,vmax=1,annot=True)
# plt.show()

st1 = pd.DataFrame({'feature1':stroke.hypertension.values,'feature2':stroke.age.values,'target':stroke.stroke})
st1.target.value_counts(normalize=True)
# plt.figure(figsize=(12,8))
# sns.scatterplot(x='feature1',y='feature2',hue='target',data=st1)
# plt.show()

# # age ,hypertension, heart_disease,avg_glucose_level - key features 
key_features = stroke.drop('stroke',axis = 1 )
label = stroke.stroke.values

# split and train daataset
X_train, X_test, y_train, y_test = train_test_split(key_features, label, test_size=0.2, shuffle=True)


#normalisation 
nd = Normalizer()
X_train=nd.fit_transform(X_train)
X_test=nd.transform(X_test)
ros = NearMiss(version = 3)
X_train_ros, y_train_ros= ros.fit_resample(X_train, y_train)
# Check the number of records after over sampling
print(sorted(Counter(y_train_ros).items()))

# # fit in model 
# cv = KFold(n_splits=10,shuffle=True,random_state=1) #cross validation with kfold method , split and shuffle data 
rfc= xgb.XGBClassifier()

super_model =  rfc.fit(X_train_ros,y_train_ros)
super_prediction = rfc.predict(X_test)
print(classification_report(y_test,super_prediction))

conf = confusion_matrix(y_test, super_prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf, display_labels = [False, True])
cm_display.plot()
plt.show()

with open('stroke_model','wb') as files:
    pickle.dump(super_model,files) 