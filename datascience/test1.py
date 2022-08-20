## We will work on the functions here and further update the other python files ###
from locale import normalize
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,Normalizer
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics 
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import RandomOverSampler,SMOTE
from collections import Counter 
import pickle

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

# #hyper parameters of grid search 
# n_estimators=[int(x) for x in np.linspace(start=100,stop=1000,num=10) ] 
# max_features= ['auto', 'sqrt']

# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)

# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]

# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]

# # Method of selecting samples for training each tree
# bootstrap = [True, False]

# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# # age ,hypertension, heart_disease,avg_glucose_level - key features 
key_features = stroke[['age','hypertension','heart_disease','avg_glucose_level']]
label = stroke.stroke.values

# split and train daataset
X_train, X_test, y_train, y_test = train_test_split(key_features, label, test_size=0.2, shuffle=True)


#normalisation 
nd = Normalizer()
X_train=nd.fit_transform(X_train)
X_test=nd.transform(X_test)
ros = SMOTE(random_state=40,k_neighbors= 10)
X_train_ros, y_train_ros= ros.fit_resample(X_train, y_train)
# Check the number of records after over sampling
print(sorted(Counter(y_train_ros).items()))

# # fit in model 
# cv = KFold(n_splits=10,shuffle=True,random_state=1) #cross validation with kfold method , split and shuffle data 
rfc= RandomForestClassifier(n_estimators= 200, min_samples_split= 10, min_samples_leaf= 4, max_features= 'auto', max_depth= 70, bootstrap= True)
# # rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model

# base_model = rfc.fit(X_train, y_train)
# base_prediction = rfc.predict(X_test)
# print(classification_report(y_test,base_prediction))

# # model = rfc.fit(X_train,y_train)
# # print(rf_random.best_params_,rf_random.best_score_)

super_model =  rfc.fit(X_train_ros,y_train_ros)
super_prediction = rfc.predict(X_test)
print(classification_report(y_test,super_prediction))

conf = confusion_matrix(y_test, super_prediction)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf, display_labels = [False, True])
# cm_display.plot()
# plt.show()

with open('stroke_model','wb') as files:
    pickle.dump(super_model,files) 


