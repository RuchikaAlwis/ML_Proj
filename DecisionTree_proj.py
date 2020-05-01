import pandas as pd
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from scipy import stats
import numpy as np






# Making a list of missing value types with getting csv
missing_values = ["?"]

#read valuesof the training set
df = pd.read_csv("data.csv", na_values = missing_values)

#read values of the test set
header_list = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16"]
tst = pd.read_csv("testdata_10%.csv",names=header_list, na_values = missing_values)


#map with numeric values

#A1
a1 = {'a':0,'b':1}
df['A1'] = df['A1'].map(a1)
tst['A1'] = tst['A1'].map(a1)

#A3
a3 = {'u':0,'y':1,'l':2}
df['A3'] = df['A3'].map(a3)
tst['A3'] = tst['A3'].map(a3)

#A4
a4 = {'g':0,'p':1,'gg':2}
df['A4'] = df['A4'].map(a4)
tst['A4'] = tst['A4'].map(a1)

#A6
a6 = {'w':0,'q':1,'c':2,'x':3,'i':4,'d':5,'e':6,'aa':7,'cc':8,'ff':9,'m':10,'k':11,'j':12,'r':13}
df['A6'] = df['A6'].map(a6)
tst['A6'] = tst['A6'].map(a6)

#A8
a8 = {True:0,False:1}
df['A8'] = df['A8'].map(a8)
tst['A8'] = tst['A8'].map(a8)

#A9
a9 = {'v':0,'h':1,'bb':2,'ff':3,'j':4,'z':5,'o':6,'dd':7,'n':8}
df['A9'] = df['A9'].map(a9)
tst['A9'] = tst['A9'].map(a9)

#A11
a11 = {True:0,False:1}
df['A11'] = df['A11'].map(a11)
tst['A11'] = tst['A11'].map(a11)

#A13
a13 = {True:0,False:1}
df['A13'] = df['A13'].map(a13)
tst['A13'] = tst['A13'].map(a13)

#A15
a15 = {'g':0,'s':1,'p':2}
df['A15'] = df['A15'].map(a15)
tst['A15'] = tst['A15'].map(a15)

#A16
a16 = {'Success':0,'Failure':1}
df['A16'] = df['A16'].map(a16)

#Replacing NaN values with mode and mean

#Replace Nan values with mode
f = ["A1","A3","A4","A6","A8","A9","A11","A13","A15","A16"]

for x in f:
   mode_value1=df[x].mode()
   df[x]=df[x].fillna(mode_value1[0])
   tst[x]=tst[x].fillna(mode_value1[0])


#Replace Nan values with mean
ff = ["A2","A5","A7","A10","A12","A14"]
for x in ff:
    mean_value1=df[x].mean()
    df[x]=df[x].fillna(mean_value1)
    tst[x]=tst[x].fillna(mean_value1)

#select the features and y
features = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15"]
X = df[features]
y = df["A16"]

#Test set X
X_tst=tst[features]

#Devideing the training set and test set for the modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#Dec algorithem
clf = DecisionTreeClassifier(criterion = 'entropy')

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_tstpred=clf.predict(X_tst)
y_pred=clf.predict(X_test)

print(y_tstpred)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
