import numpy as np 
import pandas as pd
Data=pd.read_csv("boston.csv")
X=Data.iloc[:,[2,3,4,5,6,12]].values
y=Data.iloc[:,13].values
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=(4/5))
bins=[-10,0,10,20,30,40,50,60]
bin_names= ["A","B","C","D","E","F","G"]
example=np.array([3,0,0.6,6.8,50,5])
example=example.reshape(1,-1)
y_test=pd.DataFrame(y_test)
y_test['RANGES']=pd.cut(y_test[0],bins,labels=bin_names)
y_testFinal=pd.get_dummies(y_test.iloc[:,1])


#Linear Regression 
from sklearn.linear_model import LinearRegression
LRegression=LinearRegression()
LRegression.fit(X_train,y_train) 
y_LR=LRegression.predict(X_test)  
y_LR=pd.DataFrame(y_LR)
y_LR['RANGES']=pd.cut(y_LR[0],bins,labels=bin_names)
y_predict=pd.get_dummies(y_LR.iloc[:,1])
prediction=LRegression.predict(example)
array1=confusion_matrix(y_testFinal.values.argmax(axis=0), y_predict.values.argmax(axis=0))
print("\nLinear Regression")
print('accuracy=', ((array1[0][0]+array1[1][1]+array1[2][2]+array1[3][3]+array1[4][4])/102)) 
print(prediction)

#Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor3= DecisionTreeRegressor()
regressor3.fit(X_train,y_train)     
y_DTR=regressor3.predict(X_test)    
prediction=regressor3.predict(example)
y_DTR=pd.DataFrame(y_DTR)
y_DTR['RANGES']=pd.cut(y_DTR[0],bins,labels=bin_names)
y_predict=pd.get_dummies(y_DTR.iloc[:,1],drop_first=True)
array=confusion_matrix(y_testFinal.values.argmax(axis=1), y_predict.values.argmax(axis=1))
print("\nDecision Tree")
print('accuracy=', ((array[0][0]+array[1][1]+array[2][2]+array[3][3]+array[4][4])/506))
print(prediction)

#Random Forest Generator
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300)
regressor.fit(X_train,y_train)
y_RFG=regressor.predict(X_test)    
prediction=regressor.predict(example)
y_RFG=pd.DataFrame(y_RFG)
y_RFG['RANGES']=pd.cut(y_RFG[0],bins,labels=bin_names)
y_predict=pd.get_dummies(y_RFG.iloc[:,1],drop_first=True)
array=confusion_matrix(y_testFinal.values.argmax(axis=1), y_predict.values.argmax(axis=1))
print("\nRandom Forest Generator")
print('accuracy=', ((array[0][0]+array[1][1]+array[2][2]+array[3][3]+array[4][4])/506))
print(prediction)

#Extra Tree Generator
from sklearn.ensemble import ExtraTreesRegressor
regressor4= ExtraTreesRegressor(n_estimators=300)
regressor4.fit(X_train,y_train)
y_ETG=regressor4.predict(X_test)    
prediction=regressor4.predict(example)
y_ETG=pd.DataFrame(y_ETG)
y_ETG['RANGES']=pd.cut(y_ETG[0],bins,labels=bin_names)
y_predict=pd.get_dummies(y_ETG.iloc[:,1],drop_first=True)
array=confusion_matrix(y_testFinal.values.argmax(axis=1), y_predict.values.argmax(axis=1))
print("\nExtra Tree Generator")
print('accuracy=', ((array[0][0]+array[1][1]+array[2][2]+array[3][3]+array[4][4])/506))
print(prediction)

#SVM
from sklearn.svm import SVR
regressor2=SVR()  
regressor2.fit(X_train,y_train)    
y_SVM=regressor2.predict(X_test)    
prediction=regressor2.predict(example)
y_SVM=pd.DataFrame(y_SVM)
y_SVM['RANGES']=pd.cut(y_SVM[0],bins,labels=bin_names)
y_predict=pd.get_dummies(y_SVM.iloc[:,1],drop_first=True)
array=confusion_matrix(y_testFinal.values.argmax(axis=1), y_predict.values.argmax(axis=1))
print("\nSupport Vector Machine Regression")
print('accuracy=', ((array[0][0]+array[1][1]+array[2][2]+array[3][3]+array[4][4])/506))
print(prediction)



y_testFinal_list=y_testFinal.values.tolist()
y_predict_list=y_predict.values.tolist()

def perf_measure(y_testFinal, y_predict):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for i in range(len(y_predict)): 
        if y_testFinal[i]==y_predict[i]==1:
           TP += 1
        if y_predict[i]==1 and y_testFinal[i]!=y_predict[i]:
           FP += 1
        if y_testFinal[i]==y_predict[i]==0:
           TN += 1
        if y_predict[i]==0 and y_testFinal[i]!=y_predict[i]:
           FN += 1

    return (TP, FP, TN, FN)
    
    
perf_measure(y_testFinal_list, y_predict_list)    
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

