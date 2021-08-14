import pandas as pd
import numpy as np
from sklearn import preprocessing
import math
#from scipy.spatial import distance
EDistance=0
Data=pd.read_csv("football.csv",encoding = "ISO-8859-1")
Data.columns=Data.columns.str.upper()
del Data['PLAYER ID']
X=Data.drop(['FIRST_NAME','LAST_NAME'], axis=1)
y=Data.iloc[:,[7,8]]
X=preprocessing.scale(X)
X=pd.DataFrame(X)
Player1=X[y["FIRST_NAME"]=="Song"]
Player1=np.asarray(Player1)
#Distances=X.apply(lambda Row: distance.euclidean(Row, Player1), axis=1)
def function(a):
    a=np.asarray(a)
    temp=((Player1-a)*(Player1-a))
    temp=math.sqrt(temp[0][0]+temp[0][1]+temp[0][2]+temp[0][3]+temp[0][4]+temp[0][5]+temp[0][6])
    return temp
Distances=X.apply(lambda Row : function(Row), axis =1)
DistanceFrame = pd.DataFrame(data={ "Distance": Distances,"Index": Distances.index})
DistanceFrame =DistanceFrame.sort_values("Distance",ascending=True)
SubstituteLocation= DistanceFrame.iloc[1][1]
SubstituteFN = Data.loc[int(SubstituteLocation)][7]
SubstituteLN =Data.iloc[int(SubstituteLocation)][8]
print("The appropriate replacement will be",SubstituteFN, SubstituteLN)         