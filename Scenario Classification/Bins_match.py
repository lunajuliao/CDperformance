import numpy as np
import pandas as pd
import math
from math import sin, cos, atan2, sqrt
import matplotlib.pyplot as plt

import pickle
# Match data from traffic density and weather bins
var=open('TD.pickle','rb') # info from traffic density
TD=pickle.load(var)
var.close()
var=open('Weather.pickle','rb') # info from traffic density
Cluster=pickle.load(var)
var.close()
Match=pd.DataFrame([])
Match['Date']=TD['Date']
Match['Meteo']=Cluster
Match['TrafficDensity']=TD[1]
Match['TrafficDensity'].replace({1: "L", 2: "M", 0: "H"}, inplace=True) # SHOULD BE CHANGED TO MATCH RESULTS 

# Pick randomly 4 data sets for each data categorizaion type
B=['L','M','H']
subset=[]
for i in range(len(B)):
    for j in range(len(B)):
        mask1 = Match['Meteo']==B[i]
        mask2 = Match['TrafficDensity'] == B[j]
        mask=np.logical_and(mask1, mask2)
        sub=Match.loc[mask]
        x=B[i]+B[j]
        print(len(sub))
        print(x)
        sub=sub.sample(n=4)
        subset.append(sub)

