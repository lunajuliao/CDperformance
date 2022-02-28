from netCDF4 import Dataset
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import math
from math import sin, cos, atan2, sqrt
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans
import seaborn as sns
import metpy
#
# Function coriolis: Obtain the coriolis parameter according to latitude 
#
def coriolis(lat):
    omega=2* math.pi / (24*3600)
    deg2rad=math.pi /180
    c= 2*omega*sin(lat*deg2rad)
    return c

#
# Function getD: Obtain distance between two horizontal coordinates
#
def getD(lat1,lon1,lat2,lon2):
  R = 6372 # Radius of the earth in km + h at 900hPa
  dLat = deg2rad(lat2-lat1)
  dLon = deg2rad(lon2-lon1)
  a =sin(dLat/2) * sin(dLat/2) + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dLon/2) * sin(dLon/2)
  c = 2 * atan2(sqrt(a), sqrt(1-a))
  d = R * c#  Distance in km
  return d
#
# Funtion deg2rad: Transforms variables in degrees to radians
#
def deg2rad(deg):
  return deg * (math.pi/180)


data = Dataset("900hpa.nc", "r", format="NETCDF4") # upload data obtained from ERA5

# Assign each component of the data to variables
lat= data.variables['latitude'][:]
lon= data.variables['longitude'][:]
vr = data.variables['vo'][:]
t = data.variables['t'][:]

time = data.variables['time'][:]

# Re organize the data
vr_t = np.concatenate(vr)
t_t = np.concatenate(t)
vr_tt = np.concatenate(np.concatenate(vr))
t_tt = np.concatenate(np.concatenate(t))

dataframe_hour=pd.DataFrame([], columns=['Lat','Lon'])
p_hour=len(lat)*len(lon)
for i in range(len(lat)):
    for j in range(len(lon)):
        print(i,j)
        coord=pd.DataFrame([[lat[i],lon[j]]], columns=['Lat','Lon'])
        dataframe_hour=dataframe_hour.append(coord, ignore_index=True)
        print('here')

# Auxiliar parameters calculation
dataframe_hour['Normalization']=[coriolis(dataframe_hour.iat[i,0])*0.0045 for i in range(dataframe_hour.shape[0])]  # assign the time parameter for easier identification further in the main function

dataframe_total=pd.concat([dataframe_hour]*13140, ignore_index=True) # create the longitude and latitude for all the vorticity and temperature values
dataframe_total['Seq']=[i for i in range(dataframe_total.shape[0])]
dataframe_total['Vorticity'] = pd.DataFrame(vr_tt)
dataframe_total['Temperature'] = pd.DataFrame(t_tt)

#%%
# Calculate temperature gradient
# Calculate in km the distance between two grid points, considering the values do NOT figure a 2D flat grid
Lon_step=pd.DataFrame([getD(lat[i],1,lat[i],1.25) for i in range(len(lat))])
Lat_step=pd.DataFrame([getD(55,lon[i],54.75,lon[i]) for i in range(len(lon))])

# Due to data organization, one component of the temperature gradient is almost direct as below
DT_lon = np.array_split(dataframe_total, 381060)
data_t=[]
for i in range(len(DT_lon)):
    print(i)
    aux=DT_lon[i] # data p/ hour p/ latitude
    step = Lon_step.iat[(lat == aux.iat[0, 0]).tolist().index(True),0]  # in km
    temp=aux['Temperature'].to_numpy()
    aux['GradTemp']=np.gradient(temp, step)
    data_t.append(aux)
data_t=pd.concat(data_t)

# For the second horizontal component, it is necessary to re-organize the data and then go through the same process
data_tt=[]
DT_lat = np.array_split(data_t, 13140)
for i in range(len(DT_lat)):
    aux=DT_lat[i] # data p/ hour
    aux=aux.sort_values(by=['Lon'])
    DT_lat_aux = np.array_split(aux, 29)
    for j in range(len(DT_lat_aux)):
        #print(i,j)
        aux_aux=DT_lat_aux[j]
        aux_aux = aux_aux.sort_values(by=['Lat'])
        step = Lat_step.iat[(lon == aux_aux.iat[0, 1]).tolist().index(True),0]  # in km
        temp=aux_aux['Temperature'].to_numpy()
        aux_aux['GradTemp2']=np.gradient(temp, step)
        data_tt.append(aux_aux)

data_tt=pd.concat(data_tt, ignore_index=True)
data_tt=data_tt.sort_values(by=['Seq'])
# Organize the data in 6-hour fragments, to have a better grasp of each simulation  data characteristics
DT_d = np.array_split(data_tt, 2190) # n of days*2 since the 12 hours p/ day are divided in 2 fragments
d_aux=[]
for i in range(len(DT_d)):
    aux=DT_d[i]
    aux['Date']=[i for j in range(len(aux))]
    d_aux.append(aux)
data_tt=pd.concat(d_aux, ignore_index=True)

#%%
# Compute the non dimensional main parameter
data_tt['TotalGrad']=np.sqrt(data_tt['GradTemp']*data_tt['GradTemp'] + data_tt['GradTemp2']*data_tt['GradTemp2'])
f=data_tt['Vorticity']*(np.sqrt(data_tt['GradTemp']*data_tt['GradTemp'] + data_tt['GradTemp2']*data_tt['GradTemp2']))
F=f/data_tt['Normalization']
data_tt['F']=F

# Calculate the bins with relative parameters obtained from observation
D_H=[]
D_M=[]
D_L=[]
DT_F = np.array_split(data_tt['F'], 13140)
for i in range(len(DT_F)):
    aux_F=DT_F[i] # data p/ hour p/ latitude
    VarH=sum(i > 20 for i in aux_F)
    VarM=sum(i > 12 for i in aux_F)
    VarL = sum(i > 4 for i in aux_F)
    VarM=VarM-VarH
    VarL=VarL-VarM
    D_L.append(VarL)
    D_M.append(VarM)
    D_H.append(VarH)

D_L=np.array_split(D_L, len(D_L)/6)
D_M=np.array_split(D_M, len(D_M)/6)
D_H=np.array_split(D_H, len(D_H)/6)
Cluster=[]
L=0;M=0;H=0;O=0
for i in range(len(D_L)):
    ML = max(D_L[i])
    MM = max(D_M[i])
    MH = max(D_H[i])
    if (MM<=2 and MH==0 and ML<=8):
        L=L+1
        Cluster.append('L')
    elif (MH<=2 or (MH<=4 and MM>0)):
        M=M+1
        Cluster.append('M')
    elif (MH >= 5):
        H=H+1
        Cluster.append('H')
    else:
        O=O+1
        print(ML,MM,MH)


import pickle
var=open('Weather.pickle', 'wb')
pickle.dump(Cluster, var)
var.close()
