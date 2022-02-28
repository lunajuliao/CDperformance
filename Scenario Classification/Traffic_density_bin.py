import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
import os

# Load the data
csv7 = pd.read_csv(r'ert_dly_fir_2017.csv')
csv8 = pd.read_csv(r'ert_dly_fir_2018.csv')
csv9 = pd.read_csv(r'ert_dly_fir_2019.csv')

# Organize the data and select only aircraft that go over the airspace of Belgium and Netherlands - matching the area of simulation
data7 = csv7[['YEAR', 'MONTH_NUM', 'FLT_DATE', 'ENTITY_NAME', 'FLT_ERT_1']]
data7['FLT_DATE']=data7['FLT_DATE'].astype(str).str[8:10]
data7 = data7.loc[(data7['ENTITY_NAME'] == 'Belgium') | (data7['ENTITY_NAME'] == 'Netherlands')]

data8 = csv8[['YEAR', 'MONTH_NUM', 'FLT_DATE', 'ENTITY_NAME', 'FLT_ERT_1']]
data8['FLT_DATE']=data8['FLT_DATE'].astype(str).str[8:10]
data8 = data8.loc[(data8['ENTITY_NAME'] == 'Belgium') | (data8['ENTITY_NAME'] == 'Netherlands')]

data9 = csv9[['YEAR', 'MONTH_NUM', 'FLT_DATE', 'ENTITY_NAME', 'FLT_ERT_1']]
data9['FLT_DATE']=data9['FLT_DATE'].astype(str).str[8:10]
data9 = data9.loc[(data9['ENTITY_NAME'] == 'Belgium') | (data9['ENTITY_NAME'] == 'Netherlands')]

month_day7=data7.groupby(['MONTH_NUM','FLT_DATE']).sum()['FLT_ERT_1']
month_day8=data8.groupby(['MONTH_NUM','FLT_DATE']).sum()['FLT_ERT_1']
month_day9=data9.groupby(['MONTH_NUM','FLT_DATE']).sum()['FLT_ERT_1']

MD_7= month_day7.to_numpy()
MD_8= month_day8.to_numpy()
MD_9= month_day9.to_numpy()
MD=np.concatenate((MD_7,MD_8,MD_9))

# Use K clustering to define the three bins
kmeans = KMeans(n_clusters=3)
month_reshap=np.reshape(MD, (-1, 1))
out = kmeans.fit_predict(month_reshap)
frame=pd.DataFrame([MD,out])
frame=frame.transpose()

# Put in variables the characteristics of the clusters/bins
cluster07=frame.loc[frame[1] == 0]
cluster17=frame.loc[frame[1] == 1]
cluster27=frame.loc[frame[1] == 2]
max07=max(cluster07[0]);min07=min(cluster07[0])
mean07=np.mean(cluster07[0]);std07=np.std(cluster07[0])
max17=max(cluster17[0]);min17=min(cluster17[0])
mean17=np.mean(cluster17[0]);std17=np.std(cluster17[0])
max27=max(cluster27[0]);min27=min(cluster27[0])
mean27=np.mean(cluster27[0]);std27=np.std(cluster27[0])

# Plot visual tool to observe the clustering and month distribution
plt.figure()
month7=data7.groupby(by=('MONTH_NUM')).sum()['FLT_ERT_1']
month8=data8.groupby(by=('MONTH_NUM')).sum()['FLT_ERT_1']
month9=data9.groupby(by=('MONTH_NUM')).sum()['FLT_ERT_1']
month20=data20.groupby(by=('MONTH_NUM')).sum()['FLT_ERT_1']
month7.plot()
month8.plot()
month9.plot()
month20.plot()
plt.axvline(x=3, color='pink');plt.axvline(x=6, color='pink');plt.axvline(x=9, color='pink');plt.axvline(x=12, color='pink')
# Plot visual tool to observe the clustering and day distribution
plt.figure(figsize=(16,7))
month_day7=data7.groupby(['MONTH_NUM','FLT_DATE']).sum()['FLT_ERT_1']
month_day8=data8.groupby(['MONTH_NUM','FLT_DATE']).sum()['FLT_ERT_1']
month_day9=data9.groupby(['MONTH_NUM','FLT_DATE']).sum()['FLT_ERT_1']
month_day20=data20.groupby(['MONTH_NUM','FLT_DATE']).sum()['FLT_ERT_1']
month_day7.plot()
month_day8.plot()
month_day9.plot()
month_day20.plot()
plt.axhspan(min07, max07, color='bisque', alpha=0.65, lw=0)
plt.axhspan(min17, max17, color='mistyrose', alpha=0.65, lw=0)
plt.axhspan(min27, max27, color='beige', alpha=0.65, lw=0)

# Compute date for easier identification of index
d31=[i+1 for i in range(31)]
d30=[i+1 for i in range(30)]
d28=[i+1 for i in range(28)]
d=[d31,d28,d31,d30,d31,d30,d31,d31,d30,d31,d30,d31]
d=np.concatenate(d)
m=[[1 for i in range(31)],[2 for i in range(28)],[3 for i in range(31)],[4 for i in range(30)],[5 for i in range(31)],[6 for i in range(30)],[7 for i in range(31)],[8 for i in range(31)],[9 for i in range(30)],[10 for i in range(31)],[11 for i in range(30)],[12 for i in range(31)]]
m=np.concatenate(m)
aux2=[]
aux=[0 for i in range(365)]
for i in range(len(m)):
    aux[i]=str(d[i]) +'/'+ str(m[i])
Date=np.concatenate([aux]*3)

# Save data for bins matching
frame['Date']=Date
TD=frame.loc[frame.index.repeat(2)].reset_index(drop=True)
os.chdir(r'C:\Users\lunap\PycharmProjects\pythonProject1\weather') # USE YOUR OWN DIRECTORY
var=open('TD.pickle', 'wb')
pickle.dump(TD, var)
