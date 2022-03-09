from datetime import datetime
import math
import pickle
import pandas as pd
from datetime import timedelta
import numpy as np

#
# Function get_bearing : calculate the bearing [º] based on longitude and latitude
#
def get_bearing(lat1, long1, lat2, long2):
    dLon = (long2 - long1)
    x = math.cos(math.radians(lat2)) * math.sin(math.radians(dLon))
    y = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(math.radians(dLon))
    brng = np.arctan2(x,y)
    brng = np.rad2deg(brng)
    round(brng,3)

    return brng

#
# Function clock : transform hours, minutes and seconds with one digit (e.g. 9) in two digits number (e.g. 09)
#
def clock(timec):
    if timec < 10:
        t = '0'+str(timec)
    else:
        t= str(timec)
    return t

# Initial commands of scenario
time = '00:00:00>'
cre = 'CRE'
move = 'ADDWPT'
delete = 'DEL'
scn = open('nominal.scn', 'w')
initial_command = time + 'PAN EHAM' + '\n' + time + 'ASAS ON' + '\n' + time + 'DT .5' + '\n' + time + 'FF 30000' + '\n'
datalog_command = time + 'CRELOG DLOG 1.0 Conflict log' + '\n' + time + 'DLOG ADD traf.cd.confpairs,traf.cd.dist, traf.cd.dcpa, traf.cd.tcpa, traf.cd.qdr, traf.cd.track, traf.cd.PHase, traf.cd.ALT'+ '\n' + time + 'DLOG ON'+ '\n'
scn.write(initial_command)
scn.write(datalog_command)


# Select manually the file w/ Eurocontrol data - ddr_doc has the general flight information, not the data points
file_path ='C:/Users/lunap/OneDrive/Ambiente de Trabalho/THESIS/DDR/Flights_20180901_20180930.csv' #CHANGE INTO OWN FILE
ddr_doc = pd.read_csv(file_path)

# Add extra columns for computations
ddr_doc.insert(loc=11, column='A_Off_timestamp', value=['' for i in range(ddr_doc.shape[0])])
ddr_doc.insert(loc=12, column='A_Arr_timestamp', value=['' for i in range(ddr_doc.shape[0])])
ddr_doc['A_Off_timestamp']=ddr_doc['ACTUAL OFF BLOCK TIME']
ddr_doc['A_Arr_timestamp']=ddr_doc['ACTUAL ARRIVAL TIME']

ddr=ddr_doc

# Apply the time filter by considering the flights that depart, the ones that arrive and the fly-by ones, respectively
mask1 = (ddr['FILED OFF BLOCK TIME'] > '02/09/2018 09:00:00') & (ddr['FILED OFF BLOCK TIME'] <= '02/09/2018 12:00:00')
mask2 = (ddr['FILED ARRIVAL TIME'] > '02/09/2018 09:00:00') & (ddr['FILED ARRIVAL TIME'] <= '02/09/2018 12:00:00')
mask3 = (ddr['FILED OFF BLOCK TIME'] < '02/09/2018 09:00:00') & (ddr['FILED ARRIVAL TIME'] > '02/09/2018 12:00:00')
mask=np.logical_or(mask1,mask2)
mask=np.logical_or(mask,mask3)
ddr=ddr.loc[mask]

i=0
while i<len(ddr):
    timestamp1 = ddr.iloc[i]['A_Off_timestamp']
    timestamp2 = ddr.iloc[i]['A_Arr_timestamp']
    ddr.iloc[i,11] = datetime.strptime(timestamp1, "%d/%m/%Y %H:%M")
    ddr.iloc[i,12] = datetime.strptime(timestamp2, "%d/%m/%Y %H:%M")
    i = i + 1

# Select manually the file w/ Eurocontrol data - ddr_doc_2 has the specific flight points information
file_path = 'C:/Users/lunap/OneDrive/Ambiente de Trabalho/THESIS/DDR/Flight_Points_Filed_20180901_20180930.csv' #CHANGE INTO OWN FILE
ddr_doc_2 = pd.read_csv(file_path)

# Load the WTC (Wake Turbulence Category) file to eliminate the aircraft that fall into Light category
ddr_doc_wtc = pd.read_excel(open('C:/Users/lunap/OneDrive/Ambiente de Trabalho/THESIS/DDR/WTC.xlsx', 'rb')) #CHANGE INTO OWN FILE
ddr_doc_wtc=ddr_doc_wtc[['ICAO Type Designator', 'ICAO WTC']]

# Include new parameter columns - later will be used for a more complete BlueSky simulator input
ddr_doc_2.insert(loc=4, column='Brng', value=['0' for i in range(ddr_doc_2.shape[0])])
ddr_doc_2.insert(loc=7, column='ACmodel', value=['0' for i in range(ddr_doc_2.shape[0])])
ddr_doc_2.insert(loc=8, column='Dest', value=['0' for i in range(ddr_doc_2.shape[0])])

# Apply the date and time filter
mask = (ddr_doc_2['Time Over'] > '02-09-2018 09:00:00') & (ddr_doc_2['Time Over'] <= '02-09-2018 12:00:00')
ddr_doc_2=ddr_doc_2.loc[mask]

# Apply the geographical filter - squared box
mask_Lat=ddr_doc_2['Latitude'].between(50.797719, 53.886976, inclusive=False)
mask_Long=ddr_doc_2['Longitude'].between(1.408212, 7.417641, inclusive=False)
ddr_doc_2=ddr_doc_2.loc[mask_Lat]
ddr_doc_2=ddr_doc_2.loc[mask_Long]

data=[]
i=0
while i < len(ddr): # Loop to go through every flight
    aux = ddr_doc_2.loc[ddr_doc_2['ECTRL ID'] == int(ddr.iat[i,0])] # Auxiliar variable with the flight data points for the ith flight
    info = ddr_doc.loc[ddr_doc['ECTRL ID'] == int(ddr.iat[i,0])] # Auxiliar variable with information about the flight for the ith file
    if not aux.empty:
        aux['Time Over'] = pd.to_datetime(aux['Time Over'])
        if len(aux)>=2: # if there is more than one data point, calculate initial bearing
            brng = get_bearing(aux.iat[0, 5], aux.iat[0, 6], aux.iat[1, 5], aux.iat[1, 6])  # calc bearing
            if aux.iat[1,3]==0: # assess if the second datapoint has 0m altitude - then, eliminate the first one
                aux=aux.drop(aux.index[[0]])
                if len(aux)==1: # assess if by eliminating the previous information, the total data only has the starting point
                    brng=0 # if condition is true, assume brng 0 since there is one way to calculate it
                else:brng = get_bearing(aux.iat[0, 5], aux.iat[0, 6], aux.iat[1, 5], aux.iat[1, 6])  # if condition is false, calculate brng
        else: brng=0

        aux.iat[0, 4] = brng
		# The model SU95 is not available in BlueSky, so it's replaced by the most similar
        if info.iat[0,13] == 'SU95': 
            info.iat[0,13] = 'B732'

        aux.iat[0, 7] = info.iat[0,13] # Assign to flight points database the AC Type
        aux.iat[0, 8] = info.iat[0,4] # Assign to flight points database the Destination

        aux = aux.dropna()  # NaN values can not be used

        # Include delete aircraft command
        del_row = {'Time Over': aux.iat[len(aux)-1, 2]+timedelta(seconds=1) , 'ECTRL ID': aux.iat[0, 0], 'Sequence Number': 'DEL',
                   'Flight Level': 0,'Latitude':0,'Longitude':0, 'ACmodel':0, 'Dest':0}
        # Append row to the dataframe
        aux = aux.append(del_row, ignore_index=True)
        if (ddr_doc_wtc.loc[ddr_doc_wtc['ICAO Type Designator'] == aux.iat[0, 7]].empty) or (ddr_doc_wtc.loc[ddr_doc_wtc['ICAO Type Designator'] == aux.iat[0, 7]].iat[0,1]=='L') or len(aux)<=2: # Assess wheter it's general aviation (Light under WTC)
            print("Eliminated")
        else:
            data.append(aux)
    i = i + 1

fileddata = pd.concat(data)

#%%
# Write filed flights information in the scenario file
# Initialize variables to identify which flight information is already (or on going) written in the scenario
AC_ectrl_F = [0 for k in range(len(ddr))]
AC_n_F = 0
fileddata = fileddata.sort_values(['Time Over'], ascending=True)
j=0 # Loop to go through each data line
while j<len(fileddata):
    aux_F= str(int(fileddata.iloc[j]['ECTRL ID']))+'F' in AC_ectrl_F # Flag to identify if flight information was already processed

    if aux_F: # if the flight information process is on going - flag aux_R is True

         if fileddata.iat[j,1]=='DEL': # Verify if is the last data information available

             # Include DEST and DEL commands consecutively
             command_F = time + 'DEST ' + str(int(fileddata.iloc[j]['ECTRL ID']))+'F'+' '+ dest + '\n'
             time_d =fileddata.iloc[j]['Time Over']+timedelta(seconds=1)
             time_d=clock(time_d.time().hour) + ':' + clock(time_d.time().minute) + ':' + clock(time_d.time().second)+ '>'
             command_F =command_F+ time_d + delete + ' ' + str(int(fileddata.iloc[j]['ECTRL ID']))+'F'+'\n'

         else: # This else encompasses all the ADDWPT, RTA commands between the initial commands and the two last ones (Destination and Delete)
             p = p + 1 # auxiliar variable for waypoint id

             #ADDWPT Command - Add waypoint according to the Eurocontrol data
             aircraft = str(int(fileddata.iloc[j]['ECTRL ID']))+'F'+' '+str(fileddata.iloc[j]['Latitude'])+' '+str(fileddata.iloc[j]['Longitude'])+' '+',FL'+str(clock(fileddata.iloc[j]['Flight Level']))
             command_F = time + move + ' ' + aircraft +'\n'

             #RTA Command - telling the system at what time to be in the added waypoint
             time_rta = clock(fileddata.iloc[j]['Time Over'].time().hour) + ':' + clock(fileddata.iloc[j]['Time Over'].time().minute) + ':' + clock(
                 fileddata.iloc[j]['Time Over'].time().second)
             aircraft = str(int(fileddata.iloc[j]['ECTRL ID']))+'F'+' '+str(int(fileddata.iloc[j]['ECTRL ID']))+'F'+"%03d" % p+' '+time_rta
             command_F = command_F + time + 'RTA' +' '+ aircraft+'\n'

             # Time update for the next waypoint
             t = t + timedelta(seconds=1)
             time = clock(t.time().hour) + ':' + clock(t.time().minute) + ':' + clock(t.time().second)+ '>'

    else: # if the flight information process is new - flag aux_R is False
         p = 0 # auxiliar variable for waypoint id

         # Include flight in the processed flights list
         AC_ectrl_F[AC_n_F]=str(int(fileddata.iloc[j]['ECTRL ID']))+'F'
         AC_n_F=AC_n_F+1

         dest=str(fileddata.iloc[j]['Dest']) # Destination variable - information only in first row
         # CRE Command - Create the aircraft in the simulation
         time = clock(fileddata.iloc[j]['Time Over'].time().hour) + ':' + clock(fileddata.iloc[j]['Time Over'].time().minute) + ':' + clock(
             fileddata.iloc[j]['Time Over'].time().second)+ '>'
         if str(fileddata.iloc[j]['Flight Level']*100)== '0': # if FL is 0, assign a very small value like 0.01ft
             aircraft = str(int(fileddata.iloc[j]['ECTRL ID']))+'F'+' '+str(fileddata.iloc[j]['ACmodel'])+' ' +str(fileddata.iloc[j]['Latitude'])+' '+str(fileddata.iloc[j]['Longitude'])+' '+str(fileddata.iloc[j]['Brng'])+' '+str(fileddata.iloc[j]['Flight Level']*100)+ ' 0.01'
         else: aircraft = str(int(fileddata.iloc[j]['ECTRL ID']))+'F'+' '+str(fileddata.iloc[j]['ACmodel'])+' ' +str(fileddata.iloc[j]['Latitude'])+' '+str(fileddata.iloc[j]['Longitude'])+' '+str(fileddata.iloc[j]['Brng'])+' '+str(fileddata.iloc[j]['Flight Level']*100)

         command_F = time + cre + ' ' + aircraft + '\n'

         # Time update for next commands - BlueSky processing responds better if the information is in different timesteps
         t=fileddata.iloc[j]['Time Over']+timedelta(seconds=1)
         time = clock(t.time().hour)+':'+clock(t.time().minute)+':'+clock(t.time().second)+ '>'

         # LNAV command
         command_F = command_F + time + 'LNAV' + ' ' + str(int(fileddata.iloc[j]['ECTRL ID'])) + 'F' + ' ' + 'ON' + '\n'
         # VNAV command
         command_F = command_F + time + 'VNAV' + ' ' + str(int(fileddata.iloc[j]['ECTRL ID'])) + 'F' + ' ' + 'ON' + '\n'

    scn.write(command_F) # write the command (whether is the CRE, ADDWPT, LNAV, VNAV, RTA, DEST or DEL) in the scenario
    j = j + 1

quit='15:00:00>QUIT\n'
scn.write(quit)
scn.close()

# Sort the commands by time
scn= open('nominal.scn', 'r+')
list = [line.split('>') for line in scn.readlines()]
list.sort()
scn.close()
# Copy&Paste in the final scenario with the command complete
scn2 = open('filed_6h.scn', 'w')
for element in list:
    element=">".join(element)
    scn2.write(element)
scn2.close()


