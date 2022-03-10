import os
import pickle
import pandas as pd
from datetime import timedelta
from scipy.stats import truncnorm
from datetime import datetime
from bluesky.tools.aero import tas2cas
from math import sin, cos, atan2, sqrt
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy import signal
import math
import numpy as np
from sklearn.linear_model import LinearRegression
import time

# Define directory where the project is
os.chdir(r"C:\Users\lunap\PycharmProjects\pythonProject1\lookahead_time") #CHANGE FOR YOR OWN
#
# Function inBetween: Boolean function, assesses whether a value is within an interval
#
def inBetween(minv, val, maxv):
  if minv <= val <= maxv: return True
  if minv > val:        return False
  if maxv < val:        return False

#
# Function phase_attribution: identify and assign flight phase
#
def phase_attribution(aux):
    # Calculate first linear regression
    X = aux['Seq'][0:len(aux) - 1].values.reshape(-1, 1)
    Y = aux['geoaltitude'][0:len(aux) - 1].values.reshape(-1,1)
    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions

    ERRO = Y - Y_pred # error function
    aux['PH'] = ['' for j in range(len(aux))]

    # Calculate how many phases
    if (((ERRO[:-1] * ERRO[1:]) < 0).sum() == 2 and max(ERRO) > 500 and ERRO[0] < 1000 and ERRO[
        len(ERRO) - 1] < 1000) or (
            abs(max(ERRO)) + abs(min(ERRO)) > 2500 and ERRO[0] < 1000 and ERRO[len(ERRO) - 1] < 1000):
        ERRO = ERRO.flatten().tolist()

        # Calculate phase changing point - correct if it is in borders
        xMax = ERRO.index(max(ERRO))
        if max(ERRO) == ERRO[0] or max(ERRO) == ERRO[len(ERRO) - 1] or (
                ERRO[xMax - 1] < ERRO[xMax] and ERRO[xMax] < ERRO[xMax + 1]):
            xMax = ERRO.index(min(ERRO))
            if min(ERRO) == ERRO[0] or min(ERRO) == ERRO[len(ERRO) - 1]: xMax = ERRO.index(ERRO[int(len(ERRO) / 2)])
        # Calculate new linear regressions and linear equations
        X1 = aux['Seq'][0:xMax].values.reshape(-1, 1)
        Y1 = aux['geoaltitude'][0:xMax].values.reshape(-1,1)
        linear1_regressor = LinearRegression()
        linear1_regressor.fit(X1, Y1)
        Y1_pred = linear1_regressor.predict(X1)  # make predictions

        X2 = aux['Seq'][xMax + 1:len(aux) - 1].values.reshape(-1, 1)
        Y2 = aux['geoaltitude'][xMax + 1:len(aux) - 1].values.reshape(-1,1)
        linear2_regressor = LinearRegression()
        linear2_regressor.fit(X2, Y2)
        Y2_pred = linear2_regressor.predict(X2)  # make predictions

        x = [float(X1[0]), float(X1[len(X1) - 1])]
        y = [float(Y1[0]), float(Y1[len(Y1) - 1])]
        # Calculate the coefficients. This line answers the initial question.
        coefficients1 = np.polyfit(x, y, 1)

        x = [float(X2[0]), float(X2[len(X2) - 1])]
        y = [float(Y2[0]), float(Y2[len(Y2) - 1])]
        # Calculate the coefficients. This line answers the initial question.
        coefficients2 = np.polyfit(x, y, 1)

        #Calculate delta for the coefficients
        delta1 = abs(linear1_regressor.coef_ - coefficients1[0])
        delta2 = abs(linear2_regressor.coef_ - coefficients2[0])
        min_delta1 = max(.3 * abs(linear1_regressor.coef_), .3 * abs(coefficients1[0]), 2)
        min_delta2 = max(.3 * abs(linear2_regressor.coef_), .3 * abs(coefficients2[0]), 2)

        # Conditions for phases and phases attribution
        if linear1_regressor.coef_ < -1.5:
            aux['PH'] = ['Descent' for j in range(len(aux))]
        elif (delta1 > delta2 and delta1 > min_delta1 and coefficients1[0] > 0.5 and coefficients2[0] < -1 and
              coefficients2[0] * coefficients1[0] < 0) or (
                delta1 > delta2 and coefficients2[0] * coefficients1[0] < -225):

            ERRO1 = Y1 - Y1_pred
            ERRO1 = ERRO1.flatten().tolist()
            xMax_1 = ERRO1.index(max(ERRO1))

            aux['PH'][0:xMax_1] = ['Climb' for j in range(xMax_1)]
            aux['PH'][xMax_1:xMax] = ['Cruise' for j in range(xMax - xMax_1)]
            aux['PH'][xMax:len(aux)] = ['Descent' for j in range(len(aux) - xMax)]

        elif delta2 > delta1 and delta2 > min_delta2 and coefficients2[0] < -.5 and coefficients1[0] > 2 and \
                coefficients2[0] * coefficients1[0] < 0 or (
                delta2 > delta1 and coefficients2[0] * coefficients1[0] < -225):

            ERRO2 = Y2 - Y2_pred
            ERRO2 = ERRO2.flatten().tolist()
            xMax_2 = ERRO2.index(max(ERRO2))

            aux['PH'][0:xMax] = ['Climb' for j in range(xMax)]
            aux['PH'][xMax: xMax + xMax_2] = ['Cruise' for j in range(xMax_2)]
            aux['PH'][xMax + xMax_2: len(aux)] = ['Descent' for j in range(len(aux) - xMax_2 - xMax)]

        else:

            if inBetween(-1.5, coefficients1[0], 1.5) or delta1 > min_delta1:
                aux['PH'][0: xMax] = ['Cruise' for j in range(xMax)]
                if linear2_regressor.coef_ < 0:
                    aux['PH'][xMax: len(aux)] = ['Descent' for j in range(len(aux) - xMax)]

            elif inBetween(-1.5, coefficients2[0], 1.5) or delta2 > min_delta2:
                aux['PH'][xMax: len(aux)] = ['Cruise' for j in range(len(aux) - xMax)]
                if linear1_regressor.coef_ > 0:
                    aux['PH'][0: xMax] = ['Climb' for j in range(xMax)]

            elif linear_regressor.coef_ < -.5:
                aux['PH'] = ['Descent' for j in range(len(aux))]

            elif linear_regressor.coef_ > .5:
                aux['PH'] = ['Climb' for j in range(len(aux))]

            elif inBetween(-.5, linear_regressor.coef_, .5):
                aux['PH'] = ['Cruise' for j in range(len(aux))]

    elif linear_regressor.coef_ < -.5:
        aux['PH'] = ['Descent' for j in range(len(aux))]
    elif linear_regressor.coef_ > .5:
        aux['PH'] = ['Climb' for j in range(len(aux))]
    elif inBetween(-.5, linear_regressor.coef_, .5):
        aux['PH'] = ['Cruise' for j in range(len(aux))]

    return aux

#
# Function new_coord: calculate next coordinates from current coordinates with velocity, bearing and altitude
#
def new_coord(lat1,lon1, vel, brng,alt):
    R = 6378.1 + alt/1000  # Radius of the Earth
    brng = math.radians(brng) # Bearing is 90 degrees converted to radians.
    d = vel/1000 # Distance in km

    lat1 = math.radians(lat1)  # Current lat point converted to radians
    lon1 = math.radians(lon1)  # Current long point converted to radians

    lat2 = math.asin(math.sin(lat1) * math.cos(d / R) +
                     math.cos(lat1) * math.sin(d / R) * math.cos(brng))

    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d / R) * math.cos(lat1),
                             math.cos(d / R) - math.sin(lat1) * math.sin(lat2))

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return [lat2,lon2]

#
# Function filter_vv: eliminate outliers from the vertical velocity (absolute parameters in m/s)
#
def filter_vv(aux, avg):
    # Assess if first position is an outlier and correct it
    if abs(aux.iat[0, 6]) > 17.5 and aux.iat[0, 6] < 0:
        # from samples, the negative vrate outliers are for values around -5m/s
        aux.iat[0, 6] = -5.1  # -1000fpm
    if abs(aux.iat[0, 6]) > 17.5 and aux.iat[0, 6] > 0:
        aux.iat[0, 6] = 0
    # Go through the trajectory to identify the outliers (absolute and relative criteria)
    for j in range(len(aux) - 3):

        if aux.iat[j + 1, 6] > 3 and avg < -20:
            aux.iat[j + 1, 6] = 0

        if (abs(abs(aux.iat[j + 1, 6]) - abs(aux.iat[j, 6])) > 10):
            aux.iat[j + 1, 6] = aux.iat[j, 6]
        if ((aux.iat[j + 1, 6] > aux.iat[j, 6])):
            if ((aux.iat[j + 2, 6] <= aux.iat[j, 6]) or (aux.iat[j + 3, 6] <= aux.iat[j, 6])):
                aux.iat[j + 1, 6] = aux.iat[j, 6]
        if ((aux.iat[j + 1, 6] < aux.iat[j, 6])):
            if ((aux.iat[j + 2, 6] >= aux.iat[j, 6]) or (aux.iat[j + 3, 6] >= aux.iat[j, 6])):
                aux.iat[j + 1, 6] = aux.iat[j, 6]
    return aux
#
# Function get_bearing : obtain the heading from two different position data points
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
# Function deg2rad : transform input in degrees to radians
#
def deg2rad(x):
    return math.pi * x /180
#
# Function getD : Get distance between two 3D coordinates
#
def getD(lat1,lon1,lat2,lon2, alt):
  R = 6371 + alt/1000 # Radius of the earth in km + h
  dLat = deg2rad(lat2-lat1)
  dLon = deg2rad(lon2-lon1)
  a =sin(dLat/2) * sin(dLat/2) + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dLon/2) * sin(dLon/2)
  c = 2 * atan2(sqrt(a), sqrt(1-a))
  d = R * c # Distance in km
  return d*1000

#
# Function data_A : process actual data coming from OpenSky directly - mostly eliminate duplicates, interpolation and filtering
#
def data_A(actualdata, AC,):
    # Load the WTC (Wake Turbulence Category) database
    ddr_doc_wtc = pd.read_excel(open('C:/Users/lunap/OneDrive/Ambiente de Trabalho/THESIS/DDR/WTC.xlsx', 'rb'))
    ddr_doc_wtc = ddr_doc_wtc[['ICAO Type Designator', 'ICAO WTC']]
    # Load the aircraft database - correlation between ICAO code, AC registration and typecode
    file_path = 'C:/Users/lunap/OneDrive/Ambiente de Trabalho/THESIS/DDR/aircraftDatabase.csv'
    aircraft_database = pd.read_csv(file_path, usecols=['icao24', 'registration', 'typecode'])
    aircraft_database.values.tolist()

    id=[]
    data=[]
    i=0
    while i < len(AC): # for every flight go through the processing and in the end add it the final data or discard
        print(AC[i])
        aux = actualdata.loc[actualdata['callsign'] == AC[i]] # select all tje data for the selected flight into aux
        aux.sort_values(by=['time'])  # to correct some incoherent data entries
        aux = aux[~pd.DataFrame(np.sort(aux[['lat', 'lon']], axis=1), index=aux.index).duplicated()]
        aux = aux[~pd.DataFrame(aux['time']).duplicated()]
        timestamp = aux.iloc[:, 0]
        for k in range(len(timestamp)):
            timestamp_aux = datetime.utcfromtimestamp(timestamp.iat[k]).strftime("%d-%m-%Y %H:%M:%S")
            aux.iat[k, 0] = datetime.strptime(timestamp_aux, "%d-%m-%Y %H:%M:%S")

        aux.insert(loc=8, column='AC reg', value=['' for i in range(aux.shape[0])])
        aux.insert(loc=9, column='ACmodel', value=['' for i in range(aux.shape[0])])
        aux['Seq']=[j for j in range(len(aux))]

        icao=aux.iat[0,1]
        aux2 = aircraft_database.loc[aircraft_database['icao24'] == icao] # aux2 has the ICAO code, AC registration and typecode information
        aux2 = aux2.values.tolist()
        if np.size(aux2) == 3 and not (pd.isna(aux2[0][1])):
            aux3 = aux2[0][1]
            aux4= aux2[0][2]
            aux3 = aux3.replace('-', '')
            aux.iloc[0, 8] = aux3
            if pd.isna(aux4):
                aux.iloc[0, 9] = 'UL45' # give this typecode so that in the following processing step, it will be eliminated
                id=[id,i]
            else: aux.iloc[0, 9] = aux4
        else:
            aux.iloc[0, 8] = 0
            aux.iloc[0, 9] = 'A320'


        if aux.iat[0, 9] == 'SU95': # aircraft model SU95 is not included in BlueSky simulator - a similar model was selected
            aux.iat[0, 9] = 'B732'

        if len(aux)!=0:
            #timestamp = aux.iloc[:,0]
            callsign = aux.iloc[:,7]
            diff=False # Flag to assess if the data will be shortened or not
            var=True # aux flag to break a loop

            # Loop to go through every data row - verify time step, altitude values, onground flag and include last data entry marker
            for k in range(len(callsign)):
                #timestamp_aux = datetime.utcfromtimestamp(timestamp.iat[k]).strftime("%d-%m-%Y %H:%M:%S")
                #aux.iat[k,0] = datetime.strptime(timestamp_aux, "%d-%m-%Y %H:%M:%S")
                callsign.iat[k]=callsign.iat[k].replace(" ", "")
                if k>=1:
                    if (((aux.iat[k,0]-aux.iat[k-1,0] > timedelta(seconds=60)) | (aux.iat[k,10] == 1)) & var):
                        diff=True
                        var=False
                        mark=k
                    else:
                        if (abs(aux.iat[k-1,10] - aux.iat[k, 10]) > 300):
                            aux.iat[k,10] = aux.iat[k-1, 10]

            aux['time'] = pd.to_datetime(aux['time'])
            aux=aux.dropna()
            #
            if not diff:
                print('h')
            else:
                aux=aux[0:mark]

            if (ddr_doc_wtc.loc[ddr_doc_wtc['ICAO Type Designator'] == aux.iat[0, 9]].empty) or (
                            ddr_doc_wtc.loc[ddr_doc_wtc['ICAO Type Designator'] == aux.iat[0, 9]].iat[0, 1] == 'L') or (len(aux)<30) or aux.iat[0,7]=='7':
                        print("AAAAAAAAAAAAAQQQQQQQQQQQQQQQQUUUUUUUUUUUUUUIIIIIIIIIIII")
                        if(ddr_doc_wtc.loc[ddr_doc_wtc['ICAO Type Designator'] == aux.iat[0, 9]].empty):
                            #print('eheh')
                            print(aux.iat[0, 9])

            else:
                # Filter manually vertical rate
                avg = aux['vertrate'].mean()
                if avg > -200:
                    aux = filter_vv(aux, avg)
                    # Filter manually horizontal velocity
                    for j in range(len(aux) - 3):
                        if ((aux.iat[j + 1, 4] > aux.iat[j, 4])):
                            if ((aux.iat[j + 2, 4] <= aux.iat[j, 4])):
                                aux.iat[j + 1, 4] = aux.iat[j, 4]

                        if ((aux.iat[j + 1, 4] < aux.iat[j, 4])):
                            if ((aux.iat[j + 2, 4] >= aux.iat[j, 4])):
                                aux.iat[j + 1, 4] = aux.iat[j, 4]

                    aux.reset_index(drop=True, inplace=True)
                    df = pd.DataFrame({'icao24': aux['icao24'],
                                       'lat': aux['lat'],
                                       'lon': aux['lon'],
                                       'velocity': aux['velocity'],
                                       'heading': aux['heading'],
                                       'vertrate': aux['vertrate'], 'callsign': aux['callsign'],
                                       'AC reg': aux['AC reg'],'ACmodel': aux['ACmodel'],
                                       'geoaltitude': aux['geoaltitude']},
                                      index=pd.to_datetime(np.array(aux['time'])))
                    idx = pd.to_datetime(pd.date_range(df.index[0], df.index[-1], freq='s').strftime('%Y-%m-%d %H:%M:%S'))
                    df['heading'] = np.rad2deg(np.unwrap(np.deg2rad(aux['heading'])))
                    df['lat'] = np.rad2deg(np.unwrap(np.deg2rad(aux['lat'])))
                    df['lon'] = np.rad2deg(np.unwrap(np.deg2rad(aux['lon'])))
                    df['vertrate'] = np.array(aux['vertrate'])
                    df['velocity'] = np.array(aux['velocity'])
                    df['geoaltitude'] = np.array(aux['geoaltitude'])
                    df = df.reindex(idx, fill_value=np.nan)
                    df = df.interpolate(method='time')
                    alt=df['geoaltitude']
                    df.reset_index(drop=False, inplace=True)
                    alt=pd.DataFrame(alt)
                    alt.reset_index(drop=True, inplace=True)
                    df[['lat', 'lon', 'heading']] %= 360
                    df['icao24'] = [aux.iat[0, 1] for k in range(len(df))]
                    df['callsign'] = [aux.iat[0, 7] for k in range(len(df))]
                    df['AC reg'] = [aux.iat[0, 8] for k in range(len(df))]
                    df['ACmodel'] = [aux.iat[0, 9] for k in range(len(df))]

                    df.columns = df.columns.str.replace('index', 'time')
                    aux = df
                    aux['geoaltitude']=alt['geoaltitude']
                    aux['Seq'] = [j for j in range(len(aux))]


                    miu = ['0' for i in range(aux.shape[0])]
                    av = 20
                    for k in range(len(aux)):
                        if ((len(aux) > av)):  # B - before ; A - after
                            if k < int(av / 2):
                                B = k + 1
                                A = av - B
                            elif len(aux) - k - 1 <= int(av / 2):
                                A = len(aux) - k - 2
                                B = 20 - A
                            else:
                                A = int(av / 2)
                                B = A
                            miu[k] = np.average(np.concatenate((np.array([aux.iat[k - h, 6] for h in range(B)]),
                                                                np.array([aux.iat[k + h + 1, 6] for h in range(A)]))))

                        else:
                            av = len(aux)
                            vel = sum([aux.iat[k, 6] for k in range(len(aux))]) / av
                            miu = [vel for j in range(len(aux))]

                    aux['vertrate']=pd.to_numeric(miu)
                    b_filter, a_filter = butter(2, 0.1 * 2 / 1, btype='low', analog=False)
                    zi = signal.lfilter_zi(b_filter, a_filter)
                    aux['vertrate'] = lfilter(b_filter, a_filter, np.array(aux['vertrate']), zi=zi * np.array(aux['vertrate'][0]).item(0))[0]

                    aux.iat[0, 6] = aux.iat[0, 6] * 60 / 0.3048  # vspeed m/s to fpm
                    aux.iat[0, 10] = aux.iat[0, 10] / 0.3048  # altitude m to ft
                    for l in range(len(aux) - 1):
                        k = l + 1
                        aux.iat[k, 6] = aux.iat[k, 6] * 60 / 0.3048  # vspeed m/s to fpm
                        aux.iat[k, 10] = aux.iat[k - 1, 10] + (aux.iat[k, 6] / 60) # in ft
                    aux.iat[len(aux) - 1, 6] = aux.iat[len(aux) - 2, 6]  # last cell
                    aux.iat[len(aux) - 1, 10] = aux.iat[len(aux) - 2, 10] + (aux.iat[len(aux) - 2, 6] / 60)

                    sin = ['' for j in range(len(aux))]
                    cos = ['' for j in range(len(aux))]
                    for l in range(len(aux)):
                        sin[l] = math.sin(math.radians(aux['heading'][l]))
                        cos[l] = math.cos(math.radians(aux['heading'][l]))
                    b_filter, a_filter = butter(1, 0.04 * 2 / 1, btype='low', analog=False)
                    zi = signal.lfilter_zi(b_filter, a_filter)
                    sin_f = lfilter(b_filter, a_filter, sin, zi=zi * sin[0])
                    b_filter, a_filter = butter(1, 0.04 * 2 / 1, btype='low', analog=False)
                    zi = signal.lfilter_zi(b_filter, a_filter)
                    cos_f = lfilter(b_filter, a_filter, cos, zi=zi * cos[0])

                    angle = ['' for j in range(len(aux))]
                    cos_f = np.clip(cos_f[0], -1, 1)
                    sin_f = np.clip(sin_f[0], -1, 1)
                    for l in range(len(aux)):
                        a_acos = math.acos(cos_f[l])
                        if sin_f[l] < 0:
                            angle[l] = math.degrees(-a_acos) % 360
                        else:
                            angle[l] = math.degrees(a_acos)
                    aux['heading']=angle

                    loo = ['' for k in range(len(aux))];    laa = ['' for k in range(len(aux))]
                    loo[0] = aux.iat[0, 3];     laa[0] = aux.iat[0, 2]
                    for j in range(len(aux) - 1):
                        k = j + 1
                        [laa[k], loo[k]] = new_coord(laa[j], loo[j], aux.iat[j,4], angle[j],aux.iat[j, 10]*.3048)
                    aux['LAT']=laa
                    aux['LON']=loo


                    for k in range(len(aux)):
                        aux.iat[k, 4] = tas2cas(aux.iat[k, 4], aux.iat[k, 10]*0.3048) * 1.9438 # tas to cas and to knots


                    #  include mark to delete the aircraft
                    del_row = {'time': aux.iat[len(aux) - 1, 0] + timedelta(seconds=1),
                               'icao24': 'DEL', 'AC reg': 0, 'lat': 0, 'lon': 0, 'velocity': 0, 'heading': 0, 'vertrate': 0,
                               'callsign': aux.iat[0, 7], 'AC reg': 0, 'ACmodel': 0, 'geoaltitude': 0}
                    # append row to the dataframe
                    aux = aux.append(del_row, ignore_index=True)

                    aux['velocity']=pd.to_numeric(aux['velocity'])
                    aux['heading'] = pd.to_numeric(aux['heading'])
                    aux['vertrate'] = pd.to_numeric(aux['vertrate'])
                    aux['LAT'] = pd.to_numeric(aux['LAT'])
                    aux['LON'] = pd.to_numeric(aux['LON'])

                    aux = phase_attribution(aux)


                    data.append(aux)



        i = i + 1
    ddr_doc = pd.concat(data)


    return ddr_doc

#
# Function time_shift : time shift data using the random or the compression methods
#
def time_shift(actualdata, AC, time_compression,time_random):
    i = 0
    data_actual=[]
    X = np.random.normal(0, 900, len(AC)).astype(int).tolist() # Vector for random delta time shifts
    while i < len(AC):
        aux= actualdata.loc[actualdata['callsign'] == (AC[i])] # Auxiliar variable to store all the data that will be time shifted p/ flight
        if time_compression:# Time shifting method option
            # Define the same dt to apply to every data point of each trajectory
            BL = 1535864400.0 # baseline (in seconds) match to 2018-02-09 07:00 / CHANGE THIS TO SCENARIO'S BASELINE TIME
            t0_1=aux.iat[0,0]
            t0_1=t0_1.to_pydatetime()
            t0_2=time.mktime(t0_1.timetuple()) # in seconds
            deltat=t0_2-BL
            dt=round(deltat*.2)
            DT = timedelta(seconds=dt)
        elif time_random:
            # Define the same dt to apply to every data point of each trajectory
            dt = X[i]
            DT = timedelta(seconds=dt)

        aux['time'] = aux['time']-DT  # time shift
        data_actual.append(aux)

        i=i+1

    actualdata = pd.concat(data_actual)

    return [actualdata]

#
# Function clock : transform hours, minutes and seconds with one digit (e.g. 9) in two digits number (e.g. 09)
#
def clock(timec):
    if timec < 10:
        t = '0'+str(timec)
    else:
        t= str(timec)
    return t

#ID for all the outputs
nr=25 # CHANGE FOR YOUR OWN ID

# Time shifting flags
time_compression=0
time_random=1

# Initial settings commands of scenario
time = '00:00:00>'
cre = 'CRE'
move = 'MOVE'
delete = 'DEL'
scn = open('aux'+str(nr)+'.scn', 'w')
initial_command = time + 'PAN EHAM' + '\n' + time + 'ASAS ON' + '\n' + time + 'DT .5' + '\n' + time + 'FF 80000' + '\n' + time + 'DTLOOK 0' + '\n'
scn.write(initial_command)
datalog_command = time + 'CRELOG DLOG 1.0 Conflict log' + '\n' + time + 'DLOG ADD traf.cd.confpairs,traf.cd.dist, traf.cd.dcpa, traf.cd.tLOS, traf.cd.qdr, traf.cd.track, traf.cd.PHase, traf.cd.ALT'+ '\n' + time + 'DLOG ON'+ '\n'
scn.write(datalog_command)
name ='SCN_Data'+str(nr)
exit()
#%%
# Load the data otained from OpenSky
var = open('Data_final\Data'+str(nr)+'.pickle', 'rb')
dfnew = pickle.load(var)
var.close()

actualdata=dfnew # main dataframe is actualdata
del actualdata["baroaltitude"]
del actualdata["hour"]
del actualdata["lastcontact"]
del actualdata["lastposupdate"]
del actualdata["onground"]
del actualdata["spi"]
del actualdata["alert"]
#actualdata = actualdata.loc[actualdata['time'] > 1535878800]
#actualdata = actualdata.loc[actualdata['time'] < 1535889600]
AC_ectrl=actualdata['callsign'].unique() # identifying the different aircraft

# Process actual data
actualdata=data_A(actualdata,AC_ectrl)
AC_ectrl=actualdata['callsign'].unique() # identifying the different aircraft

var=open('Data_final/actual_Data_'+str(nr)+'.pickle', 'wb')
pickle.dump(actualdata, var)
var.close()

#%% # Time Shift
var=open('Data_final/actual_Data_'+str(nr)+'.pickle', 'rb')
actualdata = pickle.load(var)
AC_ectrl=actualdata['callsign'].unique() # identifying the different aircraft

if time_compression or time_random:
    data=time_shift(actualdata,AC_ectrl,time_compression,time_random)
    actualdata=data[0]
actualdata = actualdata.sort_values(['time'], ascending=True)


# Write actual flight points information in the scenario file
time = '00:00:00>'

# Initialize variables to identify which flight information is already (or on going) written in the scenario
AC_ectrl_R = [0 for k in range(len(actualdata['callsign'].unique()))]
AC_n_R = 0

i = 0 # Loop to go through each data line
while i < len(actualdata):
    aux_R = str(actualdata.iloc[i][7]) in AC_ectrl_R # Flag to identify if flight information was already processed
    time = clock(actualdata.iloc[i][0].time().hour) + ':' + clock(actualdata.iloc[i][0].time().minute) + ':' + clock(actualdata.iloc[i][0].time().second) + '>' # Time in BlueSky scenario format
    if aux_R: # if the flight information process is on going - flag aux_R is True
        if actualdata.iat[i, 1] == 'DEL': # Verify if is the last data information available
            command_R = time + delete + ' ' + str(actualdata.iloc[i]['callsign']) + '\n'
        else: # This else encompasses all the MOVE and ALT commands between the initial commands and the last Delete
            aircraft = str(actualdata.iloc[i]['callsign']) + ' ' + str(actualdata.iloc[i]['LAT']) + ' ' + str(actualdata.iloc[i]['LON']) + ' ' + str(round(actualdata.iloc[i]['geoaltitude'],2)) + ' ' + str(round(
            actualdata.iloc[i]['heading'],3)) + ' ' + str(round(actualdata.iloc[i]['velocity'], 2)) + ' ' + str(round(actualdata.iloc[i]['vertrate'], 3))
            alt = str(actualdata.iloc[i]['callsign']) + ' ' + str(round(actualdata.iloc[i + 1]['geoaltitude'], 2)) + ' ' + str(round(actualdata.iloc[i]['vertrate'], 2))
            command_R = time + move + ' ' + aircraft + '\n' + time + 'ALT' + ' ' + alt + '\n'
    else: # if the flight information process is new - flag aux_R is False
        # Include flight in the processed flights list
        AC_ectrl_R[AC_n_R] = str(actualdata.iloc[i]['callsign'])
        AC_n_R = AC_n_R + 1

        aircraft = str(actualdata.iloc[i]['callsign']) + ' '+str(actualdata.iloc[i]['ACmodel'])+' ' + str(actualdata.iloc[i]['LAT']) + ' ' + str(
        actualdata.iloc[i]['LON']) + ' ' + str(round(actualdata.iloc[i]['heading'],3)) + ' ' + str(round(
        actualdata.iloc[i]['geoaltitude'],2)) + ' ' + str(round(actualdata.iloc[i]['velocity'], 2))
        command_R = time + cre + ' ' + aircraft + '\n'

    scn.write(command_R) # write the command (whether is the CRE, MOVE, ALT or DEL)
    i = i + 1

quit='15:00:00>QUIT\n'
scn.write(quit)
scn.close()

# Sort the commands by time
scn= open('aux'+str(nr)+'.scn', 'r+')
list = [line.split('>') for line in scn.readlines()]
list.sort()
scn.close()
# Copy&Paste in the final scenario
scn2 = open('Data_final/'+name+'.scn', 'w')
for element in list:
    element=">".join(element)
    print(element)
    scn2.write(element)
scn2.close()

# Flight phase database for all flights, for conflict detection identification per flight phase
phase_data = actualdata[~pd.DataFrame(np.sort(actualdata[['icao24','callsign', 'PH']], axis=1), index=actualdata.index).duplicated()]
phase_data.drop("icao24", axis=1, inplace=True);phase_data.drop("lat", axis=1, inplace=True);phase_data.drop("lon", axis=1, inplace=True)#;phase_data.drop("vertrate", axis=1, inplace=True)
phase_data.drop("velocity", axis=1, inplace=True);phase_data.drop("heading", axis=1, inplace=True);phase_data.drop("AC reg", axis=1, inplace=True);phase_data.drop("ACmodel", axis=1, inplace=True)
phase_data.drop("geoaltitude", axis=1, inplace=True);phase_data.drop("Seq", axis=1, inplace=True);phase_data.drop("LAT", axis=1, inplace=True);phase_data.drop("LON", axis=1, inplace=True)
BL=phase_data.iat[0,0].timestamp()- (phase_data.iat[0,0].hour * 3600 + phase_data.iat[0,0].minute*60 + phase_data.iat[0,0].second)
for i in range(len(phase_data)):
    phase_data.iat[i,0]=phase_data.iat[i,0].timestamp()-BL

import pickle

var=open('Data_final/ph_'+name+'.pickle', 'wb')
pickle.dump(phase_data, var)
var.close()

# TO DELETE
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)


 #%%
#Organize data for time shifting in the genetic algorithm
#data_1 = list[0:7]
#data_1 = pd.DataFrame(data_1,columns=['Time','Rest'])
#data_2 = list[8:len(list)]
#data_2 = pd.DataFrame(data_2,columns=['Time','Rest'])
#Obtain a column with aircraft id to time shift later
#extract=data_2['Rest'].str.split(' ')
#aux=[extract[i][1] for i in range(len(data_2)-1)]
#aux.append('')
#data_2['AC']=aux
#AC=AC_ectrl_R
#Load variables
#import pickle
#var=open('Data_'+name+'.pickle', 'wb')
#pickle.dump(data_1, var)
#pickle.dump(data_2,var)
#pickle.dump(AC,var)
