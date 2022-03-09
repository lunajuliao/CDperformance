import pandas as pd
import numpy as np
import sys
import os
#
# Function trkangle : Auxiliar function for track evaluation - transforms the track angle to a value between 0º and 180º
#
def trkangle(angle):
    angle=abs(angle)%180
    return angle
#
# Function f_single : Auxiliar function for the fitness evalutation - returns the result for each parameter based on the upper and lower bounds
#
def f_single(count, upper, lower):
    if count > upper:
        f = upper / count
    elif count < lower:
        f = count / lower
    else:
        f = 1
    return f

def exception_handler(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit()


sys.excepthook = exception_handler

def main(id):
    # Select automatically the file for logging- the last one with the id input
    list_out = os.listdir(r"C:\Users\lunap\PycharmProjects\pythonProject1\testserv\output") # CHANGE DIRECTORY TO YOUR OWN
    scn_d = 'DLOG_t_gen_' + str(id)
    res_d = [i for i in list_out if scn_d in i]
    output_dist_file = res_d[len(res_d) - 1]
    output_dist = open(r"C:/Users/lunap/PycharmProjects/pythonProject1/testserv/output/" + output_dist_file, "r")
    data_dist = output_dist.read()
    data_dist = data_dist.splitlines()
    data_dist = [i.split(',') for i in data_dist]
    data_dist = data_dist[2:]
    data_dd = pd.DataFrame(data_dist, columns=['Time', 'AC1', 'AC2', 'Dist', 'DCPA', 'TCPA', 'QDR', 'TRK1', 'TRK2', 'PH1', 'PH2','ALT1', 'ALT2'])
    # Eliminate GROUND aircraft
	data_dd = data_dd[data_dd.PH1 != '6.0']
    data_dd = data_dd[data_dd.PH2 != '6.0']
	
	# Delete variable to minimise memory issues
    del data_dist

    # Computations to obtain new parameters required for the analysis
    data_dd['TRK'] = trkangle(data_dd['TRK1'].astype(float) - data_dd['TRK2'].astype(float))
    data_dd['Dist'] = data_dd['Dist'].astype(float) / 1852
    data_dd['ALT'] = (data_dd['ALT1'].astype(float) - data_dd['ALT2'].astype(float)) / 0.3048 #m to ft
	
	# Delete variable to minimise memory issues
    del data_dd['DCPA'],data_dd['TCPA'],data_dd['QDR'],data_dd['TRK1'],data_dd['TRK2'],data_dd['ALT1'],data_dd['ALT2']
	
    # Obtain a database with the first conflict alarm for each set of aircraft
    data_d = data_dd[~pd.DataFrame(np.sort(data_dd[['AC1', 'AC2']], axis=1), index=data_dd.index).duplicated()]
    unique = data_d.drop_duplicates(['AC1', 'AC2'])
    data_final = pd.DataFrame([])
	
    for i in range(len(unique)): # Process for each aircraft encounter pair
        print(i)
        aux = data_dd.loc[(data_dd['AC1'] == unique.iat[i, 1]) & (data_dd['AC2'] == unique.iat[i, 2])]  # select all the data logged for the specific encounter
        # evaluate if there is more than one encounter with the same two aircraft, spaced in time and allowin for an encounter flag to be off for X (300) seconds
        if (float(aux.iat[0, 0]) + (len(aux) - 1) - float(aux.iat[len(aux) - 1, 0])) <= 300:
            ap = pd.DataFrame(aux[aux.Dist == aux.Dist.min()]) # select the minimum distance data entry for each encounter
            # select only the first data entry of the minimum distance database
            data_final = pd.concat([data_final, ap.head(1)])
        else:
            print('inside')
            start = 0
            for j in range(len(aux) - 1):   # go row by row to assess the time difference between datalog entries - understand if there are more than one encounters between the 2 same AC
                if (float(aux.iat[j + 1, 0]) - float(aux.iat[j, 0]) > 290) or (j + 1 == len(aux) - 1):
                    aux_aux = aux[start:j]
                    ap = pd.DataFrame(aux_aux[aux_aux.Dist == aux_aux.Dist.min()])
                    data_final = pd.concat([data_final, ap.head(1)])
                    start = j + 1

    data_d = pd.DataFrame(data_final)
    
    # Categorize all the encounters into 4 different bin types - horizontal, vertical, encounter angle and encounter geometry
    count_conf = len(data_d)
    count_5nm = (data_d.Dist.astype(float) <= 5).sum()
    count_10nm = ((data_d.Dist.astype(float) > 5) & (data_d.Dist.astype(float) <= 10)).sum()
    count_15nm = ((data_d.Dist.astype(float) > 10) & (data_d.Dist.astype(float) <= 15)).sum()
    count_20nm = ((data_d.Dist.astype(float) > 15) & (data_d.Dist.astype(float) <= 20)).sum()
    count_25nm = ((data_d.Dist.astype(float) > 20) & (data_d.Dist.astype(float) <= 25)).sum()
    count_500ft = (data_d.ALT.astype(float) <= 500).sum()
    count_1000ft = ((data_d.ALT.astype(float) > 500) & (data_d.ALT.astype(float) <= 1000)).sum()
    count_2000ft = ((data_d.ALT.astype(float) > 1000) & (data_d.ALT.astype(float) <= 2000)).sum()
    count_3000ft = ((data_d.ALT.astype(float) > 2000) & (data_d.ALT.astype(float) <= 3000)).sum()
    count_4000ft = ((data_d.ALT.astype(float) > 3000) & (data_d.ALT.astype(float) <= 4000)).sum()
    count_5000ft = ((data_d.ALT.astype(float) > 4000) & (data_d.ALT.astype(float) <= 5000)).sum()
    count_30º = ((data_d.TRK.astype(float) >= 0) & (data_d.TRK.astype(float) <= 30)).sum()
    count_60º = ((data_d.TRK.astype(float) > 30) & (data_d.TRK.astype(float) <= 60)).sum()
    count_90º = ((data_d.TRK.astype(float) > 60) & (data_d.TRK.astype(float) <= 90)).sum()
    count_120º = ((data_d.TRK.astype(float) > 90) & (data_d.TRK.astype(float) <= 120)).sum()
    count_150º = ((data_d.TRK.astype(float) > 120) & (data_d.TRK.astype(float) <= 150)).sum()
    count_180º = ((data_d.TRK.astype(float) > 150) & (data_d.TRK.astype(float) <= 180)).sum()
    count_L_L = ((data_d.PH1.astype(float) == 4) & (data_d.PH2.astype(float) == 4)).sum()
    count_L_T = ((data_d.PH1.astype(float) == 4) & (data_d.PH2.astype(float) != 4)).sum() + ((data_d.PH1.astype(float) != 4) & (data_d.PH2.astype(float) == 4)).sum()
    count_T_T = ((data_d.PH1.astype(float) != 4) & (data_d.PH2.astype(float) != 4)).sum()

    Fit = [0 for i in range(21)]
    # Define Lower and Upper limits for the fitness classification - 10% margin from the nominal considered
    Lower = [int(5880*0.9), int(21.66*.9), int(18.31*.9), int(17.80*.9), int(20.17*.9), int(22.05*.9), int(59.51*.9), int(6.41*.9),int(7.63*.9), int(9.44*.9), int(5.97*.9), int(11.04*.9), int(40.64*.9),int(13.43*.9), int(14.64*.9), int(11.60*.9),int(8.06*.9), int(11.63*.9), int(41.98*.9),
                 int(23.28*.9), int(30.74*.9)]  # dividing by 8
    Upper = [int(5880*1.1), int(21.66*1.1), int(18.31*1.1), int(17.80*1.1), int(20.17*1.1), int(22.05*1.1), int(59.51*1.1), int(6.41*1.1),int(7.63*1.1), int(9.44*1.1), int(5.97*1.1), int(11.04*1.1), int(40.64*1.1),int(13.43*1.1), int(14.64*1.1), int(11.60*1.1),int(8.06*1.1), int(11.63*1.1), int(41.98*1.1),
                 int(23.28*1.1), int(34.74*1.1)]
    Count = [count_conf, count_5nm, count_10nm, count_15nm, count_20nm, count_25nm,count_500ft,count_1000ft, count_2000ft, count_3000ft, count_4000ft, count_5000ft, count_30º, count_60º, count_90º,count_120º, count_150º, count_180º,count_L_L,count_L_T,count_T_T]
    # First element is total count but the following are all percentual values
    Percentage = [count_conf, 100*count_5nm / count_conf, 100*count_10nm / count_conf, 100*count_15nm / count_conf,
                      100*count_20nm / count_conf, 100*count_25nm / count_conf,100*count_500ft / count_conf, 100*count_1000ft / count_conf,
                      100*count_2000ft / count_conf, 100*count_3000ft / count_conf, 100*count_4000ft / count_conf,
                      100*count_5000ft / count_conf, 100*count_30º / count_conf, 100*count_60º / count_conf, 100*count_90º / count_conf,
                      100*count_120º / count_conf, 100*count_150º / count_conf, 100*count_180º / count_conf, 100*count_L_L / count_conf,
                      100*count_L_T / count_conf, 100*count_T_T / count_conf]

	# Calculate for each parameter its fitness value
    for i in range(len(Fit)): 
            Fit[i] = f_single(Percentage[i], Upper[i], Lower[i])

    result = sum(Fit)/21 # Calculate  overall fitness value - 21 stands for the 21 parameters evaluated

	# Save the result to be read in a different script
    var=open('result'+id+'.pickle','wb')
    pickle.dump(result,var)
    var.close()


main(sys.argv[1]) # when calling the script, input id has to be in the call