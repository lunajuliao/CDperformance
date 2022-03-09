from pyopensky import OpenskyImpalaWrapper
import tkinter as tk
import numpy as np
import pandas as pd
import pickle
import os

root = tk.Tk()
root.withdraw()

# %%
opensky = OpenskyImpalaWrapper() # Open the connection with the server
date="2018-09-02" # pick date - it can be directly written in the command

# Perform a query with time and geographical box filter 
# Divide into subsets due to the duration of the download (it can be too much)
df1 = opensky.query(
    type="adsb",
    start=date + " 08:00:00",
    end=date + " 9:30:00",
    bound=[50.797719,1.408212, 53.886976, 7.417641]
)

del df1["squawk"]
df1 = df1.dropna()

df2 = opensky.query(
    type="adsb",
    start=date + " 09:30:01",
    end=date + " 11:00:00",
    bound=[50.797719,1.408212, 53.886976, 7.417641]
)

del df2["squawk"]
df2 = df2.dropna()

df3 = opensky.query(
    type="adsb",
    start=date + " 11:00:01",
    end=date + " 12:30:00",
    bound=[50.797719,1.408212, 53.886976, 7.417641]
)

del df3["squawk"]
df3 = df3.dropna()

df4 = opensky.query(
    type="adsb",
    start=date + " 12:30:01",
    end=date + " 14:00:00",
    bound=[50.797719,1.408212, 53.886976, 7.417641]
)

del df4["squawk"]
df4 = df4.dropna()

dfnew = pd.concat([df1,df2,df3,df4]) # Concatenate all the sub databases downloaded

# Save the downloaded data in a pickle file to later load and process
os.chdir(r"C:\Users\lunap\PycharmProjects\pythonProject1\FINAL") # CHANGE DIRECTORY
with open("6h.pickle", "wb") as f:
    pickle.dump(dfnew, f)

