import pandas as pd
import numpy as np
import random
import os

# This code reads station precipitation data files for US stations from GHCNd dataset.
# It filters only files that contain 'US' in their names and shuffles the list of selected files randomly.
# Then it reads each CSV file into a pandas DataFrame, extracts the station name, latitude, and longitude for the first row.
# If the DataFrame contains columns with DATE and PRCP data, the program appends the 
# station name, latitude, longitude, and DataFrame to a list called df_array.
# Finally, the program saves the df_array as a numpy file named 'us_stations.npy'.
# By using the 'us_stations.npy' instead of reading each CSV file the preprocessing will become much faster.

files = os.listdir('/mh1/kgavahi/Paper4/stations')
files = [x for x in files if 'US' in x]

random.seed(0)
random.shuffle(files)


df_array = []

for count, file in enumerate(files):

    df = pd.read_csv(f'/mh1/kgavahi/Paper4/stations/{file}', low_memory=False)
    lat_gauge = df.LATITUDE.iloc[0]
    lon_gauge = df.LONGITUDE.iloc[0]  
    station = df.STATION.iloc[0]
    
    
    try:
        df = df[['DATE', 'PRCP']]
    except:
        print('Station does not report PRCP (SNOW maybe)')
        continue

    
  
    df_array.append([station, lat_gauge, lon_gauge, df])
    
    
    
    print(count)
np.save('us_stations.npy', df_array)