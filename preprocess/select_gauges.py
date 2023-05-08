import numpy as np
import pandas as pd
import os
import datetime
from netCDF4 import Dataset
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random

    
def distance(LatLons, lat_st, lon_st):


    lats = np.array(LatLons)[:, 0]
    lons = np.array(LatLons)[:, 1]

    
    d = np.sqrt((lats - lat_st)**2 + (lons - lon_st)**2) 
    
    return np.min(d)
def insideConus(lat_gauge, lon_gauge):
    flag = False
    #CONUS up, down, left, and right coordinates
    if (22 < lat_gauge < 51) and (-127 < lon_gauge < -64):
        flag = True
        
    return flag
def missingValues(df):
    # convert the 'Date' column to datetime format
    df['DATE']= pd.to_datetime(df['DATE'])
    
    # Save lat lons
    '''LatLons_no_filter.append([lat_gauge, lon_gauge])'''
    
    NumDays = (end_date - start_date).days+1

    
    # Filter all rows before the start_date
    df_filtered = df[df['DATE'] >= start_date]   
    df_filtered = df_filtered[df_filtered['DATE'] <= end_date]
    
    total_rows = df_filtered.shape[0]
    try:
        total_nans = df_filtered.isnull().sum(axis = 0)['PRCP']
    except:
        print('Station does not report PRCP (SNOW maybe)')
        total_nans = total_rows
    
    total_data_points = total_rows - total_nans
    
    return total_data_points/NumDays
    

########################################### Run this code using pyngl environment in conda ############################################

w_t = 3 # 3 days before, 3 days after
w_s = 3 # 3 cells from each side

cell_size = 0.1

min_distance = np.sqrt(2)*(w_s+0.5)*cell_size



start_date = datetime.date(2003, 1, 1)
end_date = datetime.date(2020, 12, 31)

per_available = 0.90




df_array = np.load('us_stations.npy', allow_pickle=True)



LatLons = [[0, 0]]
filtered_stations = []


for count, station_data in enumerate(df_array):
    print(count)   
    
    
    station = station_data[0]
    lat_gauge = station_data[1]
    lon_gauge = station_data[2]
    df = station_data[3]

    
    d = distance(LatLons, lat_gauge, lon_gauge)
    
    
    
    if (missingValues(df) > per_available) and (d > min_distance) and insideConus(lat_gauge, lon_gauge):
    
        
        LatLons.append([lat_gauge, lon_gauge])
        filtered_stations.append(station_data)
    
    
    
np.savetxt('LatLons_all3.csv', LatLons, delimiter=',')    
np.save('filtered_stations3.npy', filtered_stations)










