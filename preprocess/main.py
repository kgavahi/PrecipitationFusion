import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import datetime
from netCDF4 import Dataset
import sys
import time
import xarray as xr
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
def crtDirs(path):
    PATH_TEST = f'{path}/test'
    if not os.path.exists(PATH_TEST): os.mkdir(PATH_TEST)
    for k in range(3):
        if not os.path.exists(f'{path}/{k}'): os.mkdir(f'{path}/{k}')
        PATH_TRAIN = f'{path}/{k}/train'   
        if not os.path.exists(PATH_TRAIN): os.mkdir(PATH_TRAIN)
        PATH_VAL = f'{path}/{k}/val'   
        if not os.path.exists(PATH_VAL): os.mkdir(PATH_VAL)    
def kFoldIndices(n, test_size=0.25, n_splits=3):
    
    idxs = np.arange(n)
    
    test_idx = idxs[int((1-test_size)*n):]
    
    tr_val_idx = idxs[:int((1-test_size)*n)]
    
    ## cross-validation
    kfolds = []
    kf = KFold(n_splits=n_splits)
    for K, (train_index, val_index) in enumerate(kf.split(tr_val_idx)):
    
        
        kfolds.append([train_index, val_index])
        

    return test_idx, kfolds
def chunkData(data, chunk_size):
    n = int(np.ceil(len(data)/chunk_size))
    print(n)
    # using list comprehension
    return [data[i * n:(i + 1) * n] for i in range((len(data) + n - 1) // n )]
def split(a, n):
    k, m = divmod(len(a), n)
    return list((a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)))
def timeDeltaWalk(date, delta):
	date = datetime.date(
	int(str(date)[ :4]), 
	int(str(date)[5:7]), 
	int(str(date)[8:10]))
	date = str(date + datetime.timedelta(days = delta))
	date = date.replace('-','')
	return date
def ReadIMERG(start_date, end_date):
    PATH_IMERG = '/mh1/kgavahi/Paper4/Download/IMERG_F/IMERG/GPM_3IMERGDF-06'
    date_window = pd.date_range(start_date, end_date, freq='D')
    imerg_nc_files = [f'{PATH_IMERG}/3B-DAY.MS.MRG.3IMERG.%s-S000000-E235959.V06.nc4'%(str(d)[:10].replace('-','')) for d in date_window]
    
    imerg_xarray = xr.open_mfdataset(imerg_nc_files, concat_dim='time', combine='nested')

    imerg_clipped = imerg_xarray.isel(lon=(imerg_xarray.lon >= left) & (imerg_xarray.lon <= right),
                              lat=(imerg_xarray.lat >= bottom) & (imerg_xarray.lat <= top),
                              ).transpose()

    return imerg_clipped
def ReadNLDAS_h(start_date, end_date):
    
    PATH_NLDAS = '/mh1/kgavahi/Paper4/Download/NLDAS_FORC/all_years'
    date_window = pd.date_range(start_date, end_date, freq='h')  
    nldas_nc_files = [f'{PATH_NLDAS}/NLDAS_FORA0125_H.A%s.%s00.002.grb.SUB.nc4'%(str(d)[:10].replace('-',''), str(d)[11:13]) for d in date_window]    
    nldas_xarray = xr.open_mfdataset(nldas_nc_files, concat_dim='time', combine='nested')
    nldas_xarray = nldas_xarray.resample(time="1D").sum(skipna=False)
    nldas_clipped = nldas_xarray.isel(lon=(nldas_xarray.lon >= left) & (nldas_xarray.lon <= right),
                              lat=(nldas_xarray.lat >= bottom) & (nldas_xarray.lat <= top),
                              )

    return nldas_clipped
def ReadNLDAS(start_date, end_date):
    
    PATH_NLDAS = '/mh1/kgavahi/Paper4/NLDAS_FORC_daily'
    date_window = pd.date_range(start_date, end_date, freq='D')  
    nldas_nc_files = [f'{PATH_NLDAS}/NLDAS_FORA0125_H.A%s.002.grb.SUB.nc4'%(str(d)[:10].replace('-','')) for d in date_window]
    nldas_xarray = xr.open_mfdataset(nldas_nc_files, concat_dim='time', combine='nested')
    nldas_clipped = nldas_xarray.isel(lon=(nldas_xarray.lon >= left) & (nldas_xarray.lon <= right),
                              lat=(nldas_xarray.lat >= bottom) & (nldas_xarray.lat <= top),
                              )

    return nldas_clipped
def ReadTRMM(start_date, end_date):
    PATH_TRMM = '/mh1/kgavahi/Paper4/Download/TRMM/TRMM'
    date_window = pd.date_range(start_date, end_date, freq='D')
    trmm_nc_files = [f'{PATH_TRMM}/3B42RT_Daily.%s.7.nc4'%(str(d)[:10].replace('-','')) for d in date_window]
    trmm_xarray = xr.open_mfdataset(trmm_nc_files, concat_dim='time', combine='nested')
    trmm_clipped = trmm_xarray.isel(lon=(trmm_xarray.lon >= left) & (trmm_xarray.lon <= right),
                              lat=(trmm_xarray.lat >= bottom) & (trmm_xarray.lat <= top),
                              ).transpose()
    return trmm_clipped
def ReadCMORPH(start_date, end_date):
    PATH_CMORPH = '/mh1/kgavahi/Paper4/Download/CMORPH'
    date_window = pd.date_range(start_date, end_date, freq='D')
    cmorph_nc_files = [f'{PATH_CMORPH}/CMORPH_V1.0_ADJ_0.25deg-DLY_00Z_%s.nc'%(str(d)[:10].replace('-','')) for d in date_window]
    cmorph_xarray = xr.open_mfdataset(cmorph_nc_files, concat_dim='time', combine='nested')
    cmorph_xarray = cmorph_xarray.assign_coords(lon=(((cmorph_xarray.lon + 180) % 360) - 180))
    cmorph_clipped = cmorph_xarray.isel(lon=(cmorph_xarray.lon >= left) & (cmorph_xarray.lon <= right),
                              lat=(cmorph_xarray.lat >= bottom) & (cmorph_xarray.lat <= top),
                              )
    return cmorph_clipped
def ReadCPC(start_date, end_date):
    PATH_CPC = '/mh1/kgavahi/Paper4/Download/CPC'
    date_window = pd.date_range(start_date, end_date, freq='D')
    cpc_nc_files = [f'{PATH_CPC}/precip.V1.0.%s.nc'%(str(d)[:4].replace('-','')) for d in date_window]
    cpc_nc_files = sorted(set(cpc_nc_files))
    cpc_xarray = xr.open_mfdataset(cpc_nc_files, concat_dim='time', combine='nested')   
    cpc_xarray = cpc_xarray.sel(dict(time=slice(start_date, end_date)))
    cpc_xarray = cpc_xarray.assign_coords(lon=(((cpc_xarray.lon + 180) % 360) - 180))    
    cpc_clipped = cpc_xarray.isel(lon=(cpc_xarray.lon >= left) & (cpc_xarray.lon <= right),
                              lat=(cpc_xarray.lat >= bottom) & (cpc_xarray.lat <= top),
                              )
    
    return cpc_clipped
def ReadGSMaP(start_date, end_date):
    PATH_GSMaP = '/mh1/kgavahi/Paper4/Download/GSMaP/nc_daily0.1_00Z-23Z'
    date_window = pd.date_range(start_date, end_date, freq='D')
    gsmap_nc_files = [f'{PATH_GSMaP}/gsmap_nrt.%s.0.1d.daily.00Z-23Z.dat.gz.nc'%(str(d)[:10].replace('-','')) for d in date_window]
    gsmap_clipped = xr.open_mfdataset(gsmap_nc_files, concat_dim='time', combine='nested')

    return gsmap_clipped
def ReadPERSIANN(start_date, end_date):
    PATH_PERSIANN = '/mh1/kgavahi/Paper4/Download/PERSIANN/PERSIANN-CDR'
    date_window = pd.date_range(start_date, end_date, freq='D')
    persiann_nc_files = [f'{PATH_PERSIANN}/CDR_2022-04-17030747pm_%s.nc'%(str(d)[:4].replace('-','')) for d in date_window]
    persiann_nc_files = sorted(set(persiann_nc_files))
    persiann_xarray = xr.open_mfdataset(persiann_nc_files, concat_dim='datetime', combine='nested')  
    persiann_xarray = persiann_xarray.sel(dict(datetime=slice(start_date, end_date)))
    persiann_clipped = persiann_xarray.isel(lon=(persiann_xarray.lon >= left) & (persiann_xarray.lon <= right),
                              lat=(persiann_xarray.lat >= bottom) & (persiann_xarray.lat <= top),
                              )
    return persiann_clipped
def ReadCHIRPS(start_date, end_date):
    PATH_CHIRPS = '/mh1/kgavahi/Paper4/Download/CHIRPS/p05_daily'
    date_window = pd.date_range(start_date, end_date, freq='D')
    chirps_nc_files = [f'{PATH_CHIRPS}/chirps-v2.0.%s_p05.nc'%(str(d)[:10].replace('-','')) for d in date_window]
    chirps_nc_files = sorted(set(chirps_nc_files))
    chirps_clipped = xr.open_mfdataset(chirps_nc_files, concat_dim='time', combine='nested')  

    
    return chirps_clipped    
def ReadSM2RAIN(start_date, end_date):
    PATH_SM2RAIN = '/mh2/data/SM2RAIN-ASCAT/processed'
    date_window = pd.date_range(start_date, end_date, freq='D')
    sm2rain_nc_files = [f'{PATH_SM2RAIN}/SM2RAIN_ASCAT_0125_%s_v1.5_pr.nc'%(str(d)[:4].replace('-','')) for d in date_window]
    sm2rain_nc_files = sorted(set(sm2rain_nc_files))
    sm2rain_xarray = xr.open_mfdataset(sm2rain_nc_files, concat_dim='time', combine='nested')   
    sm2rain_clipped = sm2rain_xarray.sel(dict(time=slice(start_date, end_date)))

    
    return sm2rain_clipped
def FindLocationIndex(da, lat_gauge, lon_gauge):
    
    try:
        abslat = np.abs(da.lat - lat_gauge)
        abslon = np.abs(da.lon - lon_gauge)
    except:
        abslat = np.abs(da.latitude - lat_gauge)
        abslon = np.abs(da.longitude - lon_gauge)        
    
    lat_i_da = np.argmin(np.array(abslat))
    lon_i_da = np.argmin(np.array(abslon))
    
    return lat_i_da, lon_i_da   
def CheckForNaNs(a):
       
    t = ma.masked_invalid(a)

    return np.any((a < 0)|(a > 100000)) or np.any(t.mask)
def windowGenerator(da, da_np, lat_gauge, lon_gauge, product='IMERG'):
    ## Reading i, j index of station's location on NLDAS
    lat_ix, lon_ix = FindLocationIndex(da, lat_gauge, lon_gauge)
    
    
    tw = lat_ix+(w_s+1) # index of the top of the window
    bw = lat_ix-w_s     # index of the bottom of the window
    lw = lon_ix-w_s     # index of the left of the window
    rw = lon_ix+(w_s+1) # index of the right of the window
    
    if product == 'IMERG' or product == 'TRMM' or product == 'SM2RAIN':
        ## Clipping the window around the station
        window = da_np[bw:tw, lw:rw, :]
        window = np.moveaxis(np.array(window), 2, 0)
    else:       
        window = da_np[:, bw:tw, lw:rw]      
    
    if product == 'GSMaP' or product == 'PERSIANN':

        ## flip the array to be consistent with other prodcuts
        window = np.flip(window, axis=1)
    
    if not window.shape == (w_t*2+1, w_s*2+1, w_s*2+1):
        #print(f'window is out of bound for {product}')
        window = np.zeros([w_t*2+1, w_s*2+1, w_s*2+1]) * np.nan
            
    return window
class DataLoader:
    def __init__(self, products_dict, products_list, filtered_stations, w_t, w_s):
        self.products_dict = products_dict
        self.products_list = products_list
        self.filtered_stations = filtered_stations
        self.w_t = w_t
        self.w_s = w_s
        self.w_i = len(products_list)

    def load_data(self, day):
        middle_date = str(day)[:10]
        start_date_window = timeDeltaWalk(middle_date, -self.w_t)
        end_date_window = timeDeltaWalk(middle_date, self.w_t)

        print(start_date_window, day, end_date_window)

        s = time.time()
        # loop through the products and read the precip data for each one
        for product in self.products_list:

            da = self.products_dict[product]['func'](start_date_window, end_date_window)
            np_arr = np.array(getattr(da, self.products_dict[product]['atr']))

            # The reason to have poth precip_da and precip_np is to reduce
            # the number of .to_numpy() operations in the following (loop over stations).
            # This will significantly reduce the runtime at the cost of higher memory 
            # usage.

            self.products_dict[product]['da'] = da
            self.products_dict[product]['np'] = np_arr

        readtime = time.time() - s

        InputImage = np.zeros([len(self.filtered_stations), self.w_t*2+1, self.w_s*2+1, self.w_s*2+1, self.w_i], dtype='float32') * np.nan
        products_values_over_stations = np.zeros([len(self.filtered_stations), self.w_i], dtype='float32') * np.nan
        labels = np.zeros([len(self.filtered_stations)], dtype='float32') * np.nan
        lat_lons = np.zeros([len(self.filtered_stations), 2])* np.nan

        s=time.time()
        for df_c, station_data in enumerate(self.filtered_stations):

            ## Load the corresponding DataFrame for that station
            df = station_data[3]


            try:
                label = df.loc[df['DATE'] == middle_date, 'PRCP'].values[0].astype(int)
                ## ValidRange check
                if label < 0 or label > 100000:
                    label = np.nan

            except:
                ## No data recorded on this date for this station
                label = np.nan



            lat_gauge = station_data[1]
            lon_gauge = station_data[2]
            lat_lons[df_c] = lat_gauge, lon_gauge


            for product_ix, product in enumerate(self.products_list):

                np_arr = self.products_dict[product]['np']
                da = self.products_dict[product]['da']         

                window = windowGenerator(da, np_arr, lat_gauge, lon_gauge, product)
                InputImage[df_c, :, :, :, product_ix] = window
                products_values_over_stations[df_c, product_ix] = InputImage[df_c, self.w_t, self.w_s, self.w_s, product_ix]        


            labels[df_c] = label / 10
            
        return InputImage, products_values_over_stations, labels, lat_lons



        
products_dict = {
        'IMERG': {'da': None, 'np': None, 'func': ReadIMERG, 'atr':'precipitationCal'},
        'NLDAS': {'da': None, 'np': None, 'func': ReadNLDAS, 'atr':'APCP'},
        'TRMM': {'da': None, 'np': None, 'func': ReadTRMM, 'atr':'precipitation'},
        'CMORPH': {'da': None, 'np': None, 'func': ReadCMORPH, 'atr':'cmorph'},
        'CPC': {'da': None, 'np': None, 'func': ReadCPC, 'atr':'precip'},
        'GSMaP': {'da': None, 'np': None, 'func': ReadGSMaP, 'atr':'precip'},
        'PERSIANN': {'da': None, 'np': None, 'func': ReadPERSIANN, 'atr':'precip'},
        'CHIRPS': {'da': None, 'np': None, 'func': ReadCHIRPS, 'atr':'precip'},
        'SM2RAIN': {'da': None, 'np': None, 'func': ReadSM2RAIN, 'atr':'Rainfall'},
    }

products_list = ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS', 'SM2RAIN']


scenario = 'sall'
w_t = 10 # n days before, n days after
w_s = 10 # n cells from each side
w_i = len(products_list) # number of precipitation data (IMERG, NLDAS)

## the boundaries of CONUS
top = 55
bottom = 20
left = -130
right = -65



filtered_stations = np.load('filtered_stations.npy', allow_pickle=True)

test_idx, kfolds = kFoldIndices(len(filtered_stations), test_size=0.25, n_splits=3)

dateRange = pd.date_range('2007-01-15', '2020-12-01', freq='D') #for 2020 yr can go beyond 2021 and result in error in higher w_t


# Split the dates into 1000 chunks. 
# Each chunk will be handled by one core
chunks = np.array_split(dateRange, 1000)


processor = int(sys.argv[1])
print(chunks[processor])


loader = DataLoader(products_dict, 
                         products_list, 
                         filtered_stations, 
                         w_t, 
                         w_s)

save_path = 'data8'
if processor==0: crtDirs(save_path)


for day in chunks[processor]:
    
    
    
    InputImage, products_values_over_stations, labels, lat_lons = loader.load_data(day)

    # Save all the arrays in one array to reduce the number of I/O
    # operations needed. 
    arr_to_save = np.array([str(day)[:10], 
                            lat_lons[test_idx],
                            labels[test_idx], 
                            products_values_over_stations[test_idx], 
                            InputImage[test_idx]], 
                            dtype=object)
    PATH_TEST = f'{save_path}/test'
    np.save(f'{PATH_TEST}/la_pvos_im_{str(day)[:10]}', arr_to_save)
    
    for K, (train_index, val_index) in enumerate(kfolds):

        arr_to_save = np.array([str(day)[:10], 
                                lat_lons[train_index],        
                                labels[train_index], 
                                products_values_over_stations[train_index], 
                                InputImage[train_index]], 
                                dtype=object)
        PATH_TRAIN = f'{save_path}/{K}/train' 
        np.save(f'{PATH_TRAIN}/la_pvos_im_{str(day)[:10]}', arr_to_save)        
    
        arr_to_save = np.array([str(day)[:10], 
                                lat_lons[val_index],
                                labels[val_index], 
                                products_values_over_stations[val_index], 
                                InputImage[val_index]], dtype=object)
        PATH_VAL = f'{save_path}/{K}/val'   
        np.save(f'{PATH_VAL}/la_pvos_im_{str(day)[:10]}', arr_to_save)

    




   
    

    
    












