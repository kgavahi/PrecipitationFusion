import numpy as np
import os
import sys
def loadProcessData(Dir, dset = 'train'):
    if dset == 'train' or dset == 'val':
        X = np.load(f'{Dir}/X_{dset}_{Dir[-1]}.npy')
        y = np.load(f'{Dir}/y_{dset}_{Dir[-1]}.npy')
        stacked_products_st = np.load(f'{Dir}/stacked_products_st_{dset}_{Dir[-1]}.npy')
    if dset == 'test':
        X = np.load(f'{Dir}/X_{dset}.npy')
        y = np.load(f'{Dir}/y_{dset}.npy')
        stacked_products_st = np.load(f'{Dir}/stacked_products_st_{dset}.npy')

    products_dict = {
            'IMERG': 0,
            'NLDAS': 1,
            'CMORPH': 2,
            'CPC': 3,
            'GSMaP': 4,
            'PERSIANN': 5,
            'CHIRPS': 6,
            'SM2RAIN': 7,
            'TRMM': 8,
        }
    
    
    # Select the products
    product_idx = [products_dict[product] for product in products_list]
    
    
    # slice X based on the window size
    cen_win_t = (X.shape[1] - 1) // 2
    cen_win_s = (X.shape[2] - 1) // 2
    
    tl = cen_win_t - w_t
    tr = cen_win_t + w_t + 1
    wl = cen_win_s - w_s
    wr = cen_win_s + w_s + 1
    
    X = X[:, tl:tr, wl:wr, wl:wr, product_idx]
    stacked_products_st = stacked_products_st[:, :, product_idx]

    # Remove NaN values 
    # create a Boolean mask for the NaN values in X
    mask_x = (np.isnan(X) | (X<0) | (X>10000))
    # check if there are any NaN values in each slice along the first dimension
    has_nan_x = np.any(mask_x, axis=(1, 2, 3, 4))

    # create a Boolean mask for the NaN values for y
    has_nan_y = (np.isnan(y) | (y<0) | (y>10000))

    has_nan_x_y = (has_nan_x | has_nan_y)

    # select only the slices that do not contain any NaN values
    X = X[~has_nan_x_y]
    y = y[~has_nan_x_y]    
    stacked_products_st = np.concatenate(stacked_products_st, axis=0)[~has_nan_x_y]
            
                                                    
    return X, y, stacked_products_st   



sen_dict = {
    's1': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 0, 'w_t': 0},
    's2': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 1, 'w_t': 0},
    's3': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 2, 'w_t': 0},
    's4': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 3, 'w_t': 0},
    's5': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 4, 'w_t': 0},
    's6': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 5, 'w_t': 0},
    's7': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 6, 'w_t': 0},
    's8': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 7, 'w_t': 0},
    's9': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 0, 'w_t': 1},
    's10': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 0, 'w_t': 2},
    's11': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 0, 'w_t': 3},
    's12': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 0, 'w_t': 4},
    's13': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 0, 'w_t': 5},
    's14': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 0, 'w_t': 6},
    's15': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 0, 'w_t': 7},
    's16': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 1, 'w_t': 1},
    's17': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 2, 'w_t': 2},
    's18': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 3, 'w_t': 3},
    's19': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 4, 'w_t': 4},
    's20': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 5, 'w_t': 5},
    's21': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 6, 'w_t': 6},
    's22': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 7, 'w_t': 7},
    

    's23': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 7, 'w_t': 5},    
    's24': {'products': ['IMERG', 'CMORPH', 'CPC', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 7, 'w_t': 5},  
    's25': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 7, 'w_t': 5},       
    's26': {'products': ['IMERG', 'CMORPH', 'GSMaP', 'PERSIANN', 'CHIRPS'], 'w_s': 7, 'w_t': 5},     
    's27': {'products': ['IMERG', 'NLDAS', 'CMORPH', 'CPC', 'PERSIANN', 'CHIRPS'], 'w_s': 7, 'w_t': 5},    
    's28': {'products': ['NLDAS', 'CPC'], 'w_s': 7, 'w_t': 5},     
    's29': {'products': ['CPC'], 'w_s': 7, 'w_t': 5},     
    's30': {'products': ['NLDAS'], 'w_s': 7, 'w_t': 5},     


    's31': {'products': ['IMERG', 'PERSIANN', 'SM2RAIN', 'GSMaP'], 'w_s': 7, 'w_t': 5},
    's32': {'products': ['IMERG', 'PERSIANN', 'SM2RAIN'], 'w_s': 7, 'w_t': 5},
    's33': {'products': ['IMERG', 'PERSIANN'], 'w_s': 7, 'w_t': 5},
    's34': {'products': ['PERSIANN', 'SM2RAIN'], 'w_s': 7, 'w_t': 5},
    's35': {'products': ['PERSIANN'], 'w_s': 7, 'w_t': 5},    
    's36': {'products': ['SM2RAIN'], 'w_s': 7, 'w_t': 5},  
    's37': {'products': ['IMERG'], 'w_s': 7, 'w_t': 5},
    's38': {'products': ['GSMaP'], 'w_s': 7, 'w_t': 5},
    's39': {'products': ['IMERG', 'PERSIANN', 'SM2RAIN', 'GSMaP'], 'w_s': 0, 'w_t': 0}  
    
}



    

scenario = sys.argv[1]
fold = sys.argv[2]

path = f'scenarios/{scenario}/{fold}'

products_list = sen_dict[scenario]['products']
w_t = sen_dict[scenario]['w_t'] # n days before, n days after
w_s = sen_dict[scenario]['w_s'] # n cells from each side
w_i = len(products_list) 


print(f'scenario: {scenario}, fold: {fold}, w_s={w_s}, w_t={w_t}')

if not os.path.exists(path): os.makedirs(path)

X_train, y_train, stacked_products_st_train = loadProcessData(f'CV8/{fold}', dset = 'train')
X_val, y_val, stacked_products_st_val = loadProcessData(f'CV8/{fold}', dset = 'val')
X_test, y_test, stacked_products_st_test = loadProcessData(f'CV8/test', dset = 'test')   


np.save(f'{path}/X_train', X_train)
np.save(f'{path}/y_train', y_train)
np.save(f'{path}/stacked_products_st_train', stacked_products_st_train)

np.save(f'{path}/X_val', X_val)
np.save(f'{path}/y_val', y_val)
np.save(f'{path}/stacked_products_st_val', stacked_products_st_val)

np.save(f'{path}/X_test', X_test)
np.save(f'{path}/y_test', y_test)
np.save(f'{path}/stacked_products_st_test', stacked_products_st_test)