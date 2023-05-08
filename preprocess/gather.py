import numpy as np
import pandas as pd
import os
import sys
class StackFilesInDir:
    
    def __init__(self, Dir, w_t=10, w_s=10, w_i=7):
        self.Dir = Dir
        self.w_t = w_t
        self.w_s = w_s
        self.w_i = w_i
    
    def getLatLons(self):
        lat_lons = np.load(f'{self.Dir}/la_pvos_im_2008-01-31.npy', allow_pickle=True)[1]
        
        return lat_lons
    
    def stackFiles(self, date_range):
        sample_file = np.load(f'{self.Dir}/la_pvos_im_{str(date_range[0])[:10]}.npy', allow_pickle=True)
        
        X_shape = sample_file[4].shape
        X_shape = (len(date_range), ) + X_shape
        
        
        y_shape = sample_file[2].shape
        y_shape = (len(date_range), ) + y_shape

        pr_shape = sample_file[3].shape
        pr_shape = (len(date_range), ) + pr_shape
        
        stacked_x = np.empty(X_shape, dtype='float32')
        stacked_labels = np.empty(y_shape, dtype='float32')
        stacked_products = np.empty(pr_shape, dtype='float32')

            
        for counter, date in enumerate(date_range):
            full_data = np.load(f'{self.Dir}/la_pvos_im_{str(date)[:10]}.npy', allow_pickle=True)
            
            stacked_x[counter] = full_data[4]
            stacked_labels[counter] = full_data[2]
            stacked_products[counter] = full_data[3]
            
        
        stacked_x = stacked_x.reshape(X_shape[0]*X_shape[1], X_shape[2], X_shape[3], X_shape[4], X_shape[5])
        stacked_y = stacked_labels.reshape(y_shape[0]*y_shape[1])
        

        
        return stacked_x, stacked_y, stacked_labels, stacked_products



# Specifying date ranges
dateRange_test = pd.date_range('2017-01-01', '2020-12-01', freq='D')
dateRange_train = pd.date_range('2007-01-15', '2011-12-31', freq='D')
dateRange_val = pd.date_range('2012-01-01', '2016-12-31', freq='D')


processor = int(sys.argv[1])

# Test dataset
if processor==0:
    stacker_test = StackFilesInDir('data8/test')
    stacked_x, stacked_y, stacked_labels, stacked_products = stacker_test.stackFiles(dateRange_test)

    np.save('CV8/test/X_test', stacked_x)
    np.save('CV8/test/y_test', stacked_y)
    np.save('CV8/test/labels_st_test', stacked_labels)
    np.save('CV8/test/stacked_products_st_test', stacked_products)
    
    del stacked_x 

# Train dataset based on 3-fold Cross-validation
#for K in range(3):

stacker_train = StackFilesInDir(f'data8/{processor}/train')
stacked_x, stacked_y, stacked_labels, stacked_products = stacker_train.stackFiles(dateRange_train)

np.save(f'CV8/{processor}/X_train_{processor}', stacked_x)
np.save(f'CV8/{processor}/y_train_{processor}', stacked_y)
np.save(f'CV8/{processor}/labels_st_train_{processor}', stacked_labels)
np.save(f'CV8/{processor}/stacked_products_st_train_{processor}', stacked_products)    
del stacked_x

# Val dataset based on 3-fold Cross-validation
#for K in range(3):

stacker_val = StackFilesInDir(f'data8/{processor}/val')
stacked_x, stacked_y, stacked_labels, stacked_products = stacker_val.stackFiles(dateRange_val)

np.save(f'CV8/{processor}/X_val_{processor}', stacked_x)
np.save(f'CV8/{processor}/y_val_{processor}', stacked_y)
np.save(f'CV8/{processor}/labels_st_val_{processor}', stacked_labels)
np.save(f'CV8/{processor}/stacked_products_st_val_{processor}', stacked_products) 
del stacked_x