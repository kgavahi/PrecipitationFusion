import os
import sys
import numpy as np
import hydroeval as he
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from keras import backend as Kb

def MSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)
def KGE(y_true, y_pred):
    kge, r, alpha, beta = he.evaluator(he.kge, y_pred, y_true)
    return kge    
def PCC(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0][1]
def tensorPCC(y_true, y_pred):
    x = y_true
    y = y_pred
    #x = Kb.constant(y_true)
    #y = Kb.constant(y_pred)
    mx = Kb.mean(x)
    my = Kb.mean(y)
    xm, ym = x-mx, y-my
    r_num = Kb.sum(tf.multiply(xm,ym))
    r_den = Kb.sqrt(tf.multiply(Kb.sum(Kb.square(xm)), Kb.sum(Kb.square(ym))))
    r = r_num / r_den

    return r  
def tensorKGE(y_true, y_pred):
    #y_true = Kb.constant(y_true)
    #y_pred = Kb.constant(y_pred)
    
    #y_true = y_true * y_train_std + y_train_mean
    #y_pred = y_pred * y_train_std + y_train_mean
    
    r = tensorPCC(y_true, y_pred)
    
    alpha = Kb.mean(y_pred) / Kb.mean(y_true)
    
    beta = Kb.std(y_pred) / Kb.std(y_true)
    
    return 1 - Kb.sqrt(Kb.square(r - 1) + Kb.square(alpha - 1) + Kb.square(beta - 1))
    
def create_model():

	data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, w_s*2+1, w_s*2+1, w_i)), input_shape=(w_t*2+1, w_s*2+1, w_s*2+1, w_i)),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, w_t*2+1, w_s*2+1, w_s*2+1, w_i))),
	]) 


    model = tf.keras.models.Sequential([
            data_augmentation,
            tf.keras.layers.Conv3D(7, (1,3,3),padding='same',data_format='channels_last', activation='relu'),
            tf.keras.layers.Conv3D(3, (1,1,1),padding='same',data_format='channels_last', activation='relu'),
            tf.keras.layers.Conv3D(16, (3,3,3),padding='same',data_format='channels_last', activation='relu'),
            tf.keras.layers.MaxPooling3D((1, 2, 2),strides=(1, 2, 2),data_format='channels_last'),
            tf.keras.layers.Conv3D(32, (3,3,3),padding='same',data_format='channels_last', activation='relu'),
            tf.keras.layers.MaxPooling3D((1, 2, 2),strides=(1, 2, 2),data_format='channels_last'),
            tf.keras.layers.Conv3D(64, (3,3,3),padding='same',data_format='channels_last', activation='relu'),
            #tf.keras.layers.MaxPooling3D((1, 2, 2),strides=(1, 2, 2),data_format='channels_last'),
            tf.keras.layers.Conv3D(128, (3,3,3),padding='same',data_format='channels_last', activation='relu'),
            #tf.keras.layers.MaxPooling3D((1, 2, 2),strides=(1, 2, 2),data_format='channels_last'),
            tf.keras.layers.ConvLSTM2D(filters=10, kernel_size=(3, 3),
                padding='same', return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.MaxPooling3D((1, 2, 2),strides=(1, 2, 2),data_format='channels_last'),
            tf.keras.layers.ConvLSTM2D(filters=10, kernel_size=(3, 3),
                padding='same', return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling3D((1, 2, 2),strides=(1, 2, 2),data_format='channels_last'),		
                
            tf.keras.layers.ConvLSTM2D(filters=10, kernel_size=(3, 3),
                #input_shape=(24, 70, 160, 9),
                padding='same', return_sequences=True),
            tf.keras.layers.ConvLSTM2D(filters=5, kernel_size=(3, 3),
                #input_shape=(24, 70, 160, 9),
                padding='same', return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ConvLSTM2D(filters=5, kernel_size=(3, 3),
                #input_shape=(24, 70, 160, 9),
                padding='same', return_sequences=True),
            tf.keras.layers.BatchNormalization(),		
            tf.keras.layers.ConvLSTM2D(filters=5, kernel_size=(3, 3),
                #input_shape=(24, 70, 160, 9),
                padding='same', return_sequences=True),
            tf.keras.layers.BatchNormalization(),
            #tf.keras.layers.BatchNormalization(),	
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='linear'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='linear')
        ])
    
	model.compile(optimizer="adam", loss='mean_squared_error', metrics=[tensorPCC])
    
	return model



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
job_id = sys.argv[3]

path = f'preprocess/scenarios/{scenario}/{fold}'

products_list = sen_dict[scenario]['products']
w_t = sen_dict[scenario]['w_t'] # n days before, n days after
w_s = sen_dict[scenario]['w_s'] # n cells from each side
w_i = len(products_list)


print(f'scenario: {scenario}, fold: {fold}, w_s={w_s}, w_t={w_t}')


X_train = np.load(f'{path}/X_train.npy')
X_test = np.load(f'{path}/X_test.npy')
X_val = np.load(f'{path}/X_val.npy')
y_train = np.load(f'{path}/y_train.npy')
y_test = np.load(f'{path}/y_test.npy')
y_val = np.load(f'{path}/y_val.npy')




model = create_model()
print(model.summary())

# shuffle the training dataset
X_train, y_train = shuffle(X_train, y_train, random_state=1)


# Normalize the datasets
X_train_mean = np.mean(X_train, axis=(0, 1, 2, 3))
X_train_std = np.std(X_train, axis=(0, 1, 2, 3))
X_train = (X_train - X_train_mean) / X_train_std
X_val = (X_val - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std


y_train_mean = np.mean(y_train)
y_train_std = np.std(y_train)
y_train = (y_train - y_train_mean) / y_train_std
y_val = (y_val - y_train_mean) / y_train_std
y_test = (y_test - y_train_mean) / y_train_std

# Save the model weights at each epoch
PATH_TO_SAVE = f'scenarios/{scenario}/{fold}/{job_id}'
if not os.path.exists(PATH_TO_SAVE): os.makedirs(PATH_TO_SAVE)
checkpoint_path = '%s/cp-{epoch:04d}.ckpt'%PATH_TO_SAVE
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)
# EarlyStopping to avoid overfitting                                               
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=100,
                                                    mode='min')
EPOCHS = 50
history = model.fit(
    X_train,
    y_train,
    batch_size=4096,
    epochs=EPOCHS,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_val, y_val),
    callbacks=[cp_callback, early_stopping],
)
np.save(f'{PATH_TO_SAVE}/history',history.history)



pcc_test = []
mse_test = []
kge_test = []

for i in range(1,EPOCHS+1):
    # Delete the model to save memory for prediction on the test set
    del model
    model = create_model()
    model.load_weights(f'{PATH_TO_SAVE}/cp-%04d.ckpt'%i)
    
    y_pred = model.predict(X_test).ravel()
    pcc_test.append(PCC(y_test, y_pred))
    mse_test.append(MSE(y_test, y_pred))
    kge_test.append(KGE(y_test, y_pred)[0])

