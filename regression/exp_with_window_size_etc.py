import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import time
# from utils import get_daily_data
import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, Bidirectional
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing

api_key = 'UCZNJAZ4IHTWHCKL'
SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "ETH"
EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"
minmaxscaler=preprocessing.MinMaxScaler(feature_range=(0,1))
#%%
def preprocess_df(df):
    scaled_training_set=minmaxscaler.fit(df)
    scaled_training_set=minmaxscaler.transform(df)
    X_train=[]
    Y_train=[]
    
    for i in range(40,len(df)):
        X_train.append(scaled_training_set[i-40:i,0])
        Y_train.append(scaled_training_set[i,0])
    
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    
    X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
    return X_train,Y_train
    

#%%
def get_daily_data(api_key, currency):
    
    resp = requests.get('https://www.alphavantage.co/query', params={
        'function': 'DIGITAL_CURRENCY_DAILY',
        'symbol': currency,
        'market': 'USD',
        'apikey': api_key
    })
    
    doc = resp.json()
    # print(doc)
    # print(doc['Time Series (Digital Currency Daily)'])
    df = pd.DataFrame.from_dict(doc['Time Series (Digital Currency Daily)'], orient='index', dtype=np.float)
    # print(df.head())

    df.drop(columns=[c for c in df.columns.values if 'b.' in c], inplace=True)
    # print(df.head())
    
    
    return df

#%%

# btc = get_daily_data(api_key, 'BTC')
# bch = get_daily_data(api_key, 'BCH')
# ltc = get_daily_data(api_key, 'LTC')
# eth = get_daily_data(api_key, 'ETH')

# df = pd.concat([btc['4a. close (USD)'], bch['4a. close (USD)'], ltc['4a. close (USD)'], eth['4a. close (USD)']], axis=1, join='inner')
# df.columns = ['BTC', 'ETH', 'LTC']
# print(df.head())

main_df = pd.DataFrame()
trend = "BTC" #, "ETH", "LTC"
df = get_daily_data(api_key, trend)  # read in specific file

# rename volume and close to include the ticker so we can still which close/volume is which:
df.rename(columns={"4a. close (USD)": f"{trend}_close"}, inplace=True)

df = df[[f"{trend}_close"]]  # ignore the other columns besides price and volume

if len(main_df)==0:  # if the dataframe is empty
    main_df = df  # then it's just the current df
else:  # otherwise, join this data to the main one
    main_df = main_df.join(df)
print(main_df.head())

#%%
main_df = main_df.iloc[::-1]
# main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
# main_df.dropna(inplace=True)

# main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
# main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

# main_df.dropna(inplace=True)

# # ## here, split away some slice of the future data from the main main_df.
times = sorted(main_df.index.values)
last_10pct = sorted(main_df.index.values)[-int(0.1*len(times))]

validation_main_df = main_df[(main_df.index >= last_10pct)]
main_df = main_df[(main_df.index < last_10pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

# # train_x = np.asarray(train_x)
# # train_y = np.asarray(train_y)
# # validation_x = np.asarray(validation_x)
# # validation_y = np.asarray(validation_y)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
# print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
# print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

#%%
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones


#%%
model=Sequential()
model.add(Bidirectional(LSTM(units=128,return_sequences=True,input_shape=(train_x.shape[1:]))))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(units=128,return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=128,return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=128,return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(units=128)))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(units=32))
model.add(Dropout(0.2))
model.add(Dense(units=1))

#normal one doesnt work well
#128 wala works well with old dropouts but variation is there. use the 32 dense first then use 1
#128 one with 0.2 dropouts does not works well.
#added an LSTM layer

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy','mean_squared_error'])

model.fit(train_x,train_y,epochs=100,batch_size=16,validation_data=(validation_x, validation_y),callbacks=[tensorboard])
# model.save("models/{}".format(NAME))
#%%

dataset_total=df.copy()

inputs=dataset_total[len(dataset_total)-len(validation_main_df)-40:].values

inputs=inputs.reshape(-1,1)
inputs=minmaxscaler.transform(inputs)

X_test=[]
Y_test=validation_main_df.values

for i in range(40,len(validation_main_df)+40):
    X_test.append(inputs[i-40:i,0])

X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predictions=model.predict(X_test)
predictions=minmaxscaler.inverse_transform(predictions)


#%%
plt.plot(Y_test,color='blue',label='Actual Prices')
plt.plot(predictions,color='red',label='Predicted Prices')
plt.title('Trend Analysis')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()