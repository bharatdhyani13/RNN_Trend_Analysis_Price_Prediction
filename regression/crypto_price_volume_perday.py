import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
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
api_key = 'UCZNJAZ4IHTWHCKL'

# prices_dataset_train=pd.read_csv('D:\\RNN_trend_analysis-master\\trainset.csv')
# prices_dataset_test=pd.read_csv('D:\\RNN_trend_analysis-master\\testset.csv')
main_df = pd.DataFrame()
trend = "BTC" #, "ETH", "LTC"
df = get_daily_data(api_key, trend)  # read in specific file

# rename volume and close to include the ticker so we can still which close/volume is which:
df.rename(columns={"4a. close (USD)": f"{trend}_close", "5. volume": f"{trend}_volume"}, inplace=True)

df = df[[f"{trend}_close", f"{trend}_volume"]]  # ignore the other columns besides price and volume

if len(main_df)==0:  # if the dataframe is empty
    main_df = df  # then it's just the current df
else:  # otherwise, join this data to the main one
    main_df = main_df.join(df)
print(main_df.head())

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

training_set=main_df.values
test_set=validation_main_df.values



minmaxscaler=MinMaxScaler(feature_range=(0,1))
minmaxscaler1=MinMaxScaler(feature_range=(0,1))
scaled_training_set0=minmaxscaler.fit_transform(training_set[:,0].reshape(-1, 1))
scaled_training_set1=minmaxscaler1.fit_transform(training_set[:,1].reshape(-1, 1))
scaled_training_set=np.concatenate((scaled_training_set0,scaled_training_set1),axis=1)


X_train=[]
Y_train=[]

for i in range(60,len(scaled_training_set)):
    X_train.append(scaled_training_set[i-60:i,:])
    Y_train.append(scaled_training_set[i,0])

X_train=np.array(X_train)
Y_train=np.array(Y_train)

# X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

model=Sequential()
model.add(Bidirectional(LSTM(units=128,return_sequences=True,input_shape=(Y_train.shape[1:]))))
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

model.fit(X_train,Y_train,epochs=100,batch_size=16)

dataset_total=df.copy()

inputs=dataset_total[len(dataset_total)-len(validation_main_df)-60:]
inputs=inputs.values
scaled_test_set0=minmaxscaler.fit_transform(inputs[:,0].reshape(-1, 1))
scaled_test_set1=minmaxscaler1.fit_transform(inputs[:,1].reshape(-1, 1))
inputs=np.concatenate((scaled_test_set0,scaled_test_set1),axis=1)
# inputs=minmaxscaler.transform(inputs)

X_test=[]

for i in range(60,len(validation_main_df)+60):
    X_test.append(inputs[i-60:i,:])

X_test=np.array(X_test)


predictions=model.predict(X_test)
predictions=minmaxscaler.inverse_transform(predictions)

plt.plot(test_set[:,0],color='blue',label='Actual Prices')
plt.plot(predictions,color='red',label='Predicted Prices')
plt.title('Trend Analysis')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
    




###Next steps:
    # take last 40 value and predict next day price.. If pred > pred -1 . 
    # But with this we only get to know if the next day price is going to be higher or lower
    # than previous day.
    # find a way to predict the trend in the future. for the next 40 or 60 days.



    





