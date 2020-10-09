import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import LSTM,BatchNormalization,Bidirectional,ConvLSTM2D
from keras.layers import Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from collections import deque


prices_dataset_train=pd.read_csv('D:\\RNN_trend_analysis-master\\trainset.csv')
prices_dataset_test=pd.read_csv('D:\\RNN_trend_analysis-master\\testset.csv')

training_set=prices_dataset_train.iloc[:,1:2].values
test_set=prices_dataset_test.iloc[:,1:2].values

minmaxscaler=MinMaxScaler(feature_range=(0,1))
scaled_training_set=minmaxscaler.fit_transform(training_set)

#%%

X_train=[]
Y_train=[]

for i in range(60,262):
    X_train.append(scaled_training_set[i-60:i,0])
    Y_train.append(scaled_training_set[i,0])

X_train=np.array(X_train)
Y_train=np.array(Y_train)

X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

model=Sequential()
model.add(Bidirectional(LSTM(units=100,return_sequences=True,input_shape=(X_train.shape[1],1))))
model.add(Dropout(0.5))
# model.add(BatchNormalization())
model.add(Bidirectional(LSTM(units=50,return_sequences=True)))
model.add(Dropout(0.3))
# model.add(BatchNormalization())
model.add(Bidirectional(LSTM(units=50,return_sequences=True)))
model.add(Dropout(0.3))
# model.add(BatchNormalization())
model.add(Bidirectional(LSTM(units=50)))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy','mean_squared_error'])

model.fit(X_train,Y_train,epochs=100,batch_size=16)

dataset_total=pd.concat((prices_dataset_train['Price'],prices_dataset_test['Price']),axis=0)

inputs=dataset_total[len(dataset_total)-len(prices_dataset_test)-60:].values

inputs=inputs.reshape(-1,1)
inputs=minmaxscaler.transform(inputs)

#%%
X_test=[]

for i in range(60,len(prices_dataset_test)+60):
    X_test.append(inputs[i-60:i,0])

X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predictions=model.predict(X_test)
predictions=minmaxscaler.inverse_transform(predictions)

plt.plot(test_set,color='blue',label='Actual Prices')
plt.plot(predictions,color='red',label='Predicted Prices')
plt.title('Trend Analysis')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

#%%
def future_prediction(last_60_val):
    predict_next = model.predict(last_60_val)
    return predict_next
    
prev_days = deque(maxlen=60)
future_60 = []

last_inputs = X_test[len(X_test)-1]
last_inputs = np.reshape(last_inputs, (1,len(last_inputs),len(last_inputs[0])))

for i in last_inputs[0]:
    # print(i," next")
    prev_days.append(i)

arr = np.array(prev_days)
# print(arr)
arr = np.reshape(arr, (1,len(arr),len(arr[0])))
#%%
for k in range(1,60):
    # print(arr[0][0])
    next_price = future_prediction(arr)
    
    prev_days.append(next_price)
    arr = np.asfarray(prev_days)
    # print(arr)
    arr = np.reshape(arr, (1,len(arr),len(arr[0])))
    
    next_price=minmaxscaler.inverse_transform(next_price)
    future_60.append(next_price[0])

#%%
predictions = np.concatenate((predictions, future_60), axis=0)
plt.plot(test_set,color='blue',label='Actual Prices')
plt.plot(predictions,color='red',label='Predicted Prices')
plt.title('Trend Analysis')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()





