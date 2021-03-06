# -*- coding: utf-8 -*-


import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#data reference https://www.statista.com/statistics/1105613/covid19-new-daily-cases-worldwide-by-region/
#only rows of the Date and Count in Asia are taken from the original .csv data section file to LastCovid.csv
df = pd.read_csv('LastCovid.csv')
df

df.shape

#Visualize the closing price history
#plt.figure(figsize=(16,8))
#plt.title('Close Price History')
#plt.plot(df['Count'])
#plt.xlabel('Date', fontsize=18)
#plt.ylabel('Covid_19', fontsize=18)
#plt.show()

#Create a new dataframe with only the 'Close column
data = df.filter(['Count'])
#Convert the dataframe to a numpy array
dataset = data.values
#Get the number of rows to train the model on
training_data_len = math.ceil( len(dataset) * 0.8 )

training_data_len

#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len , :]
#Split the data into x_train and y_train data sets
x_train = []
y_train = []
step = 10
for i in range(step, len(train_data)):
  x_train.append(train_data[i - step:i, 0])
  y_train.append(train_data[i, 0])
  if (i<= step + 1):
    print(x_train)
    print(y_train)
    print()

#Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)

#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
#model.add(Dense(25))
model.add(Dense(1))

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=100)
#Create the testing data set
#Create a new array containing scaled values from index 1543 to 2002 
steps = 10
test_data = scaled_data[training_data_len - steps: , :]
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(steps, len(test_data)):
  x_test.append(test_data[i-steps:i, 0])

#Convert the data to a numpy array
x_test = np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

#Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Get the root mean squared error (RMSE)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse

#Plot the data
#train = data[:training_data_len]
#valid = data[training_data_len:]
#valid['Predictions'] = predictions

#Visualize the data
#plt.figure(figsize=(16,8))
#plt.title('Model')
#plt.xlabel('Date', fontsize=18)
#plt.ylabel('Covid-19', fontsize=18)
#plt.plot(train['Count'])
#plt.plot(valid[['Count', 'Predictions']])
#plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
#plt.show()

#Save model
model.save('Covid-19')
