from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline #for google colaboratory drawing

import pandas as pd
#from linearregression import scaler

model = load_model("Covid-19")

df = pd.read_csv('LastCovid.csv')
data = df.filter(['Count'])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

#print('type',type(data))
predict = []
for i in range(10):
  last_days = data[-10:].values
  #Scale the data to be values between 0 and 1
  last_days_scaled = scaler.fit_transform(last_days)
  #Create an empty list
  X_test = []
  #Append teh past 60 days
  X_test.append(last_days_scaled)
  #Convert the X_test data set to a numpy array
  X_test = np.array(X_test)
  #Reshape the data
  X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
  #Get the predicted scaled price
  pred_count = model.predict(X_test)
  #undo the scaling 
  pred_count = scaler.inverse_transform(pred_count)
  #print(pred_price)
  d = pd.DataFrame({ 'Count': pred_count[0] })
  #print(d)
  predict.append(pred_count)
  frames = [data, d]
  
  results = pd.concat(frames)
  data = results

temp = []
for i in data['Count']:
  #print(i[0])
  temp.append(i)
print(temp)

#print(temp)
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Covid-19', fontsize=18)

#plt.plot(Tempdata)
plt.plot(temp)

#plt.plot(valid[['first', 'second']])
plt.legend(['Val'], loc='upper right')
plt.show()
