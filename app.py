import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

start = '2013-06-10'
end = '2023-06-10'

st.title('Stock Price Prediction')

user_input = st.text_input('Enter Stock Ticker ','AAPL')
try:
  df = yf.download(user_input,start, end)
except:
  st.subheader('Invalid Stock Ticker')
  sys.exit("Error message")
ValueError = df.describe().count()[3]-1
print(ValueError)
if ValueError==0:
  st.subheader('Invalid Stock Ticker')
  sys.exit("Error message")
  
st.subheader('Data from 2013 - 2023')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close,'b',label='Closing Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r',label='Moving Average 100')
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(df.Close,'b',label= 'Closing Price' )
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r',label ='Moving Average 100')
plt.plot(ma200,'g',label = 'Moving Average 200')
plt.xlabel('Time')
plt.ylabel('Price')

plt.plot(df.Close,'b',label='Closing Price')
plt.legend()
st.pyplot(fig)


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing  =  pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])



scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

model = load_model('ml_model.h5')

past_100_days = data_training.tail(100)
final_df = pd.DataFrame()
final_df = pd.concat([final_df, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100,input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted *=scale_factor
y_test *=scale_factor

st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_predicted,'r',label= 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

