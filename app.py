from flask import Flask, render_template, request
import quandl
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from tensorflow.keras import backend
import numpy as np
from keras.layers import Dense, Dropout, LSTM
import pandas as pd
from keras.models import load_model
import tensorflow as tf
import os
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

app=Flask(__name__)

IMAGE_FOLDER = os.path.join('static', 'images')
app.config['IMAGE_FOLDER'] = IMAGE_FOLDER

quandl.ApiConfig.api_key = "4XHNm5Mmx7wuvUbMXfct"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/predict_page')
def page():
    return render_template('page.html')

@app.route('/graph', methods=['POST'])
def graph():
    p=request.form['company']
    df = quandl.get("WIKI/MSFT")
    #print(data.head())
    print(df.columns.values)
    #creating dataframe
    data = df.sort_index(ascending=True, axis=0)
    data['Date'] = df.index
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Value'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Value'][i] = data['Close'][i]

    #setting index
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)
    print(len(new_data))
    #creating train and test sets
    dataset = new_data.values
    length=int(len(data)*0.80)

    train = dataset[0:length,:]
    valid = dataset[length:,:]

    #converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(60,len(train)):
        x_train.append(scaled_data[i-60:i,0])
        y_train.append(scaled_data[i,0])
    x_train, y_train = np.array(x_train), np.array(y_train)

    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    # create and fit the LSTM network
    # model = Sequential()
    # model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
    # model.add(LSTM(units=50))
    # model.add(Dense(1))

    # model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
    #model = load_model('StonksModel.h5')
    #predicting 246 values, using past 60 from the train data
    loaded_model = tf.keras.models.load_model('StonksModel.h5')
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)
    #print(len(inputs))

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i-60:i,0])
    X_test = np.array(X_test)
    print(len(X_test))

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
    result=loaded_model.predict(X_test)
    closing_price = scaler.inverse_transform(result)
    print(len(closing_price))
    # closing_price = model.predict(X_test)
    train = new_data[:-len(X_test)]
    valid = new_data[-len(X_test):]
    valid['Predictions'] = closing_price
    print(valid.tail())
    print(train.tail())
    plt.plot(train['Value'])
    plt.plot(valid[['Value','Predictions']])
    #os.path.abspath('.')
    #os.path.join('/home/k_pritam/project/myApp', '/static/images')
    plt.savefig('static/new_plot.png')
    return render_template('graph.html', name = 'GDP', url ='new_plot1.png')

if __name__ == "__main__":
    app.run(debug=True, threaded=False)