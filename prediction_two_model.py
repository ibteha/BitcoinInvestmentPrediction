# Library Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
plt.style.use("ggplot")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Bidirectional, Activation, LeakyReLU
import datetime
import schedule
import time
import os 

model = keras.models.load_model("test2.h5")
def split_sequence(seq, n_steps_in, n_steps_out):
    """
    Splits the univariate time sequence
    """
    X, y = [], []
    for i in range(len(seq)):
        end = i + n_steps_in
        out_end = end + n_steps_out
        if out_end > len(seq):
            break
        seq_x, seq_y = seq[i:end], seq[end:out_end]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)
def input_create(dat):

    
    dat=dat.reshape(-1)
    train, test = dat[:-30], dat[-30:]
    print(train.shape)
    return train, test

def data_set():
    print("start work ......")
    n_steps_in = 25
    n_steps_out = 5
    df = pd.read_csv('final_data.csv')
    df.dropna(inplace=True)
    data = np.array(df['close'].values)
    data = data.reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0,1))

    rescale = scaler.fit_transform(data)
    train , test = input_create(rescale)
    #train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
    test_x, test_y = split_sequence(list(test), n_steps_in, n_steps_out)
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)
    return scaler, test_x, test_y
def prediction():
    print("I'm working...")
    scaler, test_x, test_y = data_set()
    current_time = datetime.datetime.now()
    predi = model.predict(test_x)
    plt.figure(figsize=(12,5))
    # Getting predictions by predicting from the last available X variable
    #yhat = model.predict(X[-1].reshape(1, n_per_in, n_features)).tolist()[0]
    # Transforming values back to their normal prices
    yhat = scaler.inverse_transform(np.array(predi).reshape(-1,1)).tolist()
    # Getting the actual values from the last available y variable which correspond to its respective X variable
    actual = scaler.inverse_transform(test_y[-1].reshape(-1,1))
    # Printing and plotting those predictions
    #print("Predicted Prices:\n", yhat)
    plt.plot(yhat, label='Predicted')
    # Printing and plotting the actual values
    #print("\nActual Prices:\n", actual.tolist())
    plt.plot(actual.tolist(), label='Actual')
    plt.title(f"Predicted vs Actual Closing Prices")
    plt.ylabel("Price")
    plt.legend()
    #print()
    if not os.path.exists("model_two/"):
        os.mkdir("model_two/")

    plt.savefig('model_two/'+str(current_time)+"_test2.png")
if __name__ == '__main__':
 
    prediction()
    schedule.every(5).minutes.do(prediction)
    while True:
        schedule.run_pending()
        time.sleep(10)
