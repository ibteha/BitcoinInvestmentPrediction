import pandas as pd 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Bidirectional, Activation, LeakyReLU
import datetime
import schedule
import time
# this is use for RSI indicator

def RSI4 (stock):    
    Hist_data = pd.read_csv(stock)
    Company = ((os.path.basename(stock)).split(".csv")[0])  # Name of the company
    # This list holds the closing prices of a stock
    prices = []
    c = 0
    # Add the closing prices to the prices list and make sure we start at greater than 2 dollars to reduce outlier calculations.
    while c < len(Hist_data):
        if Hist_data.iloc[c,4] > float(2.00):  # Check that the closing price for this day is greater than $2.00
            prices.append(Hist_data.iloc[c,4])
        c += 1
    # prices_df = pd.DataFrame(prices)  # Make a dataframe from the prices list
    i = 0
    #print(prices)
    upPrices=[]
    downPrices=[]
    #  Loop to hold up and down price movements
    while i < len(prices):
        if i == 0:
            upPrices.append(0)
            downPrices.append(0)
        else:
            if (prices[i]-prices[i-1])>0:
                upPrices.append(prices[i]-prices[i-1])
                downPrices.append(0)
            else:
                downPrices.append(prices[i]-prices[i-1])
                upPrices.append(0)
        i += 1
    x = 0
    avg_gain = []
    avg_loss = []
    #  Loop to calculate the average gain and loss
    while x < len(upPrices):
        if x <15:
            avg_gain.append(0)
            avg_loss.append(0)
        else:
            sumGain = 0
            sumLoss = 0
            y = x-14
            while y<=x:
                sumGain += upPrices[y]
                sumLoss += downPrices[y]
                y += 1
            avg_gain.append(sumGain/14)
            avg_loss.append(abs(sumLoss/14))
        x += 1
    p = 0
    RS = []
    RSI = []
    #  Loop to calculate RSI and RS
    while p < len(prices):
        if p <15:
            RS.append(0)
            RSI.append(0)
        else:
            RSvalue = (avg_gain[p]/avg_loss[p])
            RS.append(RSvalue)
            RSI.append(100 - (100/(1+RSvalue)))
        p+=1

    R=RSI[-1]
    tt=""
    if R <= 20:
        tt="Sell"
    elif R >= 80:
        tt="Buy"
    else:
        tt="Neutral"
    return R, tt
# this is use for ICC indicator
def CCI(stock, ndays):
    data = pd.read_csv(stock)
    TP = (data['high'] + data['low'] + data['close']) / 3 
    CCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()),
                    name = 'CCI') 
    data = data.join(CCI)
    c=0
    cci=[]
    while c < len(data):    
        cci.append(data.iloc[c,5])
        c+=1
    R=cci[-1]
    tt=""
    if R <= -100:
        tt="Sell"
    elif R >= 100:
        tt="Buy"
    else:
        tt="Neutral"
    return R, tt


# this is use for SOs
def SOc(stock, ndays):
    df = pd.read_csv(stock)
    df['L14'] = df['low'].rolling(window=14).min()
#Create the "H14" column in the DataFrame
    df['H14'] = df['high'].rolling(window=14).max()
#Create the "%K" column in the DataFrame
    df['%K'] = 100*((df['close'] - df['L14']) / (df['H14'] - df['L14']) )
#Create the "%D" column in the DataFrame
    df['%D'] = df['%K'].rolling(window=3).mean()
    k=[]
    d=[]
    c=0
    while c < len(df):
        k.append(df.iloc[c,7])
        d.append(df.iloc[c,8])
        c+=1
    R=k[-1]
    S=d[-1]
    tt=""
    ss=""
    if R <= 20:
        tt="Sell"
    elif R >= 80:
        tt="Buy"
    else:
        tt="Neutral"
    if (R < S)  & (S > 80):
        ss="Sell"
    elif (R > S ) & (S < 20):
        ss="Buy"
    else:
        ss="Neutral" 
    return R, tt,ss
      

# this is use for EMO
def EMO(stock):

    status=[]
    Hist_data = pd.read_csv(stock)
    Company = ((os.path.basename(stock)).split(".csv")[0])  # Name of the company
    # This list holds the closing prices of a stock
    prices = []
    c = 0
    while c < len(Hist_data):
        if Hist_data.iloc[c,4] > float(2.00):  # Check that the closing price for this day is greater than $2.00
            prices.append(Hist_data.iloc[c,4])
        c += 1
    list2=[]
    ema=[]
    p= 0
    s=0
    k=0
    f=4
    m=0
    i = 0
    ins=[]
#     print(prices)
# calculate SMO
    while p < len(prices)+1:
        if p>=f:
            s=sum(prices[i:p])
            t=s/f
            t = round(t)
            list2.append(t)
            i+=1
        p+=1
    wm=(2/(f+1))  
    while k < len(prices):
        if len(ema)==0:
            emma=((wm*(prices[k]-list2[k]))+list2[k])
            ema.append(emma)  
        else:
            em=((wm*(prices[k]-ema[k-1]))+ema[k-1])
            ema.append(em)
        if k >0:
            ll=ema[k-1]-prices[k]
            # ins.append(ll)
            if ll > 15:
                kk= "Buy"
                status.append(kk)
            elif ll < -15:
                kk= "Sell"
                status.append(kk)
            else:
                kk= "Neutral"
                status.append(kk)
            jj=list2[-1]-prices[-1]
            if jj > 15:
                mm= "Buy"
            elif jj < -15:
                mm= "Sell"
            else:
                mm= "Neutral"
                
        else:
            status.append(0)
            ins.append(0)
            mm="NULL"
        R=ema[-1]
        ss=status[-1]   
        k+=1
    pr=prices[-1]
    return R,ss,pr,jj,mm

def model_predic(stock):
    model = keras.models.load_model("test1.h5")
    def split_sequence(seq, n_steps_in, n_steps_out):
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
    def input_create(dat,s):

        dat=dat.reshape(-1)
        if s==1:
            train, test = dat[:-25], dat[-25:]
            return train, test
        else:
            train, test = dat[:-30], dat[-30:]
            return train, test
    def data_set(s):
        n_steps_in = 25
        n_steps_out = 5
        df = pd.read_csv(stock)
        df.dropna(inplace=True)
        df.drop_duplicates(subset=['date'], keep='last', inplace=True)
        data = np.array(df['close'].values)
        data = data.reshape(-1,1)
        scaler = MinMaxScaler(feature_range=(0,1))

        rescale = scaler.fit_transform(data)
        train,test = input_create(rescale,s)
    #     print(len(test))
        #train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
        if len(test) == 25:
            test_x_1=np.array([test])
            test_x_1 = test_x_1.reshape(test_x_1.shape[0], test_x_1.shape[1], 1)
            return scaler, test_x_1
        else:
            test_x, test_y = split_sequence(list(test), n_steps_in, n_steps_out)
            test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)
            return scaler, test_x,test_y
    def prediction():
        s=0
        current_time = datetime.datetime.now()
        # Getting predictions by predicting from the last available X variable
        #yhat = model.predict(X[-1].reshape(1, n_per_in, n_features)).tolist()[0]
        # Transforming values back to their normal prices
        if s==0:
            scaler, test_x_1, test_y = data_set(s)
            predi = model.predict(test_x_1)
            yhat = scaler.inverse_transform(np.array(predi).reshape(-1,1)).tolist()
            yhat = [np.round(x,2) for x in yhat]
            # Getting the actual values from the last available y variable which correspond to its respective X variable
            actual = scaler.inverse_transform(test_y[-1].reshape(-1,1)).tolist()
            actual = [np.round(x,2) for x in actual]
            rms = mean_squared_error(actual, yhat, squared=False)
        s+=1    
        scaler, test_x= data_set(1)
        predi_1 = model.predict(test_x)
        yhaat = scaler.inverse_transform(np.array(predi_1).reshape(-1,1)).tolist()
        yhaat = [np.round(x,2) for x in yhaat]
        R=yhaat[0][0]
        t=yhaat[0]-actual[-1]
        t=t+rms
        if t > 50:
            tt="Buy"
        elif t < -50:
            tt="Sell"
        else:
            tt="Neutral"
        return R,tt
    R,tt=prediction()
    return R,tt
