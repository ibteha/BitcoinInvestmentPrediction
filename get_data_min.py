from binance.client import Client
import pandas as pd
import time
import schedule 
import calendar 

api_key = "ygBYNib4GxVPqCDcjxOwXulDAQ5XPB74qXNFhXuYej4RGSkJFPdiETxki5ELTyVg"
api_secret = "PkUBgXzp39N013qqFXMaKXRFDSGxpvEA5rtQ1SkZm9S3c7WtBkndeUZzN71pjWK4bars"

def data_get_daily():
    print("---------- working ------------")
    client = Client(api_key, api_secret)
    df = pd.read_csv("final_data.csv")
    df.drop_duplicates(subset=['date'], keep='last', inplace=True)
    start_timestamp = df['date'].tail(1).values
    current_timestamp = calendar.timegm(time.gmtime())*1000
    diffe = current_timestamp - start_timestamp
    time_in_mins =  diffe/(1000*60)
    time_in_mins= round(time_in_mins[0])
    bars = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, str(time_in_mins)+' minutes ago UTC')
    print(time_in_mins)
    for line in bars:
            del line[5:]

    # option 4 - create a Pandas DataFrame and export to CSV
    btc_df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close'])
    #btc_df.set_index('date', inplace=True)
    #btc_df.tail()
    df2 = df.append(btc_df)
    try:
       df2.reset_index(inplace=True)
    except:
        pass
    df2.drop(['index'], axis = 1, inplace=True)
    df2.set_index('date' , inplace=True)
    df2.to_csv("final_data.csv")
if __name__ == '__main__':
    try:
        schedule.every().minute.do(data_get_daily)
    except:
        try:
             data_get_daily()
        except:
            print("waiting One minute")
            pass
    while True:
        schedule.run_pending()
        time.sleep(1)
