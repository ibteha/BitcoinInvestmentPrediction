from binance.client import Client
import pandas as pd
api_keys = "ygBYNib4GxVPqCDcjxOwXulDAQ5XPB74qXNFhXuYej4RGSkJFPdiETxki5ELTyVg"
api_secrets = "PkUBgXzp39N013qqFXMaKXRFDSGxpvEA5rtQ1SkZm9S3c7WtBkndeUZzN71pjWK4bars"

def save_full_data(api_key, api_secret ):
    client = Client(api_key, api_secret)

    timestamp = client._get_earliest_valid_timestamp('BTCUSDT', '1m')
    bars = client.get_historical_klines('BTCUSDT', '1m', timestamp)
    for line in bars:
        del line[5:]
    
    # option 4 - create a Pandas DataFrame and export to CSV
    btc_df = pd.DataFrame(bars, columns=['date', 'open', 'high', 'low', 'close'])
    #btc_df.set_index('date', inplace=True)
    #btc_df.tail()
    btc_df.to_csv("data.csv")
    last = btc_df[-40000:].copy()
    last.to_csv("final_data.csv")

save_full_data(api_keys, api_secrets)    