import pandas as pd
from multiFunction import RSI4,CCI,SOc,EMO,model_predic
from datetime import datetime
import time, schedule
now = datetime.now()
import time
import sys
import argparse
import csv
def main_1():
    file="final_data.csv"
    a,f,=RSI4(file)
    b,g=CCI(file,20)
    c,h,i=SOc(file,14)
    d,j,l=EMO(file)
    e,k=model_predic(file)
    return  a,b,c,c,d,e,f,g,h,i,j,k,l
def abc(c):
    l=c.count('Buy')
    m=c.count('Sell')
    if l > 4 :
        tt='Strong Buy'
    elif l > m and l <= 4 and l>1 :
        tt='Buy' 
    elif m > 4:
        tt='Strong Sell'
    elif m > l and m <= 4 and m>1:
        tt='Sell'
    else:
        tt='Neutral'
    return tt
def last_btc():
    df = pd.read_csv("final_data.csv")
    last = df['close'].tail(1).values
    return last
def result(value,btc,btc_1,v):
    a=0.0
    y=0.0
    prc=abc(v)
    if prc == "Strong Buy":
        y=0.5*value
        a = y/btc_1
        value=value-y
        print("BTC_Value==>",a)
        btc= a+btc
    elif prc == "Buy":
        y=0.3*value
        a = y/btc_1
        value=value-y
        print("BTC_Value==>",a)
        btc= a+btc
    elif prc == "Strong Sell":
        if btc == 0.0:
            print("you have no btc")
            a=a
            y=y
            btc=btc
            value=value
        else:
            a = 0.4*btc
            btc= btc-a
            y=btc_1*a
            value=value+y
            a=-a
    elif prc == "Sell":
        if btc == 0.0:
            a=a
            y=y
            btc=btc
            value=value
        else:
            a = 0.3*btc
            btc= btc-a
            y=btc_1*a
            value=value+y
            a=-a
        print("btc",btc)
    else:
        print("Our prediction is Neutral")
        a=a
        y=y
        btc=btc
        value=value
        
    cu = now.strftime("%H:%M:%S")
    return prc,btc,y,a,value,cu
# ki rola ja
def write_csv(User_Name,A_Amount,btc):
    b = []
    g= []
    main_1()
    g.append(main_1())
    b.append(g[0][-7:-1])
    v=b[0]
    btc_1=g[0][-1]
    c=btc_1
    last=c
    cu,inv_am,inv_btc,n_btc,n_amount,predict= now.strftime("%H:%M:%S"),0.0,0.0,0.0,0.0,"Neutral",
    df_a={"Current Time":cu,"Actual Ammount":A_Amount,"Actual_BTC":btc,"O_BTC_DV":c,"Invest_Amount":inv_am,"Invest_BTc":inv_btc,"New_BTC":n_btc,"New_Amount":n_amount,"Prediction":predict,"New_BTC_DV":last}
    df=pd.DataFrame(df_a,index=[0])
    df.to_csv(User_Name+"_1.csv", index = False)
    while (True):
        if btc==0.0 and A_Amount==0.0:
            break
        predict,n_btc, inv_am, inv_btc, n_amount,cu = result(A_Amount,btc,c,v)  
#         df_b = pd.read_csv("User_Name+"_1.csv")
        v=User_Name +"_1.csv"
        with open(v, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([cu,A_Amount,btc,c,inv_am,inv_btc,n_btc,n_amount,predict,last])
        o_btc,o_amount=btc,A_Amount
        btc,A_Amount=n_btc,n_amount
        time.sleep(80)
        last = last_btc()
        last=last[0]
        c=last
    print("You have no amount of dollar and btc, Invest more ammount")

if __name__ == "__main__":
    a,b,c="None",0.0,0.0
    if len(sys.argv) == 4:
        a = str(sys.argv[1])
        b = float(sys.argv[2])
        c = float(sys.argv[3])
    write_csv(a,b,c)
