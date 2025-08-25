import yfinance as yf
import pandas as pd
import os
def inputh():
    print("Amigo its doing the predection of the META and you might make some money \n")
    return "META"                                   
def inputt():
    print("holo its now inputing the the frame")
    return "2010-01-01"
stock =yf.Ticker("^GSPC")
stock =stock.history(period = "max")
stock.to_csv("pro_log.csv")
stock.index = pd.to_datetime(stock.index)
stock.plot.line(y= "Close" , use_index = True)      
del stock["Dividends"]
del stock["Stock Splits"]
stock["Next_Day"] = stock["Close"].shift(-1)
stock["Target"] = (stock["Next_Day"] > stock["Close"]).astype(int)
stock = stock.loc[inputt():].copy()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score  
Mudel = RandomForestClassifier(n_estimators=200, min_samples_split =50,random_state = 1)
train = stock.iloc[:-169]
test  = stock.iloc[-169:]
pre_var= ["Close" , "Open" ,"Volume" ,"High", "Low"]
def pre_fun(train,test ,pre_var,Mudel):
    Mudel.fit(train[pre_var],train["Target"])
    pee = Mudel.predict_proba(test[pre_var]) [:,1]
    pee[pee >=.6] = 1 
    pee[pee < .6] = 0
    pee = pd.Series(pee,index=test.index,name = "PRE")
    com = pd.concat([test["Target"] ,pee] ,axis = 1)
    return com
def back_fun(data, Model ,pre, start_at = 2500,stop = 250):
    pre_all = []
    for i in range(start_at,data.shape[0],stop):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+stop)].copy()
        temp_pre = pre_fun(train,test,pre,Model)
        pre_all.append(temp_pre)
    return pd.concat(pre_all)

inte = [2,5,10,15,30,60,90,150,190,240,280,30,400,1000]
ninu =  []
for i in inte:
    tem_avg = stock.rolling(i).mean()
    motucol = f"Avg_close {i}"
    stock[motucol] = stock["Close"] / tem_avg["Close"]

    tre_col = f"Tre{i}"
    stock[tre_col] = stock.shift(1).rolling(i).sum()["Target"]
    ninu.extend([motucol, tre_col])

stock = stock.dropna(subset = stock.columns[stock.columns != "Next_Day"])  
pre_dick = back_fun(stock,Mudel,ninu)
print(pre_dick["Target"].value_counts()/pre_dick.shape[0])
print("this is the score", precision_score(pre_dick["Target"] ,pre_dick["PRE"]))




    
    
    

                                                                                                        
