# %%
import pyodbc
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import precision_score
import pickle
from datetime import datetime
import warnings
conn = pyodbc.connect(
    "DRIVER={SQL Server};SERVER=lenovo-desktop;DATABASE=Qihuo;UID=samtsql;PWD=F(W}q:TsyK,7^+>`P28e79s#Uc5n")


def getData(symbol):
    query = f"""select top 2059 m1.date as 'Date',DATEPART(hour, m1.Date) AS 'Hour', m1.[Close] as 'Market', m2.[close] as 'Stock' from MinuteQuote m1
    inner join MinuteQuote m2 on m1.Date = m2.date
    where m1.Date > getdate() - 45 and m1.Contract = 'ym' and m2.Contract = '{symbol}'
    order by m1.date desc"""

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        df = pd.read_sql(query, conn)
        df = df.iloc[::-1]
        df.set_index('Date', inplace=True)
        return df


# %%
from enum import Enum
class ToTrade(Enum):
    OpenLong = 1
    OpenShort = 2
    Close = 3
    CloseLong = 4
    CloseShort = 5
    

# %%
def getCurrentPosition(account, symbol):
    query = f"""select * from ArbitrageMLTrade
                where AccountName = '{account}' and Symbol1 = '{symbol}' and Active = 1"""

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        df = pd.read_sql(query, conn)
        if df.empty:
            return 0
        else:
            return df['Position'][0]


# %%
def OpenTrade(account, symbol, marketPrice, stockPrice, position):
    share2 = 1.0
    if account > "xieie263":
        share2 = 0.1

    share1 = round((marketPrice * share2 * 5 / stockPrice) / 100, 0) * 100
    
    query = f"""INSERT INTO [dbo].[ArbitrageMLTrade]
           ([AccountName]
           ,[SecurityType1]
           ,[SecurityType2]
           ,[Symbol1]
           ,[Symbol2]
           ,[Position]
           ,[Share1]
           ,[Share2]
           ,[EnterPrice1]
           ,[EnterPrice2]
           ,[ExitPrice1]
           ,[ExitPrice2]
           ,[EnterTime]
           ,[ExitTime]
           ,[Active]
           ,[CreatedOn]
           ,[ModifiedOn])
     VALUES
           ('{account}'
           ,'Stock'
           ,'Future'
           ,'{symbol}'
           ,'YM'
           ,{position}
           ,{share1}
           ,{share2}
           ,{stockPrice}
           ,{marketPrice}
           ,null
           ,null
           ,getdate()
           ,null
           ,1
           ,getdate()
           ,getdate())"""

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()


# %%
def CloseTrade(account, symbol):
    query = f"""UPDATE [dbo].[ArbitrageMLTrade]
           SET Active = 0, ExitTime = GETDATE(), ModifiedOn = GETDATE() WHERE Symbol1 = '{symbol}' AND AccountName = '{account}' AND Active = 1"""

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()

# %%

def transformData(df):
    df[f'StockReturn'] = df["Stock"] / df["Stock"].shift(1) - 1
    df[f'MarketReturn'] = df["Market"] / df["Market"].shift(1) - 1
    df[f"OutPerform"] = df['StockReturn'] - df['MarketReturn']
    df["Target"] = (df.apply(lambda x: x > 0)["OutPerform"]).astype(int)


# %%
def getPredictors(df):
    predictors = ['Hour']
    for i in range(12):
        df[f'OutPerform{pow(2,i)*5}'] = df['OutPerform'].rolling(pow(2,i)).sum()
        predictors.append(f'OutPerform{pow(2,i)*5}')
        df[f'StockReturn{pow(2,i)*5}'] = df['StockReturn'].rolling(pow(2,i)).sum()
        predictors.append(f'StockReturn{pow(2,i)*5}')
        df[f'MarketReturn{pow(2,i)*5}'] = df['MarketReturn'].rolling(pow(2,i)).sum()
        predictors.append(f'MarketReturn{pow(2,i)*5}')
    return predictors


# %%
def getMLdata(df, predictors):
    df = df[(df.index.minute == 25) | (df.index.minute == 55) ]
    return df.iloc[-1:]

# %%
def predict(test, predictors, model):
    preds = model.predict_proba(test[predictors])[:, 1]
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# %%
def processResult(currentPosition, predictions):
    prediction = predictions[len(predictions) - 1]

    if currentPosition == 0:
        if prediction > .57:
            return ToTrade.OpenLong
        elif prediction < .43:
            return ToTrade.OpenShort

    if currentPosition == 1:
        if prediction < .43:
            return ToTrade.CloseShort
        elif prediction < .50:
            return ToTrade.Close

    if currentPosition == -1:
        if prediction > .57:
            return ToTrade.CloseLong
        elif prediction > .50:
            return ToTrade.Close


# %%

import sys
symbols = sys.argv[1:]  
accounts = ['xieie181']  
for s in symbols:
    df = getData(s)
    transformData(df)
    predictors = getPredictors(df)
    data = getMLdata(df, predictors)
    with open(f"Model/{s}", 'rb') as f:
        model = pickle.load(f)
        predictions = model.predict_proba(data[predictors])[:, 1]
        print(s, predictions)
        for account in accounts:
            currentPosition = getCurrentPosition(account, s)
            totrade = processResult(currentPosition, predictions)
            if totrade == ToTrade.OpenLong:
                OpenTrade(account, s, df['Market'][len(df) - 1], df['Stock'][len(df) - 1], 1)
            elif totrade == ToTrade.OpenShort:     
                OpenTrade(account, s, df['Market'][len(df) - 1], df['Stock'][len(df) - 1], -1)
            elif totrade == ToTrade.Close:
                CloseTrade(account, s)   
            elif totrade == ToTrade.CloseLong:
                CloseTrade(account, s)
                OpenTrade(account, s, df['Market'][len(df) - 1], df['Stock'][len(df) - 1], 1)  
            elif totrade == ToTrade.CloseShort:  
                CloseTrade(account, s)
                OpenTrade(account, s, df['Market'][len(df) - 1], df['Stock'][len(df) - 1], -1)       
sys.stdout.flush()        


