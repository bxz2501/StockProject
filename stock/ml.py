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
    query = f"""select top 2050 m1.date as 'Date',DATEPART(hour, m1.Date) AS 'Hour', m1.[Close] as 'Market', m2.[close] as 'Stock'  from MinuteQuote m1
    inner join MinuteQuote m2 on m1.Date = m2.date
    where m1.Date > getdate() - 720 and m1.Contract = 'ym' and m2.Contract = '{symbol}'
    and (DATEPART(minute, m1.Date) % 30 = 0)
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
def getCurrentPosition(symbol):
    query = f"""select * from ArbitrageMLTrade
                where AccountName = 'xieie181' and Symbol1 = '{symbol}' and Active = 1"""

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        df = pd.read_sql(query, conn)
        if df.empty:
            return 0
        else:
            return df['Position'][0]


# %%
def OpenTrade(symbol, marketPrice, stockPrice, position):
    share2 = 1.0
    share1 = round((marketPrice / stockPrice) / 100, 0) * 100
    
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
           ('xieie181'
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
def CloseTrade(symbol):
    query = f"""UPDATE [dbo].[ArbitrageMLTrade]
           SET Active = 0 WHERE Symbol1 = '{symbol}' AND Active = 1"""

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()

# %%
def transformData(df):
    df["StockReturn"] = df.rolling(2).apply(
        lambda x: x.iloc[1] / x.iloc[0] - 1)["Stock"]
    df["MarketReturn"] = df.rolling(2).apply(
        lambda x: x.iloc[1] / x.iloc[0] - 1)["Market"]
    df["OutPerform"] = df["StockReturn"] - df["MarketReturn"]
    df["Target"] = (df.apply(lambda x: x > 0)["OutPerform"]).astype(int)


# %%
def getPredictors(df):
    predictors = ['Hour']
    for i in range(12):
        df[f'OutPerform{pow(2,i)*5}'] = df['OutPerform'].rolling(pow(2, i)).sum()
        predictors.append(f'OutPerform{pow(2,i)*5}')
        df[f'StockReturn{pow(2,i)*5}'] = df['StockReturn'].rolling(pow(2, i)).sum()
        predictors.append(f'StockReturn{pow(2,i)*5}')

    return predictors


# %%
def getMLdata(df, predictors):
    prev = df.copy()
    prev = prev.shift(1)
    data = df[["OutPerform", "Target"]]
    data = data.join(prev[predictors])
    data = data.copy().dropna()
    return data


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
        if prediction > .55:
            return ToTrade.OpenLong
        elif prediction < .45:
            return ToTrade.OpenShort

    if currentPosition == 1:
        if prediction < .45:
            return ToTrade.CloseShort
        elif prediction < .48:
            return ToTrade.Close

    if currentPosition == -1:
        if prediction > .55:
            return ToTrade.CloseLong
        if prediction > .52:
            return ToTrade.Close


# %%
symbols = [
    'CVX',
    'HON',
    'CRM',
    'UNH',
    'CSCO',
    'WMT',
    'AXP',
    'JPM',
    'MCD',
    'HD',
    'AMGN',
    'V',
    'INTC',
    'WBA',
    'GS',
    'JNJ',
    'PG',
    'AAPL',
    'DIS',
    'MMM',
    'MRK',
    'MSFT',
    'TRV',
    'VZ',
    'IBM',
    'CAT',
    'NKE']
for s in symbols:
    df = getData(s)
    transformData(df)
    predictors = getPredictors(df)
    data = getMLdata(df, predictors)
    with open(s, 'rb') as f:
        model = pickle.load(f)
        predictions = model.predict_proba(data[predictors])[:, 1]
        currentPosition = getCurrentPosition(s)
        totrade = processResult(currentPosition, predictions)
        if totrade == ToTrade.OpenLong:
            OpenTrade(s, df['Market'][len(df) - 1], df['Stock'][len(df) - 1], 1)
        elif totrade == ToTrade.OpenShort:     
            OpenTrade(s, df['Market'][len(df) - 1], df['Stock'][len(df) - 1], -1)
        elif totrade == ToTrade.Close:
            CloseTrade(s)   
        elif totrade == ToTrade.CloseLong:
            CloseTrade(s)
            OpenTrade(s, df['Market'][len(df) - 1], df['Stock'][len(df) - 1], 1)  
        elif totrade == ToTrade.CloseShort:  
            CloseTrade(s)
            OpenTrade(s, df['Market'][len(df) - 1], df['Stock'][len(df) - 1], -1)       


