import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta, time
import pytz

# ---------------------------------------------------------
# 1. 設定と関数定義
# ---------------------------------------------------------

def get_usdjpy_rate():
    try:
        ticker = yf.Ticker("USDJPY=X")
        data = ticker.history(period="1d")
        if not data.empty:
            return data['Close'].iloc[-1]
        return 150.0
    except:
        return 150.0

def calculate_technical_indicators(df):
    df = df.copy()
    # 欠損値対策
    df = df.ffill().dropna()
    
    # 基本指標
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['BB_upper'] = df['SMA_20'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_lower'] = df['SMA_20'] - 2 * df['Close'].rolling(window=20).std()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Momentum'] = df['Close'] - df['Close'].shift(10)

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']

    df['SMA_20_Slope'] = df['SMA_20'].diff()

    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # ADX
    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['+DM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0.0)
    df['-DM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0.0)
    
    safe_atr = df['ATR'].replace(0, np.nan)
    df['+DI'] = 100 * (pd.Series(df['+DM']).ewm(alpha=1/14, adjust=False).mean() / safe_atr)
    df['-DI'] = 100 * (pd.Series(df['-DM']).ewm(alpha=1/14, adjust=False).mean() / safe_atr)
    di_sum = df['+DI'] + df['-DI']
    df['DX'] = 100 * np.abs(df['+DI'] - df['-DI']) / di_sum.replace(0, np.nan)
    df['ADX'] = df['DX'].ewm(alpha=1/14, adjust=False).mean()
    
    df['RSI_Lag1'] = df['RSI'].shift(1)
    df['Close_Pct_Change'] = df['Close'].pct_change()
    df['Close_Pct_Lag1'] = df['Close_Pct_Change'].shift(1)

    df.dropna(inplace=True)
    return df

def fetch_and_resample_data(ticker, timeframe, duration_days):
    """
    指定された期間と時間足でデータを取得・リサンプルする
    """
    end_date = datetime.now(pytz.utc) + timedelta(days=2)
    start_date = end_date - timedelta(days=duration_days + 60)
    
    yf_intervals = {"5m": "5m", "1h": "1h", "1d": "1d"}
    
    # yfinanceから
