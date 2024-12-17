import pandas as pd
import ta as ta
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import ta.momentum
import ta.trend
import ta.volatility  # For normalization
import yfinance as yf
from datetime import datetime, timedelta

def get_data_from_yfinance():
    start_date = datetime.now() - timedelta(days=730)
    df = yf.download('TSLA', interval='1d', period='5y')
    # print(df)

    # Flatten multi-level columns and rename them
    # df = df.reset_index()  # Move 'Datetime' from index to column
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]  # Remove multi-level
    
    # Select and rename the desired columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # df.to_csv('test.csv')
    return df

def get_data_indicators():
    # Read the data from CSV file
    # df = pd.read_csv('DQN/tsla_training.csv')
    df = get_data_from_yfinance()

    # Convert 'timestamp' from string to datetime
    # df['Local time'] = pd.to_datetime(df['Local time'])
    # df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Ensure the data has the following columns: Open, Close, High, Low, Volume
    # df['Open'], df['Close'], df['High'], df['Low'], df['Volume'], df['puts'], df['calls']

    # 1. 20 SMA
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)

    # 2. 50 SMA
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)

    # 3. 10 EMA
    df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)

    # 4. RSI (Relative Strength Index)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

    # 5. Bollinger Bands (using a 20-period moving average, and 2 standard deviations)
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['Upper_BB'] = bollinger.bollinger_hband()
    df['Middle_BB'] = bollinger.bollinger_mavg()
    df['Lower_BB'] = bollinger.bollinger_lband()

    # 6. ATR (Average True Range)
    # Calculate ATR
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14, fillna=True)

    # Check the first 20 rows to see the ATR values
    # print(df[['High', 'Low', 'Close', 'ATR']].head(20))

    # 7. VWAP (Volume Weighted Average Price) - Custom calculation
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

    # 8. MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()

    # 9. Chaikin Money Flow (CMF)
    # df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=3)
    # 10. Fibonacci Retracement Levels
    # def fibonacci_retracement(high, low):
    #     diff = high - low
    #     return {
    #         '23.6%': high - 0.236 * diff,
    #         '38.2%': high - 0.382 * diff,
    #         '50%': high - 0.500 * diff,
    #         '61.8%': high - 0.618 * diff,
    #         '78.6%': high - 0.786 * diff
    #     }

    # high_max = df['High'].max()
    # low_min = df['Low'].min()
    # fib_levels = fibonacci_retracement(high_max, low_min)
    # df['Fib_23.6'] = fib_levels['23.6%']
    # df['Fib_38.2'] = fib_levels['38.2%']
    # df['Fib_50'] = fib_levels['50%']
    # df['Fib_61.8'] = fib_levels['61.8%']
    # df['Fib_78.6'] = fib_levels['78.6%']

    # 11. PCR (Put/Call Ratio)
    #df['PCR'] = np.where(df['calls'] != 0, df['puts'] / df['calls'], np.nan)

    # Normalization
    scaler = MinMaxScaler()
    columns_to_normalize = ['Open', 'High', 'Low', 'Close', 'SMA_20', 'SMA_50', 'EMA_10', 'RSI', 
                            'Upper_BB', 'Middle_BB', 'Lower_BB', 'ATR', 'VWAP', 
                            'MACD', 'MACD_Signal', 'MACD_Hist'
                            ]

    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize]) # Fit and transform specified columns

    # Drop rows with missing values
    df.dropna(inplace=True)

    # df.to_csv('test_indicators.csv')

    return df

    # # Save the DataFrame with the technical indicators to a new CSV file
    # df.to_csv('tsla_training_indicators.csv', index=False)

    # # View the DataFrame
    # print(df.head())