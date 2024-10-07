import pandas as pd
import ta as ta
import numpy as np

# Read the data from CSV file
df = pd.read_csv('tsla_training.csv')

# Convert 'timestamp' from string to datetime
df['Local time'] = pd.to_datetime(df['Local time'], format="%d.%m.%Y %H:%M:%S.%f GMT%z")

# Convert from UTC to Central Time (considering Daylight Saving Time)
df['Local time'] = df['Local time'].dt.tz_convert('America/Chicago')

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
print(df[['High', 'Low', 'Close', 'ATR']].head(20))


# 7. VWAP (Volume Weighted Average Price) - Custom calculation
df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()

# 8. MACD (Moving Average Convergence Divergence)
macd = ta.trend.MACD(df['Close'])
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['MACD_Hist'] = macd.macd_diff()

# 9. Chaikin Money Flow (CMF)
df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=3)
# 10. Fibonacci Retracement Levels
def fibonacci_retracement(high, low):
    diff = high - low
    return {
        '23.6%': high - 0.236 * diff,
        '38.2%': high - 0.382 * diff,
        '50%': high - 0.500 * diff,
        '61.8%': high - 0.618 * diff,
        '78.6%': high - 0.786 * diff
    }

high_max = df['High'].max()
low_min = df['Low'].min()
fib_levels = fibonacci_retracement(high_max, low_min)
df['Fib_23.6'] = fib_levels['23.6%']
df['Fib_38.2'] = fib_levels['38.2%']
df['Fib_50'] = fib_levels['50%']
df['Fib_61.8'] = fib_levels['61.8%']
df['Fib_78.6'] = fib_levels['78.6%']

# 11. PCR (Put/Call Ratio)
#df['PCR'] = np.where(df['calls'] != 0, df['puts'] / df['calls'], np.nan)

# Save the DataFrame with the technical indicators to a new CSV file
df.to_csv('tsla_training_indicators.csv', index=False)

# View the DataFrame
print(df.head())