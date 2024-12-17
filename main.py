import time
import datetime
import pytz
import torch
import numpy as np
import schedule
import alpaca_trade_api as tradeapi
import pandas_market_calendars as mcal
import os

from DQN.dqn import DQN
from SentimentAnalysis.TestModel import SentAnal
from DQN.data import get_data_indicators
from DQN.train import train_model

# Python 3.10.4

# ALPACA LOGIN
# api key and secret should be located in .env file
api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_API_SECRET')
base_url="https://paper-api.alpaca.markets" # "https://api.alpaca.markets" for live trading
api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
try:
    account_info = api.get_account()
    print(f'Login successful at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
except tradeapi.rest.APIError as e:
    print(f"Login failed. Error: {e}")

# market hours and timezone
market_timezone = pytz.timezone('US/Central')
market_open = datetime.time(8, 30)
market_close = datetime.time(15, 0)

# stock to trade
symbol = 'TSLA'

# last trade date (if we decide to implement cooldown)
last_trade_date = None


"""
BUY AND SELL
"""

def buy(symbol, qty=5):
    global last_trade_date
    try:
        current_price = float(api.get_latest_trade(symbol).price)
        stop_price = float(current_price * 0.90)
        stop_price_truncated = float("%.2f" % stop_price)
        
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side='buy',
            type='market',
            time_in_force='gtc',
            order_class='oto',
            stop_loss={'stop_price': stop_price_truncated}
        )
        print(f'MARKET BUY order for {symbol}.')
        last_trade_date = datetime.datetime.now()
    except Exception as e:
        print(f"Error buying shares of {symbol}: {e}")

def sell(symbol, qty=5):
    global last_trade_date
    try:
        # Close stop loss
        open_orders = api.list_orders(status='open', symbols=[symbol])
        for order in open_orders:
            api.cancel_order(order.id)
            print(f"Stop sell order cancelled. id: {order.id}. type: {order.type}")
        
        time.sleep(60)
        
        # Close position
        api.close_position(symbol)
        print(f'CLOSING POSITION for {symbol}.')
        last_trade_date = datetime.datetime.now()
    except Exception as e:
        print(f"Error selling shares of {symbol}: {e}")

def __get_position_qty(symbol):
    try:
        qty = float(api.get_position(symbol).qty)
    except Exception:
        qty = 0
    return qty


"""
TRADING STRATEGY
"""

def DECISION():
    # # Ensure there's a cooldown period between trades (if we decide to impleent cooldown, uncomment this)
    # if last_trade_date:
    #     days_since_last_trade = (datetime.datetime.now() - last_trade_date).days
    #     if days_since_last_trade < 3:
    #         print(f"Cooldown active. Last trade was {days_since_last_trade} days ago.")
    #         return

    df = get_data_indicators()

    # Example of calling your DQN model to predict an action (if needed):
    state_size = 16  # Ensure this matches your model
    action_size = 3  # Number of actions (Buy, Sell, Hold)
    model_path = 'dqn_model.pth'
    loaded_agent = load_model(model_path, state_size, action_size)
    
    # Get the most recent row of data as the current state
    current_state = df.iloc[-1][[
        'Open', 'High', 'Low', 'Close', 'SMA_20', 'SMA_50', 'EMA_10', 
        'RSI', 'Upper_BB', 'Middle_BB', 'Lower_BB', 'ATR', 'VWAP', 
        'MACD', 'MACD_Signal', 'MACD_Hist'
    ]].values.astype(np.float32)  # Convert to a numpy array of type float32
    print(current_state)

    # actions from sentiment analysis and dqn model
    sent_anal_action = SentAnal('Tesla')
    dqn_action = predict_action(loaded_agent, current_state)

    def final_action(dqn_action, sent_anal_action):
        if dqn_action == sent_anal_action:
            return dqn_action
        elif dqn_action == 0 and sent_anal_action == 2:
            return 0 # Buy if DQN says buy and sentiment is neutral
        elif dqn_action == 2 and sent_anal_action == 1:
            return 1 # Sell if DQN is neutral and sentiment says sell
        elif dqn_action == 1 and sent_anal_action == 0:
            return 2 # Hold if DQN says sell and sentiment says buy
        else:
            return 2 # Default to hold in other cases
    
    action = final_action(dqn_action=dqn_action, sent_anal_action=sent_anal_action)

    current_position_quantity = __get_position_qty(symbol=symbol)

    if action == 0 and current_position_quantity <= 0: # buy if action is 0 and we dont have position
        buy(symbol=symbol)
    elif action == 1 and current_position_quantity > 0: # sell if action is 1 and we have position
        sell(symbol=symbol)
    else:
        print("holding")

def __is_market_open():
    """
    Checks if the current time is within market hours.
    """
    def __is_trading_day():
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        nyse = mcal.get_calendar("NYSE").schedule(start_date=today, end_date=today, tz='America/Chicago')
        return not nyse.empty
    
    if __is_trading_day():
        now = datetime.datetime.now(market_timezone).time()
        return market_open <= now <= market_close

def load_model(model_path, state_size, action_size):
    """Load a DQN model from a specified path."""
    model = DQN(state_size, action_size)
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    model.eval()
    return model

# Get DQN prediction
def predict_action(model, state):
    """Predict the action given a state."""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: (1, state_size)
    with torch.no_grad():  # No gradient calculation needed
        action_values = model(state_tensor)  # Get action values
        action = torch.argmax(action_values[0]).item()  # Get the action with the highest Q-value
    return action

def run_strategy():
    """
    Runs the strategy if the market is open and logs the time.
    """
    now = datetime.datetime.now(market_timezone)
    print(f"Checking market status at {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    if __is_market_open():
        print(f"Running strategy at {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        equity = float(account_info.equity)
        print(f"Account equity: {equity}")
        DECISION()
    else:
        print(f"Market is closed. Current time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}")

# # Schedule the strategy to run every hour on the hour
# schedule.every().hour.at(":00").do(run_strategy)

# # Train model every 60 days
# schedule.every(60).days.do(train_model)

def check_and_train_model():
    """Check if today is the first of the month and run train_model."""
    if datetime.datetime.now(market_timezone).day == 1:
        train_model()

# Schedule run_strategy to run at the market's closing session.
schedule.every().day.at("14:45").do(run_strategy)

# Schedule check_and_train_model to check daily at midnight.
schedule.every().day.at("00:00").do(check_and_train_model)

# main loop
def main():
    """
    Main function to keep the scheduler running.
    """
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    main()
