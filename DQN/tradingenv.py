import numpy as np
import pandas as pd
import gym
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.max_steps = len(data) - 1
        
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(18,), dtype=np.float32)
        
        # Initialize portfolio state
        self.cash = 10000  # Starting cash in dollars
        self.stock = 0  # Number of stocks currently held (0 or 1)
        self.stock_price = 0  # Price at which we bought stock (for calculating profits)
        
    def reset(self):
        self.current_step = 0
        self.cash = 10000  # Reset cash
        self.stock = 0  # Reset stock holdings
        self.stock_price = 0  # Reset stock purchase price
        return self._get_observation()  # Return the first observation
    
    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        current_price = self.data.iloc[self.current_step]['Close']
        prev_price = self.data.iloc[self.current_step - 1]['Close']
        
        reward = 0
        
        if action == 0:  # Buy
            if self.stock == 0 and self.cash >= current_price:  # Can only buy if no stock is held
                self.stock = 1  # Buy one share
                self.cash -= current_price  # Subtract cash
                self.stock_price = current_price  # Record the buy price
                reward = 0  # No reward until we sell
            
        elif action == 1:  # Sell
            if self.stock == 1:  # Can only sell if holding stock
                self.cash += current_price  # Add cash from selling
                reward = current_price - self.stock_price  # Profit or loss from buy price
                self.stock = 0  # Sold the stock, so no stock held anymore
        
        elif action == 2:  # Hold
            reward = 0  # No immediate reward for holding
        
        # Reward for price increase while holding
        if action == 2 and self.stock == 1:  # Holding a stock and have a position
            reward += current_price - prev_price  # Gain due to price increase while holding, penalty for holding while price falls
        
        # Reward based on change in portfolio value
        # reward += portfolio_value - (self.cash + self.stock * prev_price)  # Portfolio change
        
        obs = self._get_observation()  # Get the new observation
        return obs, reward, done, {}

    
    def _get_observation(self):
        """Return the current observation state."""
        features = self.data.iloc[self.current_step][[
            'Open', 'High', 'Low', 'Close', 
            'SMA_20', 'SMA_50', 'EMA_10', 
            'RSI', 'Upper_BB', 'Middle_BB', 
            'Lower_BB', 'ATR', 'VWAP', 
            'MACD', 'MACD_Signal', 'MACD_Hist'
        ]].astype(np.float32).values
        
        return features
