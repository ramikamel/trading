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
        # Adjust the observation space to match the number of features
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(18,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self._get_observation()  # Return the observation directly

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        current_price = self.data.iloc[self.current_step]['Close']
        prev_price = self.data.iloc[self.current_step - 1]['Close']
        
        reward = 0
        if action == 0:  # Buy
            reward = current_price - prev_price
        elif action == 1:  # Sell
            reward = prev_price - current_price

        obs = self._get_observation()  # Get the new observation
        return obs, reward, done, {}

    def _get_observation(self):
        """Return the current observation state."""
        # Ensure you return exactly 18 features here
        features = self.data.iloc[self.current_step][[
            'Open', 'High', 'Low', 'Close', 
            'SMA_20', 'SMA_50', 'EMA_10', 
            'RSI', 'Upper_BB', 'Middle_BB', 
            'Lower_BB', 'ATR', 'VWAP', 
            'MACD', 'MACD_Signal', 'MACD_Hist', 
            'CMF'
        ]].astype(np.float32).values
        
        # Check if you have the correct number of features
        # print(f"Observation shape: {features.shape}")  # Check the shape here
        return features
