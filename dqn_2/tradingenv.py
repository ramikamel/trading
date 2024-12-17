import numpy as np
import pandas as pd

class TradingEnv:
    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.balance = 10000
        self.position = 0  # 1 if holding, 0 if not
        self.position_price = 0
        self.initial_balance = self.balance
        self.actions = [0, 1, 2]  # buy, sell, hold

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        self.position_price = 0
        return self._get_state()

    def step(self, action):
        reward = 0
        done = False
        if action == 0:  # Buy
            if self.position == 0:  # Only buy if not already holding
                self.position = 1
                self.position_price = self.data.iloc[self.current_step]["Close"]
        elif action == 1:  # Sell
            if self.position == 1:  # Only sell if holding
                reward = self.data.iloc[self.current_step]["Close"] - self.position_price
                self.balance += reward
                self.position = 0
        elif action == 2:  # Hold
            pass  # No action taken

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True
            if self.position == 1:
                # Close any remaining position at the end of the episode
                self.balance += self.data.iloc[self.current_step]["Close"] - self.position_price
                self.position = 0

        return self._get_state(), reward, done

    def _get_state(self):
        return self.data.iloc[self.current_step].values

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")
