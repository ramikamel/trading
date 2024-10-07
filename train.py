import numpy as np
import pandas as pd
from gym import spaces
from tradingenv import TradingEnv
from dqn import DQNAgent
import torch

if __name__ == "__main__":
    df = pd.read_csv('tsla_training_indicators.csv')

    # Ensure all required columns are numeric
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                        'SMA_20', 'SMA_50', 'EMA_10', 'RSI', 
                        'Upper_BB', 'Middle_BB', 'Lower_BB', 
                        'ATR', 'VWAP', 'MACD', 'MACD_Signal', 
                        'MACD_Hist', 'CMF']
    
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=required_columns, inplace=True)

    env = TradingEnv(df)
    state_size = 17  # Ensure this matches the number of features in your observation
    action_size = 3  # Buy, Sell, Hold
    agent = DQNAgent(state_size, action_size)

    episodes = 100
    for e in range(episodes):
        print(f"Episode: {e}/{episodes}")
        state = env.reset()
        state = np.reshape(state, [1, state_size])  # Make sure state is of shape (1, 18)
        
        for time in range(500):  # Limit the number of steps
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])  # Ensure shape matches

            # Check the type before passing to remember
            if isinstance(state, np.ndarray) and state.dtype.kind in 'fi':
                agent.remember(state, action, reward, next_state, done)

            state = next_state
            
            if done:
                print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break

            agent.replay(32)

    # Save the model after training
    torch.save(agent.model.state_dict(), 'dqn_model.pth')
