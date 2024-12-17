import numpy as np
import pandas as pd
from gym import spaces
from tradingenv import TradingEnv
from dqn import DQNAgent
import torch
import os  # For setting the script's directory as the working directory
from data import get_data_indicators
from sklearn.model_selection import train_test_split

def train_model_old():
    df = get_data_indicators()

    # Ensure all required columns are numeric
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                        'SMA_20', 'SMA_50', 'EMA_10', 'RSI', 
                        'Upper_BB', 'Middle_BB', 'Lower_BB', 
                        'ATR', 'VWAP', 'MACD', 'MACD_Signal', 
                        'MACD_Hist']
    
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.dropna(subset=required_columns, inplace=True)

    env = TradingEnv(df)
    state_size = 16  # Ensure this matches the number of features in your observation
    action_size = 3  # Buy, Sell, Hold
    agent = DQNAgent(state_size, action_size)

    episodes = 50
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

def train_model():
    # Fetch and prepare the daily data
    df = get_data_indicators()  # Get daily data with indicators
    
    # Train-Test split (80/20)
    train_data, test_data = train_test_split(df, test_size=0.2, shuffle=False)
    
    # Create the environment with daily data
    train_env = TradingEnv(train_data)
    test_env = TradingEnv(test_data)
    
    state_size = 16  # Number of features for daily data
    action_size = 3  # Buy, Sell, Hold
    agent = DQNAgent(state_size, action_size)
    
    # Train the model
    episodes = 1000
    for e in range(episodes):
        state = train_env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0  # Initialize cumulative reward for the episode

        for time in range(500):  # Max number of steps per episode
            action = agent.act(state)
            next_state, reward, done, _ = train_env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            # Accumulate reward for the episode
            episode_reward += reward
            
            # Train the agent with the replay method
            agent.replay(32)
            
            # Calculate the loss
            # The total loss will be updated inside the replay method, so we add that to the total loss
            # If you need a more granular view of the loss, print it after each step inside replay

            if done:
                break

        # Log the episode details
        print(f"Episode {e+1}/{episodes} - Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    # Save the model after training
    torch.save(agent.model.state_dict(), 'DQN/dqn_model.pth')
    
    # Evaluate on the test set
    print("Training complete. Starting evaluation on the testing set...")

    if 'signal' not in test_data.columns:
        test_data['signal'] = calculate_signals(test_data)

    correct_predictions = 0
    total = 0
    state_columns = [
            'Open', 'High', 'Low', 'Close', 
            'SMA_20', 'SMA_50', 'EMA_10', 
            'RSI', 'Upper_BB', 'Middle_BB', 
            'Lower_BB', 'ATR', 'VWAP', 
            'MACD', 'MACD_Signal', 'MACD_Hist'
        ]

    for index, row in test_data.iterrows():
        state = row[state_columns].values  # Extract the state from the row
        true_action = row['signal']            # True action from the signal column

        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Shape: (1, state_size)
        with torch.no_grad():  # No gradient calculation needed
            action_values = agent.model(state_tensor)  # Get action values
            predicted_action = torch.argmax(action_values[0]).item()  # Get the action with the highest Q-value

        # Update counters
        if predicted_action == true_action:
            correct_predictions += 1
        total += 1

    # Calculate and print accuracy
    accuracy = (correct_predictions / total) * 100
    print(f"Evaluation Accuracy: {accuracy:.2f}%")

def calculate_signals(test_data):
    """
    Calculate the best action (signal) for each state in the testing set based on future price movement.
    
    Args:
        test_data: DataFrame containing the testing data with a 'Close' column and a 'Position' column.
                  'Position' should be 1 if there's a position and 0 otherwise.
    
    Returns:
        A Series of best actions (0: Buy, 1: Sell, 2: Hold) for the testing data.
    """
    signals = []
    for i in range(len(test_data) - 1):  # We stop at len(test_data) - 1 to avoid out-of-bounds access
        current_close = test_data.iloc[i]['Close']
        next_close = test_data.iloc[i + 1]['Close']
        position = test_data.iloc[i]['Position']

        if next_close > current_close:  # Next close is higher
            if position == 0:
                signals.append(0)  # Buy
            else:
                signals.append(2)  # Hold
        else:  # Next close is lower
            if position == 1:
                signals.append(1)  # Sell
            else:
                signals.append(2)  # Hold

    # For the last row, since no future close price is available, append Hold (2)
    signals.append(2)
    return pd.Series(signals, index=test_data.index)


    

# def main():
#     train_model()

# if __name__ == "__main__":
#     main()