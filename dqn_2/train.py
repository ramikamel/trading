import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tradingenv import TradingEnv
from dqn import DQNAgent
from data import get_data_indicators

def train_model():
    data = get_data_indicators()

    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    env = TradingEnv(train_data)
    agent = DQNAgent(state_dim=data.shape[1], action_dim=3)

    episodes = 50
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            agent.replay()
        agent.update_target_model()
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

    evaluate_model(agent, test_data)

def evaluate_model(agent, test_data):
    env = TradingEnv(test_data)
    state = env.reset()
    done = False
    predictions = []
    actions = []
    true_prices = [test_data["Close"].iloc[0]]
    while not done:
        action = agent.act(state)
        predictions.append(action)
        actions.append(action)
        true_prices.append(env.data["Close"].iloc[env.current_step])
        state, _, done = env.step(action)

    # Calculate accuracy (percentage of correct predictions)
    accuracy = calculate_accuracy(predictions, actions, true_prices)
    print(f"Prediction Accuracy: {accuracy:.2f}%")
    
    # Plot the strategies
    plot_strategy(test_data, predictions, true_prices)

def calculate_accuracy(predictions, actions, true_prices):
    # Assuming that the correct action is to buy (0) when the price is going up,
    # sell (1) when the price is going down, and hold (2) when there's no significant change.
    correct_predictions = 0
    for i in range(1, len(predictions)):
        predicted_action = predictions[i]
        real_action = 2  # Default to 'hold'

        if actions[i] == 0 and true_prices[i] < true_prices[i-1]:  # Buy, but price went down
            real_action = 0
        elif actions[i] == 1 and true_prices[i] > true_prices[i-1]:  # Sell, price went up
            real_action = 1
        elif actions[i] == 2:
            real_action = 2
        
        if predicted_action == real_action:
            correct_predictions += 1

    return correct_predictions / len(predictions) * 100

def plot_strategy(data, predictions, true_prices):
    data = data.reset_index(drop=True)
    position = 1 # start with a position
    dqn_share_values = []
    last_sell_price = data["Close"].iloc[0]

    for i, action in enumerate(predictions):
        if action == 0 and position == 0:  # Buy if no position
            position = 1
            current_position = data["Close"].iloc[i]
        elif action == 1 and position == 1:  # Sell if has position
            position = 0
            current_position = data["Close"].iloc[i]
            last_sell_price = current_position
        elif action == 2 and position == 0: # hold with no position
            current_position = last_sell_price
        elif action == 2 and position == 1: # hold, have position
            current_position = data["Close"].iloc[i]
        elif action == 1 and position == 0: # sell prediction, but dont have position (same as hold)
            current_position = last_sell_price
        elif action == 0 and position == 1: # buy prediction, but aready have position (same as hold)
            current_position = data["Close"].iloc[i]
        else:
            current_position = last_sell_price

        dqn_share_values.append(current_position)

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, true_prices, label="Buy-and-Hold Strategy (1 Share)", color='blue')
    plt.plot(data.index, dqn_share_values, label="DQN Strategy (1 Share)", color='green')
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value (USD)")
    plt.title("DQN Strategy vs Buy-and-Hold for One Share of Tesla")
    plt.legend()
    plt.show()


def main():
    train_model()

if  __name__ == "__main__":
    main()