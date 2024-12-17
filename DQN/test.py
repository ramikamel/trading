import matplotlib.pyplot as plt
from data import get_data_indicators
from tradingenv import TradingEnv
from dqn import DQNAgent
import numpy as np
from train import train_model



def main():
    train_model()
    # simulate_and_compare()

if __name__ == "__main__":
    main()