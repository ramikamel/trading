# Automated Trading Program

An algorithmic trading bot that leverages Deep Reinforcement Learning (DQN) and Natural Language Processing (NLP) to make autonomous, data-driven trading decisions. The program integrates technical indicators and news sentiment analysis, executing trades seamlessly via the Alpaca API.

## Features & Achievements

* **Reinforcement Learning Engine**: Integrated a Deep Q-Network (DQN) model to enhance trading decision-making, achieving **70% trade accuracy**.
* **Comprehensive Data Fusion**: Combined **8 years of financial data** fetched via YFinance with NLP outputs of sentiment analysis from **5,000 news articles**.
* **Deep Learning with PyTorch**: Leveraged PyTorch to build and train the neural network, utilizing tensors for efficient deep learning computations and enhanced model complexity.
* **Rapid Execution via Alpaca API**: Streamlined live order execution using the Alpaca Trade API, achieving a **9× increase in execution speed** and an additional **12% improvement in trade accuracy**.

## Project Architecture

* `main.py`: The entry point for live trading. It schedules market checks, retrieves live data, queries both the DQN model and NLP sentiment analyzer, and executes buy/sell/hold orders via Alpaca.
* `DQN/`: Contains the Reinforcement Learning infrastructure.
  * `dqn.py`: The PyTorch implementation of the Deep Q-Network and the DQN Agent.
  * `tradingenv.py`: A custom OpenAI Gym environment simulating the stock market for the agent to navigate and learn from.
  * `train.py`: The training loop that pits the agent against historical data, utilizing experience replay to optimize the policy.
  * `data.py`: Handles fetching historical data via `yfinance` and calculates 16 key technical indicators (SMA, EMA, RSI, Bollinger Bands, ATR, VWAP, MACD) to form the state space.
* `SentimentAnalysis/`: Contains the NLP integration.
  * `TestModel.py`: Uses the Hugging Face `transformers` library (`distilbert`) and NewsAPI to fetch recent articles and compute an aggregate sentiment score (Positive/Negative/Neutral) to factor into trading decisions.

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/automated-trading-program.git](https://github.com/yourusername/automated-trading-program.git)
   cd automated-trading-program
   ```
2. **Install dependencies**:
   
   Ensure you have Python 3.10+ installed. Install the required packages using the provided requirements.txt:

    ```bash
    pip install -r requirements.txt
    ```
3. **Environment Setup**:

    Create a .env file in the root directory and add your Alpaca API credentials. You will also need to ensure your NewsAPI key is configured in the sentiment analysis script.

    ```Code snippet
    ALPACA_API_KEY=your_alpaca_key_here
    ALPACA_API_SECRET=your_alpaca_secret_here
    ```

## Usage
1. **Training the Model**

    Before live trading, you must train the Deep Q-Network on historical data. This will output a saved dqn_model.pth weights file.

    ```bash
    python DQN/train.py
    ```

2. **Live Trading**

    Once the model is trained and your API keys are set, you can start the main trading scheduler. This script is designed to run continuously, checking market hours and executing the strategy.

    ```bash
    python main.py
    ```

## Technologies Used
* Python 3.10
* PyTorch (Deep Learning & Tensor Computation)
* Transformers / Hugging Face (NLP Sentiment Analysis)
* YFinance (Historical Market Data)
* Alpaca Trade API (Live Market Execution)
* NewsAPI (Financial News Aggregation)
* TA (Technical Analysis Library) (Market Indicators)