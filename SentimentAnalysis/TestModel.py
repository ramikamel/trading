import datetime
import torch
from transformers import pipeline
from newsapi import NewsApiClient

# Initialize the Hugging Face sentiment analysis pipeline
sentiment_analysis = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Initialize News API client (get your API key from https://newsapi.org)
newsapi = NewsApiClient(api_key='635757ea54df478c890d63eec011f44a')

# Function to fetch the most recent articles about Tesla
def get_stock_news(query, page_size=20, days_ago=7):
    # Calculate the date 'days_ago' from today
    from_date = (datetime.datetime.now() - datetime.timedelta(days=days_ago)).strftime('%Y-%m-%d')
    to_date = datetime.datetime.now().strftime('%Y-%m-%d')

    # Fetch the most recent news articles within the date range, exact match for Tesla
    articles = newsapi.get_everything(
        q=f'"{query}"',  # Exact match for Tesla
        language='en',
        sort_by='publishedAt',
        from_param=from_date,  # Start date
        to=to_date,  # End date (today)
        page_size=page_size  # Fetch more articles to ensure we get 5 valid ones
    )
    return articles['articles']

# Function to analyze sentiment of news articles and return an array of "Sentiment (Description)"
def analyze_news_sentiment(stock_name, max_articles=5):
    # Get news articles
    articles = get_stock_news(stock_name)
    
    # List of key terms to search for in the description to determine relevance
    key_terms = ['Tesla', 'Elon Musk', 'electric vehicle', 'EV', 'Autopilot', 'Cybertruck']

    # Counter for valid Tesla articles
    valid_articles_count = 0

    # Array to store the sentiment analysis results of the descriptions
    sentiment_descriptions = []

    for article in articles:
        if valid_articles_count >= max_articles:
            break  # Stop once we have 5 valid articles

        title = article['title']
        description = article['description']

        # Check if any of the key terms are in the description
        if not any(term in description for term in key_terms):
            print(f"Skipped non-relevant article based on description: {title}")
            print("-" * 80)
            continue
        
        # Perform sentiment analysis on the description
        sentiment_description = sentiment_analysis(description)
        
        # Append the sentiment result to the array
        sentiment_descriptions.append(sentiment_description)
        
        # Output the sentiment
        print(f"Title: {title}")
        print(f"Description: {description}")
        print(f"Sentiment (Description): {sentiment_description}")
        print("-" * 80)

        valid_articles_count += 1  # Increment the count of valid articles

    # Return the array of sentiment descriptions
    return sentiment_descriptions


def SentAnal(stockName):
    # Example: analyze sentiment for the past 5 Tesla stock news articles
    sentiment_data = analyze_news_sentiment(stockName)
    print("Sentiment (Description) Array:")
    print(sentiment_data)

    count = 0

    for i in sentiment_data:
        if (i[0]['label'] == 'POSITIVE' and i[0]['score'] > .5):
            count += 1
        elif (i[0]['label'] == 'NEGATIVE' and i[0]['score'] > .5):
            count -= 1
            
    if count >= 4:
        return 0
    elif count <= -4:
        return 1
    else:
        return 2
        
        
# print(SentAnal('Tesla'))