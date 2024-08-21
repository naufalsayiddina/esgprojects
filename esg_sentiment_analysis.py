# Data Import
import tweepy
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from wordcloud import WordCloud
import re

# Replace with your own Twitter API credentials
consumer_key = "YOUR_CONSUMER_KEY"
consumer_secret = "YOUR_CONSUMER_SECRET"
access_token = "YOUR_ACCESS_TOKEN"
access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

# Authenticate to Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Tweet Collection
query = "ESG"
max_tweets = 100
tweets = tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode="extended").items(max_tweets)
# noinspection PyTypeChecker
tweets_df: DataFrame = pd.DataFrame([tweet.full_text for tweet in tweets], columns={'Tweet'})

nltk.download('stopwords')
nltk.download('punkt')


# Cleaning Tweet Text
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+|#', '', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [word for word in tweet_tokens if word.lower() not in stopwords.words('english')]
    return " ".join(filtered_words)


tweets_df['Cleaned_Tweet'] = tweets_df['Tweet'].apply(clean_tweet)

sia = SentimentIntensityAnalyzer()
tweets_df['Sentiment'] = tweets_df['Cleaned_Tweet'].apply(lambda tweet: sia.polarity_scores(tweet)['compound'])


def classify_sentiment(score):
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"


tweets_df['Sentiment_Label'] = tweets_df['Sentiment'].apply(classify_sentiment)

# Data Visualization
sns.countplot(x='Sentiment_Label', data=tweets_df, palette='viridis')
plt.title('Sentiment Distribution of ESG-Related Tweets on Indonesia')
plt.show()

# Generate Word Cloud
all_words = ' '.join([text for text in tweets_df['Cleaned_Tweet']])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, background_color="white").generate(
    all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()