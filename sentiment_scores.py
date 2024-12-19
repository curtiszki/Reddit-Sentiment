from nltk.sentiment.vader import SentimentIntensityAnalyzer, SentiText, VaderConstants
import pandas as pd
from pyspark.sql.types import MapType,StringType
from pyspark.sql.functions import  udf, col, pandas_udf, from_json, monotonically_increasing_id
from pyspark.sql import SparkSession, functions, types, Column

sid_obj = SentimentIntensityAnalyzer()

# https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/ for reference
def get_sentiment(submission):

    sentiment = sid_obj.polarity_scores(submission)
    
    if sentiment['compound'] >= 0.05:
        return 'Postive'
    elif sentiment['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


def get_score(submission):
    sentiment = sid_obj.polarity_scores(submission)
    return sentiment['compound']

def determine_sentiment(sc, submissions): 
    sentiment_val = udf(get_sentiment, returnType=types.StringType())
    sentiment_score = udf(get_score, returnType=types.FloatType())

    submissions = submissions.withColumn('sentiment',  sentiment_val(col('title')))
    submissions = submissions.withColumn('sentiment_score', sentiment_score(col('title')))
    return submissions
