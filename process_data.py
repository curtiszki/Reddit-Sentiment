import os
import config, sentiment_scores
import significant_values as sv
from pyspark.sql import SparkSession, functions, types
from pyspark.sql import Window

spark = SparkSession.builder.appName('reddit extracter').getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


#reddit_submissions_path = '/courses/datasets/reddit_submissions_repartitioned/'

#output = 'reddit-subset'

submissions_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('num_comments', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('selftext', types.StringType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType())
])

grouped_schema = types.StructType([
    types.StructField('month', types.IntegerType()),
    types.StructField('score', types.LongType()),
    types.StructField('sentiment', types.StringType()),
    types.StructField('sentiment_score', types.DoubleType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('year', types.IntegerType())

])

# Clean the raw input data
def filter_data():
    # get data from 4 folders
    # TODO: how to read the zip file?
    reddit_submissions = spark.createDataFrame(data=[], schema=submissions_schema)
    dir = os.listdir(config.INPUT_DIR)
    for item in dir:
        if not os.path.isfile(item):
            path = os.path.join(os.path.abspath('.'), config.INPUT_DIR, item)
            additional_submissions = spark.read.json(path, schema=submissions_schema)
            reddit_submissions = reddit_submissions.unionByName(additional_submissions)

    filtered_data = reddit_submissions.select('subreddit', 'year', 'month', 'title', 'score')
    filtered_data = filtered_data.orderBy('subreddit', 'year', 'month')
    filtered_data.write.json(config.OUTPUT_DIR + '/all_cleaned_data', mode='overwrite', compression='gzip')

    return filtered_data

# Process filtered data for later testing
def process_data(reddit_submissions1):
    
    #added date columns later
    all_data = reddit_submissions1.filter(functions.col('title').isNotNull()).cache()
    #added gets overall avg sentiment score
    avg_sentiment_score = all_data.groupBy('year', 'month', 'subreddit').agg(
        functions.avg('sentiment_score').alias('avg_sentiment_score')
        )

    # get top 40 upvote score
    window2 = Window.partitionBy('year', 'month', 'subreddit').orderBy(functions.col('score').desc())
    ranked_score = all_data.withColumn('upvote_rank', functions.row_number().over(window2))
    top40_score = ranked_score.filter(functions.col('upvote_rank')<=40)
      
    #added gets top40 sentiment score
    top40_avg_sentiment = top40_score.groupBy('year', 'month', 'subreddit').agg(\
        functions.avg('sentiment_score').alias('top40_avg_sentiment_score')
        )
    
      
    sentiment_frequency = all_data.groupBy('year', 'month', 'subreddit', 'sentiment').agg(functions.count('sentiment').alias('frequency'))
     
         
    window1 = Window.partitionBy('year', 'month', 'subreddit').orderBy(functions.col('frequency').desc())
    sentiment_rank = sentiment_frequency.withColumn('sentiment_rank', functions.row_number().over(window1))
    sentiment_most = sentiment_rank.filter(functions.col('sentiment_rank')==1)\
          .select('year', 'month','subreddit','sentiment')\
          .withColumnRenamed('sentiment', 'sentiment_most')
       
        # most common sentiment for top 40
    sentiment_frequency_40 = top40_score.groupBy('year', 'month', 'subreddit', 'sentiment').agg(
            functions.count('sentiment').alias('frequency_40')
      )
         
    window3 = Window.partitionBy('year', 'month', 'subreddit').orderBy(functions.col('frequency_40').desc())
    # get the rank for sentiment frequency for top 40
    sentiment_rank_40 = sentiment_frequency_40.withColumn('sentiment_rank_40', functions.row_number().over(window3))
    sentiment_most_40 = sentiment_rank_40.filter(functions.col('sentiment_rank_40') == 1)\
        .select('year', 'month', 'subreddit', 'sentiment')\
        .withColumnRenamed('sentiment', 'top40_sentiment_most')
 
    #added - joining all dataframes together
    output_data = avg_sentiment_score.join(sentiment_most, on=['year', 'month', 'subreddit'], how='left')\
        .join(top40_avg_sentiment, on=['year', 'month', 'subreddit'], how='left')\
        .join(sentiment_most_40, on=['year', 'month', 'subreddit'], how='left')
   
    #selected the columns to output
    output_data = output_data.select('year', 'month', 'subreddit', 'avg_sentiment_score', 'sentiment_most', 'top40_avg_sentiment_score', 'top40_sentiment_most')

    #added date column      
    output_data = output_data.withColumn('format_month', functions.lpad(functions.col('month'), 2, '0'))\
        .withColumn('date', functions.to_date(functions.concat_ws('-', functions.col('year'), functions.col('format_month'), functions.lit('01')), 'yyyy-MM-dd'))
       
    #ordered by date column and subreddit 
    output_data = output_data.orderBy('subreddit', 'date')
    
    #output_data.show()
    out_directory = os.path.join(os.path.abspath('.'), config.OUTPUT_DIR)
    output_data.write.json(out_directory + '/grouped_sentiment_data_', mode='overwrite', compression='gzip')


    #data for each month
    overall_data = output_data.groupBy('date', 'year', 'month').agg(
        functions.avg('avg_sentiment_score').alias('overall_sentiment_score'),
        functions.avg('top40_avg_sentiment_score').alias('overall_top40_sentiment_score')
    )

    #get most common sentiment for each month
    overall_sentiment_frequency = output_data.groupBy('date', 'year', 'month', 'sentiment_most', 'format_month').agg(
        functions.count('sentiment_most').alias('sentiment_sum')
    )

    window4 = Window.partitionBy('year', 'month').orderBy(functions.col('sentiment_sum').desc())
    overall_sentiment_rank = overall_sentiment_frequency.withColumn('overall_sentiment_rank', functions.row_number().over(window4))
    #overall_sentiment
    overall_sentiment = overall_sentiment_rank.filter(functions.col('overall_sentiment_rank')==1) \
    .select('date', 'year', 'month','sentiment_most', 'format_month') \
    .withColumnRenamed('sentiment_most', 'overall_sentiment')

    output_data_overall = overall_data.join(overall_sentiment, on=['date', 'year', 'month'], how='left')\
        .withColumn('top40_sentiment', functions.col('overall_sentiment'))

    output_data_overall = output_data_overall.orderBy('date')
    output_data_overall.write.json(out_directory + '/overall_grouped_sentiment_data_', mode='overwrite', compression='gzip')

if __name__=='__main__':
    filtered_data = filter_data()
    print('Finished Filtering Data')
    sentiment = sentiment_scores.determine_sentiment(spark, filtered_data)
    print('Finished Parsing Sentiment')
    sentiment.write.json(config.OUTPUT_DIR + '/sentiment_data_', mode='overwrite', compression='gzip')
    process_data(sentiment)
    print('Finished Grouping Data')
    sv.generateSignificantValues(spark)
    print('Finished extracting significant values')