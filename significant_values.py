import config
from pyspark.sql import SparkSession, functions, types


def generateSignificantValues(sc):
    in_directory = config.OUTPUT_DIR+'/sentiment_data_'
    out_directory = config.OUTPUT_DIR+'/overallsign'
    out_directory1 = config.OUTPUT_DIR+'/subsign'
    submissions = sc.read.json(in_directory)
    
    #add date - based off Kai's function
    submissions = submissions.withColumn('format_month', functions.lpad(functions.col('month'), 2, '0'))\
        .withColumn('date', functions.to_date(functions.concat_ws('-', functions.col('year'), functions.col('format_month'), functions.lit('01')), 'yyyy-MM-dd'))
    
    #split into two categories
    submissions_pos = submissions.filter(submissions['sentiment_score'] > 0.5) 
    submissions_neg = submissions.filter(submissions['sentiment_score'] < -0.5) 
    
    #ordering them by sub and date
    submissions_pos = submissions_pos.orderBy('subreddit', 'date')
    submissions_neg = submissions_neg.orderBy('subreddit', 'date')
    
    #getting monthly count of pos and neg submissions
    monthly_pos_count = submissions_pos.groupBy('date').agg(functions.count('title').alias('num_pos_submissions')).orderBy('date')
    monthly_neg_count = submissions_neg.groupBy('date').agg(functions.count('title').alias('num_neg_submissions')).orderBy('date')
    monthly_total_count = submissions.groupBy('date').agg(functions.count('title').alias('overall_num_submissions')).orderBy('date')
    
    #getting monthly sum oscore of pos and neg submissions
    monthly_pos_score_sum = submissions_pos.groupBy('date').agg(functions.sum('score').alias('pos_sum_score')).orderBy('date')
    monthly_neg_score_sum = submissions_neg.groupBy('date').agg(functions.sum('score').alias('neg_sum_score')).orderBy('date')
    monthly_total_sum = submissions.groupBy('date').agg(functions.sum('score').alias('overall_sum_score')).orderBy('date')
    
    #getting monthly avg of pos and neg submissions
    monthly_pos_score_avg = submissions_pos.groupBy('date').agg(functions.avg('score').alias('pos_avg_score')).orderBy('date')
    monthly_neg_score_avg = submissions_neg.groupBy('date').agg(functions.avg('score').alias('neg_avg_score')).orderBy('date')
    monthly_total_score_avg = submissions.groupBy('date').agg(functions.avg('score').alias('total_avg_score')).orderBy('date')
    
    #combining data
    data_combined = monthly_pos_count.join(monthly_neg_count, on='date', how='outer')
    data_combined = data_combined.join(monthly_pos_score_sum, on='date', how='outer')
    data_combined = data_combined.join(monthly_neg_score_sum, on='date', how='outer')
    data_combined = data_combined.join(monthly_pos_score_avg, on='date', how='outer')
    data_combined = data_combined.join(monthly_neg_score_avg, on='date', how='outer')
    data_combined = data_combined.join(monthly_total_count, on='date', how='outer')
    data_combined = data_combined.join(monthly_total_sum, on='date', how='outer')
    data_combined = data_combined.join(monthly_total_score_avg, on='date', how='outer')
    
    
    #creating it for each subreddit
    sub_monthly_pos_count = submissions_pos.groupBy('date','subreddit').agg(functions.count('title').alias('num_pos_submissions')).orderBy('date','subreddit')
    sub_monthly_neg_count = submissions_neg.groupBy('date','subreddit').agg(functions.count('title').alias('num_neg_submissions')).orderBy('date','subreddit')
    sub_monthly_total_count = submissions.groupBy('date','subreddit').agg(functions.count('title').alias('num_total_submissions')).orderBy('date','subreddit')
    
    sub_monthly_pos_score_sum = submissions_pos.groupBy('date','subreddit').agg(functions.sum('score').alias('pos_sum_score')).orderBy('date','subreddit')
    sub_monthly_neg_score_sum = submissions_neg.groupBy('date','subreddit').agg(functions.sum('score').alias('neg_sum_score')).orderBy('date','subreddit')
    sub_monthly_total_score_sum = submissions.groupBy('date','subreddit').agg(functions.sum('score').alias('total_sum_score')).orderBy('date','subreddit')
    
    sub_monthly_pos_score_avg = submissions_pos.groupBy('date','subreddit').agg(functions.avg('score').alias('pos_avg_score')).orderBy('date','subreddit')
    sub_monthly_neg_score_avg = submissions_neg.groupBy('date','subreddit').agg(functions.avg('score').alias('neg_avg_score')).orderBy('date','subreddit')
    sub_monthly_total_score_avg = submissions.groupBy('date','subreddit').agg(functions.avg('score').alias('total_avg_score')).orderBy('date','subreddit')
    
    sub_data_combined = sub_monthly_pos_count.join(sub_monthly_neg_count, on=['date', 'subreddit'], how='outer')
    sub_data_combined = sub_data_combined.join(sub_monthly_pos_score_sum, on=['date', 'subreddit'], how='outer')
    sub_data_combined = sub_data_combined.join(sub_monthly_neg_score_sum, on=['date', 'subreddit'], how='outer')
    sub_data_combined = sub_data_combined.join(sub_monthly_pos_score_avg, on=['date', 'subreddit'], how='outer')
    sub_data_combined = sub_data_combined.join(sub_monthly_neg_score_avg, on=['date', 'subreddit'], how='outer')
    sub_data_combined = sub_data_combined.join(sub_monthly_total_count, on=['date', 'subreddit'], how='outer')
    sub_data_combined = sub_data_combined.join(sub_monthly_total_score_sum, on=['date', 'subreddit'], how='outer')
    sub_data_combined = sub_data_combined.join(sub_monthly_total_score_avg, on=['date', 'subreddit'], how='outer')

    
    data_combined.write.json(out_directory, mode='overwrite', compression='gzip')
    sub_data_combined.write.json(out_directory1, mode='overwrite', compression='gzip')    
    

