# subreddit_tests.py
from pyspark.sql import SparkSession, types, functions as f
from scipy import stats
import config
import pandas as pd
import numpy as np

spark = SparkSession.builder.appName('subreddit_tests').getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

grouped_schema = types.StructType([
    types.StructField('date', types.DateType()),
    types.StructField('format_month', types.StringType()), 
    types.StructField('month', types.IntegerType()),
    types.StructField('sentiment_most', types.StringType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('avg_sentiment_score', types.DoubleType()),
    types.StructField('top40_avg_sentiment_score', types.DoubleType()),
    types.StructField('top40_sentiment_most', types.StringType()),
    types.StructField('year', types.IntegerType())
])


def runTests():
    data = spark.read.json(config.OUTPUT_DIR + '/grouped_sentiment_data_', schema=grouped_schema)
        
    # Get values from sentiment columns
    sentiment = data.select(f.collect_list('avg_sentiment_score')).first()[0]
    top_sentiment = data.select(f.collect_list('top40_avg_sentiment_score')).first()[0]

    # Data per subreddit    
    pd_data = data.toPandas()
    subreddits = pd_data['subreddit'].unique()
    pd_data_by_subreddit = []

    for sr in subreddits:
        values = pd_data[pd_data['subreddit'] == sr]
        pd_data_by_subreddit.append(values)
    
    # before and after march 2020
    before_march_2020 = data.filter(((data.year == 2020)&(data.month<3))|(data.year < 2020))
    after_march_2020 = data.filter(((data.year == 2020)&(data.month>=3))|((data.year > 2020)))

    before_m2020_sentiment = before_march_2020.select(f.collect_list('avg_sentiment_score')).first()[0]
    before_m2020_top_sentiment = before_march_2020.select(f.collect_list('top40_avg_sentiment_score')).first()[0]

    after_m2020_sentiment = after_march_2020.select(f.collect_list('avg_sentiment_score')).first()[0]
    after_m2020_top_sentiment =after_march_2020.select(f.collect_list('top40_avg_sentiment_score')).first()[0]

    # Chi2 tests for independence
        
    # sentiment label by label
    pvalue = stats.chi2_contingency(pd.crosstab(pd_data.subreddit, pd_data.sentiment_most)).pvalue
    print(f"\nChi2 contingency test for sentiment top sentiment label: \
           \np value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")
    
    pvalue = stats.chi2_contingency(pd.crosstab(pd_data.subreddit, pd_data.top40_sentiment_most)).pvalue
    print(f"\nChi2 contingency test for sentiment top40 sentiment top label: \
           \np value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")

    # Test for normality
    pvalue = stats.normaltest(sentiment).pvalue
    print(f"\nNormality test for sentiment averages: \
           \np value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")
    
    pvalue = stats.normaltest(top_sentiment).pvalue
    print(f"\nNormality test for top submission sentiment averages: \
           \np value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")
    
    pvalue = stats.normaltest(before_m2020_sentiment).pvalue
    print(f"\nNormality test for submission sentiment averages pre Mar 2020: \
           \np value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")

    pvalue = stats.normaltest(before_m2020_top_sentiment).pvalue
    print(f"\nNormality test for top submission sentiment averages pre Mar 2020: \
           \np value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")
    
    pvalue = stats.normaltest(after_m2020_sentiment).pvalue
    print(f"\nNormality test for submission sentiment averages post Mar 2020: \
           \np value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")

    pvalue = stats.normaltest(after_m2020_top_sentiment).pvalue
    print(f"\nNormality test for top submission sentiment averages post Mar 2020: \
           \np value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")
    
    
    # Levene tests
    # Testing variance between pre-post 2020
    pvalue = stats.levene(before_m2020_sentiment, after_m2020_sentiment).pvalue
    print(f"\nLevene test for submission sentiment averages before/after Mar 2020: \
           \nP value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")
    
    pvalue = stats.levene(before_m2020_top_sentiment, after_m2020_top_sentiment).pvalue
    print(f"\nLevene test for top submission sentiment averages before/after Mar 2020: \
           \nP value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")
    
    
    # ANOVA tests
    #
    anova = stats.f_oneway(*([sr['avg_sentiment_score'] for sr in pd_data_by_subreddit]))
    print(f"\nANOVA test for submission sentiment averages: \
           \nP value: {anova.pvalue} \
           \nNull Hypothesis: {anova.pvalue >= 0.05}")
    #
    anova = stats.f_oneway(*([sr['top40_avg_sentiment_score'] for sr in pd_data_by_subreddit]))
    print(f"\nANOVA test for top submission sentiment averages in {sr}: \
           \nP value: {anova.pvalue} \
           \nNull Hypothesis: {anova.pvalue >= 0.05}")
    
    #
    for sr in pd_data_by_subreddit:
        name = sr.iloc[0]['subreddit']
        pvalue = stats.normaltest(sr['avg_sentiment_score']).pvalue
        print(f"\nNormality test for sentiment in {name}: \
           \nP value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")
        
        pvalue = stats.normaltest(sr['top40_avg_sentiment_score']).pvalue
        print(f"\nNormality test for top sentiment in {name}: \
           \nP value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")
    #
    pvalue = stats.kruskal(*([sr['top40_avg_sentiment_score'] for sr in pd_data_by_subreddit])).pvalue
    print(f"\nKruskal-Wallis test for median value: \
    \nP value: {pvalue} \
    \nNull Hypothesis: {pvalue >= 0.05}")

    min, max = pd_data['avg_sentiment_score'].min(), pd_data['avg_sentiment_score'].max()
    bins = np.arange(min, max+0.05, 0.05)
    labs = np.arange(0, abs(max-min)/0.05, 1)
    pd_data['bins'] = pd.cut(pd_data['avg_sentiment_score'], bins=bins, labels=labs)

    # Chi2 Contingency to check the independence of sentiment scores and subreddits
    pvalue = stats.chi2_contingency(pd.crosstab(pd_data['subreddit'], pd_data['bins'])).pvalue
    print(f'\nChi2 test for subreddit and scores avg: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')

    min, max = pd_data['top40_avg_sentiment_score'].min(), pd_data['top40_avg_sentiment_score'].max()
    bins = np.arange(min, max+0.05, 0.05)
    labs = np.arange(0, abs(max-min)/0.05, 1)
    pd_data['bins'] = pd.cut(pd_data['top40_avg_sentiment_score'], bins=bins, labels=labs)

    # Chi2 Contingency to check the independence of sentiment scores and subreddits
    pvalue = stats.chi2_contingency(pd.crosstab(pd_data['subreddit'], pd_data['bins'])).pvalue
    print(f'\nChi2 test for subreddit and scores top 40: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')
    
    before_grouped = before_march_2020.toPandas().groupby('subreddit').agg({
            'avg_sentiment_score': 'median',
            'top40_avg_sentiment_score': 'mean'
        })
    
    after_grouped = after_march_2020.toPandas().groupby('subreddit').agg({
            'avg_sentiment_score': 'median',
            'top40_avg_sentiment_score': 'mean'
        })
    
    before_grouped.columns = ['before_median', 'before_mean']
    after_grouped.columns = ['after_median', 'after_mean']
    merged= before_grouped.join(after_grouped, on='subreddit', how='outer').reset_index()

    median = merged
    median['absolute_change'] = median['after_median'] - median['before_median']
    median = median.drop(['before_mean', 'after_mean'], axis=1)
    median['relative_change'] = (median['after_median'] - median['before_median'])/np.abs(median['before_median']) * 100

    mean = merged
    mean['absolute_change'] = mean['after_mean'] - mean['before_mean']
    mean = mean.drop(['before_median', 'after_median'], axis=1)
    mean['relative_change'] = (mean['after_mean'] - mean['before_mean'])/np.abs(mean['before_mean']) * 100

    pvalue = stats.shapiro(median['before_median']).pvalue
    print(f'\nShapiro test for before median change: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')
    pvalue = stats.shapiro(median['after_median']).pvalue
    print(f'\nShapiro test for after median change: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')
    pvalue = stats.shapiro(mean['before_mean']).pvalue
    print(f'\nShapiro test for before mean change: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')
    pvalue = stats.shapiro(mean['after_mean']).pvalue
    print(f'\nShapiro test for after mean change: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')

    pvalue = stats.shapiro(median['absolute_change']).pvalue
    print(f'\nShapiro test for median absolute change: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')
    
    pvalue = stats.shapiro(median['relative_change']).pvalue
    print(f'\nShapiro test for median relative change: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')
    
    pvalue = stats.shapiro(mean['absolute_change']).pvalue
    print(f'\nShapiro test for mean absolute change: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')
    pvalue = stats.shapiro(mean['relative_change']).pvalue
    print(f'\nShapiro test for mean relative change: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')
    
    pvalue = stats.levene(median['absolute_change'], mean['absolute_change']).pvalue
    print(f'\nlevene test for median/mean absolute change: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')
    
    pvalue = stats.levene(median['relative_change'], mean['relative_change']).pvalue
    print(f'\nlevene test for median/mean relative change: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')
    
    pvalue = stats.levene(mean['before_mean'], mean['after_mean']).pvalue
    print(f'\nlevene test for equal variance before/after mean: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')
    pvalue = stats.levene(median['before_median'], median['after_median']).pvalue
    print(f'\nlevene test for equal variance before/after median: {pvalue}\
          \nP value: {pvalue} \
          \nNull Hypothesis: {pvalue >= 0.05}')
    
    pvalue= stats.ttest_ind(mean['relative_change'], median['relative_change']).pvalue
    print(f"\nIndependent t test for relative change: \
           \nP value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")
    pvalue= stats.mannwhitneyu(mean['relative_change'], median['relative_change']).pvalue
    print(f"\nMann-Whitney test for relative change: \
           \nP value: {pvalue} \
           \nNull Hypothesis: {pvalue >= 0.05}")
    
if __name__ == '__main__':
    runTests()