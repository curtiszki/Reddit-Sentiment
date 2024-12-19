import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from numpy import mean, std, sqrt
import config, util


# this function was taken from the first answer: https://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)
  
def main():
    data = util.readJSONFiles(config.OUTPUT_DIR, 'overall_grouped_sentiment_data_')
  
    #splitting the dataframe into the two categories to intepret
    pre_covid = data[data['date'] < '2020-03-01']
    post_covid = data[data['date'] >= '2020-03-01']
    
    data_to_print = data[['date', 'overall_sentiment_score', 'overall_top40_sentiment_score']]
    print(data_to_print)
    
    #get means and standard deviations for further testing
    pre_covid_avg = pre_covid['overall_sentiment_score'].mean()
    pre_covid_std = pre_covid['overall_sentiment_score'].std()
   
    post_covid_avg = post_covid['overall_sentiment_score'].mean() 
    post_covid_std = post_covid['overall_sentiment_score'].std()
    
    pre_covid_top40_avg = pre_covid['overall_top40_sentiment_score'].mean()
    pre_covid_top40_std = pre_covid['overall_top40_sentiment_score'].std()
   
    post_covid_top40_avg = post_covid['overall_top40_sentiment_score'].mean() 
    post_covid_top40_std = post_covid['overall_top40_sentiment_score'].std()
    
    
    #print first results
    print('Precovid average post sentiment score: ', pre_covid_avg)
    print('Postcovid average post sentiment score: ', post_covid_avg)
    print('Precovid top40 post sentiment score: ', pre_covid_top40_avg)
    print('Postcovid top40 post sentiment score: ', post_covid_top40_avg)
    
       
    #shapiro-wilks test done because data is too small for our regular stats.normal
    pre = pre_covid['overall_sentiment_score']
    post = post_covid['overall_sentiment_score']
    
    pre1 = pre_covid['overall_top40_sentiment_score']
    post1 = post_covid['overall_top40_sentiment_score']
    
    
    pre_normality = stats.shapiro(pre).pvalue
    post_normality = stats.shapiro(post).pvalue
    levene_overall = stats.levene(pre, post).pvalue
    
    pre_normality_top40 = stats.shapiro(pre1).pvalue
    post_normality_top40 = stats.shapiro(post1).pvalue
    levene_overall_top40 = stats.levene(pre1, post1).pvalue
    
    print('') 
    print('All posts: ')
    print('Precovid Normality: ', pre_normality)    # Precovid Normality:  0.1752023547887802
    print('Postcovid Normality: ', post_normality)  # Postcovid Normality:  0.004631702788174152 < 0.05 so much do mann-whitney
    print('Levene P-value: ', levene_overall)       # Levene P-value:  0.20727536316161055 > 0.05 so equal variance
   
    print('') 
    print('Top40 posts:')
    print('Precovid Normality: ', pre_normality_top40)      # Precovid Normality:  0.4017297923564911
    print('Postcovid Normality: ', post_normality_top40)    # Postcovid Normality:  0.27774184942245483
    print('Levene P-value: ', levene_overall_top40)         # Levene P-value:  0.12435057411621132 > 0.05 so equal variance
    
    
    mw_overall_pval = stats.mannwhitneyu(pre, post).pvalue
    ttest_top40 = stats.ttest_ind(pre1, post1).pvalue
    
    print('')
    print('Comparing pre and post covid stats through T-tests:')
    print('Overall data (Mann-Whitney): ', mw_overall_pval) #0.8329459088671121
    print('Top40 data (ttest_ind): ', ttest_top40) #0.5822607029761935
    #both are above 0.05 so not significant difference between sentiment scores pre/post covid
    
    
    a = pre_covid['overall_sentiment_score'].mean()
    b = post_covid['overall_sentiment_score'].mean()
    diff_pos = b - a
       
    c = pre_covid['overall_top40_sentiment_score'].mean()
    d = post_covid['overall_top40_sentiment_score'].mean()
    diff_neg = d - c
    
    print('')
    print('Mean Differences pre and post covid:')
    print('overall_sentiment_score: ', diff_pos) 
    print('overall_top40_sentiment_score: ', diff_neg)
    

    cohen_pos = cohen_d(pre, post)
    cohen_neg = cohen_d(pre1, post1)
    print('')
    print('Cohen D:')
    print('overall_sentiment_score: ', cohen_pos) 
    print('overall_top40_sentiment_score: ', cohen_neg)
      
    
    #print a plot of sentiment over time  
    plt.figure(figsize=(12, 4))
    plt.plot(data['date'], data['overall_sentiment_score'], label='Overall Sentiment Score')
    plt.plot(data['date'], data['overall_top40_sentiment_score'], label='Top 40 Sentiment Score', color = 'purple')
    plt.xlabel('Date')
    plt.ylabel('Sentiment Score')
    plt.axvline(x = pd.Timestamp('2020-03-01'), color = 'red', linestyle='--', label='Covid Began')
    plt.title('Sentiment Score on Reddit Submission Titles Over 3 years')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
