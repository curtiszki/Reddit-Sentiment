
import sys
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy import mean, std, sqrt
import config, util


# this function was taken from the first answer: https://stackoverflow.com/questions/21532471/how-to-calculate-cohens-d-in-python
def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (mean(x) - mean(y)) / sqrt(((nx-1)*std(x, ddof=1) ** 2 + (ny-1)*std(y, ddof=1) ** 2) / dof)

    
def main():
    data = util.readJSONFiles(config.OUTPUT_DIR, 'overallsign')
  
    print(data)
  
    #splitting the dataframe into the two categories to intepret
    pre_covid = data[data['date'] < '2020-03-01']
    post_covid = data[data['date'] >= '2020-03-01']
    
    pre_covid_pos_avg = pre_covid['pos_avg_score'].mean()
    pre_covid_pos_std = pre_covid['pos_avg_score'].std()
   
    post_covid_pos_avg = post_covid['pos_avg_score'].mean() 
    post_covid_pos_std = post_covid['pos_avg_score'].std()
    
    pre_covid_neg_avg = pre_covid['neg_avg_score'].mean()
    pre_covid_neg_std = pre_covid['neg_avg_score'].std()
   
    post_covid_neg_avg = post_covid['neg_avg_score'].mean() 
    post_covid_neg_std = post_covid['neg_avg_score'].std()
    
    print('Precovid Significantly Positive Avg Score: ', pre_covid_pos_avg)
    print('Postcovid Significantly Positive Avg Score: ', post_covid_pos_avg)
    print('Precovid Significantly Negative Avg Score: ', pre_covid_neg_avg)
    print('Postcovid Significantly Negative Avg Score: ', post_covid_neg_avg)

   #shapiro-wilks test done because data is too small for our regular stats.normal
    pre = pre_covid['pos_avg_score']
    post = post_covid['pos_avg_score']
    
    pre1 = pre_covid['neg_avg_score']
    post1 = post_covid['neg_avg_score']
    
    pre2 = pre_covid['total_avg_score']
    post2 = post_covid['total_avg_score']
      
    pre_normality = stats.shapiro(pre).pvalue
    post_normality = stats.shapiro(post).pvalue
    levene_overall = stats.levene(pre, post).pvalue   
    
    pre_normality_neg = stats.shapiro(pre1).pvalue
    post_normality_neg = stats.shapiro(post1).pvalue
    levene_overall_neg = stats.levene(pre1, post1).pvalue
    
    pre_normality_tot = stats.shapiro(pre2).pvalue
    post_normality_tot = stats.shapiro(post2).pvalue
    levene_overall_tot = stats.levene(pre2, post1).pvalue
    
    print('') 
    print('Positive: ')
    print('Precovid Normality: ', pre_normality)   
    print('Postcovid Normality: ', post_normality) 
    print('Levene P-value: ', levene_overall)      
   
    print('') 
    print('Negative:')
    print('Precovid Normality: ', pre_normality_neg)      
    print('Postcovid Normality: ', post_normality_neg)    
    print('Levene P-value: ', levene_overall_neg)   
    
    print('') 
    print('Total:')
    print('Precovid Normality: ', pre_normality_tot)      
    print('Postcovid Normality: ', post_normality_tot)    
    print('Levene P-value: ', levene_overall_tot) 
    
    # Positive: 
    # Precovid Normality:  0.5511788725852966
    # Postcovid Normality:  0.3545244038105011
    # Levene P-value:  0.2685045867404562
    
    # Negative:
    # Precovid Normality:  0.3576086461544037
    # Postcovid Normality:  0.5503909587860107
    # Levene P-value:  0.008143624145598622 unequal variances
    
    ttest_pos = stats.ttest_ind(pre, post).pvalue
    ttest_neg = stats.ttest_ind(pre1, post1, equal_var=False).pvalue
    ttest_tot = stats.ttest_ind(pre2, post2, equal_var=False).pvalue
    
    print('')
    print('Comparing pre and post covid stats through T-tests:')
    print('Positive: ', ttest_pos) 
    print('Negative: ', ttest_neg) 
    print('Total Posts: ', ttest_tot)
    
    #Comparing pre and post covid stats through T-tests:
    # Positive:  1.455390944815971e-05
    # Negative:  1.2381149292792083e-09
    
    a = pre_covid['pos_avg_score'].mean()
    b = post_covid['pos_avg_score'].mean()
    diff_pos = b - a
       
    c = pre_covid['neg_avg_score'].mean()
    d = post_covid['neg_avg_score'].mean()
    diff_neg = d - c
    
    print('')
    print('Mean Differences avg score pre and post covid:')
    print('Positive: ', diff_pos) 
    print('Negative: ', diff_neg)
    

    cohen_pos = cohen_d(pre, post)
    cohen_neg = cohen_d(pre1, post1)
    print('')
    print('Cohen D on avg score:')
    print('Positive: ', cohen_pos) 
    print('Negative: ', cohen_neg)
    
    
    e = pre_covid['num_pos_submissions'].mean()
    f = post_covid['num_pos_submissions'].mean()
    diff_num_pos = f - e
    
    g = pre_covid['num_neg_submissions'].mean()
    h = post_covid['num_neg_submissions'].mean()
    diff_num_neg = h - g
    
    print('')
    print('Mean Differences pre and post covid for num submissions:')
    print('Positive: ', diff_num_pos) 
    print('Negative: ', diff_num_neg)
    
    cohen_pos_num = cohen_d(pre_covid['num_pos_submissions'], post_covid['num_pos_submissions'])
    cohen_neg_num = cohen_d(pre_covid['num_neg_submissions'], post_covid['num_neg_submissions'])
    print('')
    print('Cohen D on num submissions:')
    print('Positive: ', cohen_pos_num) 
    print('Negative: ', cohen_neg_num)
    
    i = pre_covid['overall_num_submissions'].mean()
    j = post_covid['overall_num_submissions'].mean()
    diff_num_total = j - i
    
    print('')
    print('Mean Differences pre and post covid for total num submissions: ', diff_num_total)
    print('mean before covid: ', i)
    print('mean after covid: ', j)
    cohen_total_num = cohen_d(pre_covid['overall_num_submissions'], post_covid['overall_num_submissions'])
    print('Cohen D on total num submissions: ', cohen_total_num)
    
    
    k = pre_covid['total_avg_score'].mean()
    l = post_covid['total_avg_score'].mean()
    diff_score_total = l - k
    
    print('')
    print('Mean Differences pre and post covid for total score: ', diff_score_total)
    print('mean before covid: ', k)
    print('mean after covid: ', l)
    cohen_total_score = cohen_d(pre_covid['total_avg_score'], post_covid['total_avg_score'])
    print('Cohen D on total score submissions: ', cohen_total_score)
 
    
        
    # plot of neg and pos submission numbers over the data
    plt.figure(figsize=(12, 4))
    plt.plot(data['date'], data['num_pos_submissions'], label='Number of Significantly Positive Submissions')
    plt.plot(data['date'], data['num_neg_submissions'], label='Number of Significantly Negative Submissions')
    plt.xlabel('Date')
    plt.ylabel('Number of Submissions')
    plt.axvline(x = pd.Timestamp('2020-03-01'), color = 'red', linestyle='--', label='Covid Began')
    plt.title('Number of Significant Positive and Negative Submissions over 3 years')
    plt.legend()
    plt.show()
    
    
    # plot of neg and pos submission average score over data
    plt.figure(figsize=(12, 4))
    plt.plot(data['date'], data['pos_avg_score'], label='Average score of Significantly Positive Submissions')
    plt.plot(data['date'], data['neg_avg_score'], label='Average score of Significantly Negative Submissions')
    plt.xlabel('Date')
    plt.ylabel('Average Score')
    plt.axvline(x = pd.Timestamp('2020-03-01'), color = 'red', linestyle='--', label='Covid Began')
    plt.title('Average Score of Significantly Positive and Negative Submissions over 3 years')
    plt.legend()
    plt.show()
    
    #overall number of submissions
    plt.figure(figsize=(12, 4))
    plt.plot(data['date'], data['num_pos_submissions'], label='Number of Significantly Positive Submissions')
    plt.plot(data['date'], data['num_neg_submissions'], label='Number of Significantly Negative Submissions')
    plt.plot(data['date'], data['overall_num_submissions'], label='Number of submissions', color = 'black')
    plt.xlabel('Date')     
    plt.ylabel('Number of Submissions')
    plt.axvline(x = pd.Timestamp('2020-03-01'), color = 'red', linestyle='--', label='Covid Began')
    plt.title('Number of Total and Significantly Positive and Negative Submissions over 3 years')
    plt.legend()
    plt.show()
    
    # plot of neg and pos submission and total average score over data
    plt.figure(figsize=(12, 4))
    plt.plot(data['date'], data['pos_avg_score'], label='Average score of Significantly Positive Submissions')
    plt.plot(data['date'], data['neg_avg_score'], label='Average score of Significantly Negative Submissions')
    plt.plot(data['date'], data['total_avg_score'], label='Average score of average post', color = 'black')
    plt.xlabel('Date')
    plt.ylabel('Average Score')
    plt.axvline(x = pd.Timestamp('2020-03-01'), color = 'red', linestyle='--', label='Covid Began')
    plt.title('Average Score of Total and Significantly Positive and Negative Submissions over 3 years')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
