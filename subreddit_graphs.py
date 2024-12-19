from scipy import stats
import config, util
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime
from matplotlib import ticker
import seaborn as sns
import pathlib

significance = 0.05

def generateGraphs(data):
    # Generate violin plots
    n = len(data['subreddit'].unique())

    sns.violinplot(data, x='subreddit', y='avg_sentiment_score')
    plt.title('Average Sentiment Score by Subreddit')
    plt.xlabel('Subreddit')
    plt.ylabel('Average Sentiment Score')
    plt.savefig(pathlib.Path('.').resolve()/config.GRAPHS_DIR/'violinplot.png')

    sns.violinplot(data, x='subreddit', y='top40_avg_sentiment_score')
    plt.title('Average Sentiment Score by Top 40 Submissions')
    plt.xlabel('Subreddit')
    plt.ylabel('Average Sentiment Score')
    plt.savefig(pathlib.Path('.').resolve()/config.GRAPHS_DIR/'top40violinplot.png')
    plt.show()
    plt.close()

    grouped = data.groupby('subreddit').agg({
        'avg_sentiment_score': 'median',
        'top40_avg_sentiment_score': 'mean'
    }).reset_index()

    grouped.sort_values(by=['avg_sentiment_score'], ascending=False, inplace=True)
    f, (ax1, ax2) = plt.subplots(2, 1)
    f.tight_layout()
    ax2 = sns.barplot(grouped, x='subreddit', y='avg_sentiment_score', errorbar=None, ax=ax2, dodge=False, hue='subreddit')
    ax2.bar_label(ax2.containers[0], fontsize=8, fmt='%.4f')
    ax2.set_title('Median Sentiment Score by Subreddit')
    ax2.set_xlabel('Subreddit')
    ax2.set_ylabel('Median Sentiment Score')
    ax2.set_xticks(range(len(grouped['subreddit'])))
    ax2.set_xticklabels(grouped['subreddit'], rotation=0)
    ax2.get_legend().remove()

    grouped.sort_values(by=['top40_avg_sentiment_score'],ascending=False, inplace=True)
    ax1 = sns.barplot(grouped, x='subreddit', y='top40_avg_sentiment_score', errorbar=None, ax=ax1, dodge=False, hue='subreddit')
    ax1.bar_label(ax1.containers[0], fontsize=8, fmt='%.4f')
    ax1.set_title('Mean Sentiment Score For Top 40 Submissions by Subreddit')
    ax1.set_xlabel('Subreddit')
    ax1.set_ylabel('Mean Sentiment Score')
    ax1.set_xticks(range(len(grouped['subreddit'])))
    ax1.set_xticklabels(grouped['subreddit'], rotation=0)
    ax1.get_legend().remove()
    plt.draw_all()
    plt.savefig(pathlib.Path('.').resolve()/config.GRAPHS_DIR/'average_sentiment.png')
    plt.show()
    plt.close()

    before_covid = data[((data['month'] < 3) & (data['year'] == 2020))
                        | (data['year'] < 2020)]
    after_covid = data[((data['month'] >= 3) & (data['year'] == 2020))
                       | (data['year'] >= 2021)]

    before_grouped = before_covid.groupby('subreddit').agg({
            'avg_sentiment_score': 'median',
            'top40_avg_sentiment_score': 'mean'
        })
    
    after_grouped = after_covid.groupby('subreddit').agg({
            'avg_sentiment_score': 'median',
            'top40_avg_sentiment_score': 'mean'
        })
    
    before_grouped.columns = ['before_median', 'before_mean']
    after_grouped.columns = ['after_median', 'after_mean']
    merged= before_grouped.join(after_grouped, on='subreddit', how='outer').reset_index()

    median = merged
    median['absolute_change'] = median['after_median'] - median['before_median']
    median = median.drop(['before_mean', 'after_mean'], axis=1)

    melted = pd.melt(median, id_vars='subreddit', value_name='average', var_name='time')
    g = sns.barplot(
        data=melted, x='subreddit', y='average', hue='time'
    )
    g.bar_label(g.containers[0], fontsize=8, fmt='%.4f')
    g.bar_label(g.containers[1], fontsize=8, fmt='%.4f')
    g.bar_label(g.containers[2], fontsize=8, fmt='%.4f')
    handles, _ = g.get_legend_handles_labels()
    g.legend(handles=handles, labels=['Pre-Covid Median', 'Post-Covid Median', 'Absolute Change'])
    plt.axhline(0, 0, len(median['subreddit'].unique()), color='#202020', linewidth=1)
    plt.xlabel('Subreddits')
    plt.ylabel('Median Sentiment')
    plt.title('Median sentiment before/after covid of general submissions')
    plt.savefig(pathlib.Path('.').resolve()/config.GRAPHS_DIR/'median_sentiment.png')
    plt.show()
    plt.close()
    median['relative_change'] = (median['after_median'] - median['before_median'])/np.abs(median['before_median']) * 100
    #print(median)

    mean = merged
    mean['absolute_change'] = mean['after_mean'] - mean['before_mean']
    mean = mean.drop(['before_median', 'after_median'], axis=1)

    melted = pd.melt(mean, id_vars='subreddit', value_name='average', var_name='time')
    g = sns.barplot(
        data=melted, x='subreddit', y='average', hue='time'
    )
    g.bar_label(g.containers[0], fontsize=8, fmt='%.4f')
    g.bar_label(g.containers[1], fontsize=8, fmt='%.4f')
    g.bar_label(g.containers[2], fontsize=8, fmt='%.4f')
    handles, _ = g.get_legend_handles_labels()
    g.legend(handles=handles, labels=['Pre-Covid Mean', 'Post-Covid Mean', 'Absolute Change'])
    plt.axhline(0, 0, len(mean['subreddit'].unique()), color='#202020', linewidth=1)
    plt.xlabel('Subreddits')
    plt.ylabel('Mean Sentiment')
    plt.title('Mean sentiment before/after covid of Top 40 submissions')
    plt.savefig(pathlib.Path('.').resolve()/config.GRAPHS_DIR/'mean_sentiment.png')
    plt.show()
    plt.close()
    mean['relative_change'] = (mean['after_mean'] - mean['before_mean'])/np.abs(mean['before_mean']) * 100

if __name__ == '__main__':
    data = util.readJSONFiles(config.OUTPUT_DIR , 'grouped_sentiment_data_')
    generateGraphs(data)