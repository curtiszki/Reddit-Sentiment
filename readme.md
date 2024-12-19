# CMPT 353 - Reddit
This project uses sentiment analysis to determine changes in sentiment over time across Reddit.

## Installing required libraries
Ideally set up a virtual environment and install using:
```
pip install -r requirements.txt
```

## Retrieving the Data
The data used was retrieved from the SFU cluster using reddit_extracter.py. It should retrieve submissions for selected subreddits from the 2019-2021 time period.
```
sentiment_scores.py all_cleaned_data
```
## Cleaning the Data
The data is cleaned using process_data.py which can be ran with no arguments. The corresponding input and output directories can be set up in the config.py file. Corresponding output is written to data/output/...
**Note:** This can take some time to complete.
```
spark-submit process_data.py
```

## Tests
If you are interested in seeing some of the statistical tests we ran during analysis, you can run them via:
```
python stats_overall.py
python overall_stats_pos_neg.py
spark-submit subreddit_tests.py
```
