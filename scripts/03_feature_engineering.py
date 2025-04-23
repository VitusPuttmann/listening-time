##############################################################################
##                                                                          ##
##  Project: Podcast Listening Time Prediction                              ##
##                                                                          ##
##  Script 03: Feature engineering                                          ##
##                                                                          ##
##                                                                          ##
##  Author: Vitus Puttmann                                                  ##
##                                                                          ##
##  Version: 1.1                                                            ##
##                                                                          ##
##  Date: 23.04.2025                                                        ##
##                                                                          ##
##############################################################################


##                                                                          ##
## Background information

# https://www.kaggle.com/datasets/ysthehurricane/
# podcast-listening-time-prediction-dataset?select=podcast_dataset_info.txt


##                                                                          ##
## Import libraries

import pandas as pd


##                                                                          ##
## Define functions

def shift_col(dataset: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    cols = dataset.columns.tolist()
    cols.remove(right)
    cols.insert(cols.index(left) + 1, right)
    dataset = dataset[cols]
    return dataset[cols]


##                                                                          ##
## Load data

X_train = pd.read_csv('data/01_01_X_train_prep.csv')

X_test = pd.read_csv('data/01_02_X_test_prep.csv')


##                                                                          ##
## Adapt features

X_train.info()


#                                                                           #
# Id

X_train['id'].describe()


#                                                                           #
# Podast name

X_train['podcast_name'].describe()


#                                                                           #
# Episode title

X_train['episode_title'].unique()
X_train['episode_title'].describe()

# -> Transform to numeric variable

for dataset in [X_train, X_test]:
    dataset['episode_title_num'] = (
        dataset['episode_title'].str.replace('Episode ', '')
    )
    dataset['episode_title_num'] = dataset['episode_title_num'].astype(int)

X_train = shift_col(X_train, 'episode_title', 'episode_title_num')
X_test = shift_col(X_test, 'episode_title', 'episode_title_num')


#                                                                           #
# Episode length

X_train['episode_length'].describe()
X_train['episode_length_imp'].describe()
X_train['episode_length_imp_dum'].describe()


#                                                                           #
# Genre

X_train['genre'].describe()


#                                                                           #
# Host popularity

X_train['host_popularity'].describe()


#                                                                           #
# Guest popularity

X_train['guest_popularity'].describe()
X_train['guest_popularity_imp'].describe()
X_train['guest_popularity_imp_dum'].describe()


#                                                                           #
# Publication day

X_train['publication_day'].unique()

weekday_mapping = {
    'Monday': 1,
    'Tuesday': 2,
    'Wednesday': 3,
    'Thursday': 4,
    'Friday': 5,
    'Saturday': 6,
    'Sunday': 7
}

X_train['publication_day_num'] = X_train['publication_day'].map(weekday_mapping)
X_train = shift_col(X_train, 'publication_day', 'publication_day_num')

X_test['publication_day_num'] = X_test['publication_day'].map(weekday_mapping)
X_test = shift_col(X_test, 'publication_day', 'publication_day_num')


#                                                                           #
# Publication time

X_train['publication_time'].unique()

daytime_mapping = {
    'Morning': 1,
    'Afternoon': 2,
    'Evening': 3,
    'Night': 4
}

X_train['publication_time_num'] = (
    X_train['publication_time'].map(daytime_mapping)
)
X_train = shift_col(X_train, 'publication_time', 'publication_time_num')

X_test['publication_time_num'] = (
    X_test['publication_time'].map(daytime_mapping)
)
X_test = shift_col(X_test, 'publication_time', 'publication_time_num')


#                                                                           #
# Number of ads

X_train['number_ads'].describe()


#                                                                           #
# Episode sentiment

X_train['episode_sentiment'].unique()

sentiment_mapping = {
    'Negative': 1,
    'Neutral': 2,
    'Positive': 3
}

X_train['episode_sentiment_num'] = (
    X_train['episode_sentiment'].map(sentiment_mapping)
)
X_train = shift_col(X_train, 'episode_sentiment', 'episode_sentiment_num')

X_test['episode_sentiment_num'] = (
    X_test['episode_sentiment'].map(sentiment_mapping)
)
X_test = shift_col(X_test, 'episode_sentiment', 'episode_sentiment_num')


##                                                                          ##
## Save dataset

X_train.to_csv('data/02_01_X_train_eng.csv', index=False)

X_test.to_csv('data/02_02_X_test_eng.csv', index=False)
