##############################################################################
##                                                                          ##
##  Project: Podcast Listening Time Prediction                              ##
##                                                                          ##
##  Script 03: Feature engineering                                          ##
##                                                                          ##
##                                                                          ##
##  Author: Vitus Puttmann                                                  ##
##                                                                          ##
##  Version: 2.0                                                            ##
##                                                                          ##
##  Date: 24.04.2025                                                        ##
##                                                                          ##
##############################################################################


##                                                                          ##
## Background information

# https://www.kaggle.com/datasets/ysthehurricane/
# podcast-listening-time-prediction-dataset?select=podcast_dataset_info.txt


##                                                                          ##
## Import libraries

import numpy as np
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

# -> Handcode topic based on title

topic_mapping = {
    'Crime':        'Crime',
    'Learning':     'Education',
    'Education':    'Education',
    'Tech':         'Technology',
    'Technology':   'Technology',
    'Laugh':        'Comedy',
    'Comedy':       'Comedy',
    'Athlete':      'Sport',
    'Sport':        'Sport',
    'Lifestyle':    'Lifestyle',
    'Fashion':      'Lifestyle',
    'Market':       'Economy',
    'Economy':      'Economy',
    'Style':        'Lifestyle',
    'News':         'News',
    'Daily':        'News',
    'Digital':      'Technology',
    'Funny':        'Comedy',
    'Criminal':     'Crime',
    'Wellness':     'Lifestyle',
    'Watch':        'News',
    'Melody':       'Music',
    'Music ':       'Music',
    'Gadget':       'Technology',
    'Sport':        'Sport',
    'Study':        'Education',
    'Living':       'Lifestyle',
    'Detective':    'Crime',
    'Finance':      'Economy',
    'Mystery':      'Crime',
    'Tune':         'Music',
    'Life':         'Lifestyle',
    'Humor':        'Comedy',
    'Money':        'Economy',
    'Brain':        'Education',
    'Sound':        'Music',
    'Fitness':      'Lifestyle',
    'Health':       'Lifestyle',
    'Business':     'Economy',
    'Joke':         'Comedy',
    'Body':         'Lifestyle',
    'Business':     'Economy',
    'Game':         'Technology',
    'Innovator':    'Technology',
    'Affairs':      'News'
}

def label_topic(inst_feat):
    for substr, label in topic_mapping.items():
        if substr.lower() in inst_feat.lower():
            return label
    return 'Other'

for dataset in [X_train, X_test]:
    dataset['podcast_topic'] = dataset['podcast_name'].apply(label_topic)

X_train = shift_col(X_train, 'podcast_name', 'podcast_topic')
X_test = shift_col(X_test, 'podcast_name', 'podcast_topic')


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

# -> Prepare alternative approach with imputation of median without imputation
#       dummy

for dataset in [X_train, X_test]:
    dataset['episode_length_imp_alt'] = dataset['episode_length_imp']
    dataset.loc[
        dataset['episode_length_imp_dum'] == 1, 'episode_length_imp_alt'
    ] = X_train['episode_length'].median()

X_train = shift_col(X_train, 'episode_length_imp', 'episode_length_imp_alt')
X_test = shift_col(X_test, 'episode_length_imp', 'episode_length_imp_alt')

# -> Include polynomial

for dataset in [X_train, X_test]:
    dataset['episode_length_imp_squ'] = dataset['episode_length_imp'] ** 2

X_train = shift_col(X_train, 'episode_length_imp_alt', 'episode_length_imp_squ')
X_test = shift_col(X_test, 'episode_length_imp_alt', 'episode_length_imp_squ')


#                                                                           #
# Genre

X_train['genre'].describe()
X_train['genre'].value_counts()

pd.crosstab(
    X_train['genre'], X_train['podcast_topic'], margins=True, normalize=False
)


#                                                                           #
# Host popularity

X_train['host_popularity'].describe()

# -> Add polynomial

for dataset in [X_train, X_test]:
    dataset['host_popularity_squ'] = dataset['host_popularity'] ** 2

X_train = shift_col(X_train, 'host_popularity', 'host_popularity_squ')
X_test = shift_col(X_test, 'host_popularity', 'host_popularity_squ')


#                                                                           #
# Guest popularity

X_train['guest_popularity'].describe()
X_train['guest_popularity_imp'].describe()
X_train['guest_popularity_imp_dum'].describe()

# -> Add polynomial

for dataset in [X_train, X_test]:
    dataset['guest_popularity_imp_squ'] = dataset['guest_popularity_imp'] ** 2

X_train = shift_col(X_train, 'guest_popularity_imp', 'guest_popularity_imp_squ')
X_test = shift_col(X_test, 'guest_popularity_imp', 'guest_popularity_imp_squ')

# -> Add interaction term with host popularity

for dataset in [X_train, X_test]:
    dataset['host_guest_popularity'] = (
        dataset['host_popularity'] * dataset['guest_popularity_imp']
    )

X_train = shift_col(
    X_train, 'guest_popularity_imp_dum', 'host_guest_popularity'
)
X_test = shift_col(
    X_test, 'guest_popularity_imp_dum', 'host_guest_popularity'
)


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

# -> Add weekend dummy feature for weekend

for dataset in [X_train, X_test]:
    dataset['publication_weekend'] = (
        dataset['publication_day_num'].isin([6, 7]).astype(int)
    )

X_train = shift_col(X_train, 'publication_day_num', 'publication_weekend')
X_test = shift_col(X_test, 'publication_day_num', 'publication_weekend')


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

# -> Add interaction term between publication day and publication time

for dataset in [X_train, X_test]:
    dataset['publication_day_time'] = (
        dataset['publication_day'] + " " + dataset['publication_time']
    )

X_train = shift_col(X_train, 'publication_time_num', 'publication_day_time')
X_test = shift_col(X_test, 'publication_time_num', 'publication_day_time')


#                                                                           #
# Number of ads

X_train['number_ads'].describe()

# -> Calculate ads per time

for dataset in [X_train, X_test]:
    dataset['ads_per_time'] = np.where(
        dataset['episode_length_imp'] == 0, 0,
        dataset['number_ads'] / dataset['episode_length_imp']
    )

X_train = shift_col(X_train, 'number_ads', 'ads_per_time')
X_test = shift_col(X_test, 'number_ads', 'ads_per_time')


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

# -> Add dummy feature for extreme sentiment

for dataset in [X_train, X_test]:
    dataset['extreme_sentiment'] = (
        dataset['episode_sentiment_num'].isin([1, 3]).astype(int)
    )

X_train = shift_col(X_train, 'episode_sentiment_num', 'extreme_sentiment')
X_test = shift_col(X_test, 'episode_sentiment_num', 'extreme_sentiment')


##                                                                          ##
## Save dataset

X_train.to_csv('data/02_01_X_train_eng.csv', index=False)

X_test.to_csv('data/02_02_X_test_eng.csv', index=False)
