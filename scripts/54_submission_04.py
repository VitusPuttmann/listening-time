##############################################################################
##                                                                          ##
##  Project: Podcast Listening Time Prediction                              ##
##                                                                          ##
##  Script 54: Submission 04                                                ##
##                                                                          ##
##                                                                          ##
##  Author: Vitus Puttmann                                                  ##
##                                                                          ##
##  Version: 1.0                                                            ##
##                                                                          ##
##  Date: 30.04.2025                                                        ##
##                                                                          ##
##############################################################################


##                                                                          ##
## Import libraries

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


##                                                                          ##
## Load data

X_test = pd.read_csv('original/test.csv')
submission = X_test['id']

train = pd.read_csv('original/train.csv')
X_train = train.drop('Listening_Time_minutes', axis=1)
y_train = train['Listening_Time_minutes']


##                                                                          ##
## Prepare data

X_test.info()

X_train.info()
y_train.info()


#                                                                           #
# Rename variables

for dataset in [X_test, X_train]:
    dataset.rename(
        columns={
            'Podcast_Name': 'podcast_name',
            'Episode_Title': 'episode_title',
            'Episode_Length_minutes': 'episode_length',
            'Genre': 'genre',
            'Host_Popularity_percentage': 'host_popularity',
            'Publication_Day': 'publication_day',
            'Publication_Time': 'publication_time',
            'Guest_Popularity_percentage': 'guest_popularity',
            'Number_of_Ads': 'number_ads',
            'Episode_Sentiment': 'episode_sentiment',
        },
        inplace=True
    )


#                                                                           #
# Adapt outcome

y_train = y_train.to_numpy().ravel()


#                                                                           #
# Clean data

# -> Impute missing values of episode length with 0
# -> Create imputation dummy

for dataset in [X_test, X_train]:
    dataset['episode_length_imp'] = dataset['episode_length']
    dataset.loc[
        dataset['episode_length'].isna(), 'episode_length_imp'
    ] = 0
    dataset['episode_length_imp_dum'] = 0
    dataset.loc[dataset['episode_length'].isna(), 'episode_length_imp_dum'] = 1

# -> Impute missing values of guest popularity with 0
# -> Create imputation dummy

for dataset in [X_test, X_train]:
        dataset['guest_popularity_imp'] = dataset['guest_popularity']
        dataset.loc[
            dataset['guest_popularity'].isna(),
            'guest_popularity_imp'
        ] = 0
        dataset['guest_popularity_imp_dum'] = 0
        dataset.loc[
            dataset['guest_popularity'].isna(), 'guest_popularity_imp_dum'
        ] = 1

# -> Impute missing values of episode length with median of training data
# -> Create imputation dummy

ads_median = X_train['number_ads'].median()

for dataset in [X_test, X_train]:
    dataset.loc[dataset['number_ads'].isna(), 'number_ads'] = ads_median

# -> Replace extreme values for number of ads with maximum of observations with
#       non-extreme values in the training data (3)

for dataset in [X_test, X_train]:
    dataset.loc[dataset['number_ads'] > 3, 'number_ads'] = 3


#                                                                           #
# Engineer features

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

# -> Include polynomial

for dataset in [X_train, X_test]:
    dataset['episode_length_imp_squ'] = dataset['episode_length_imp'] ** 2


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


#                                                                           #
# Guest popularity

X_train['guest_popularity'].describe()
X_train['guest_popularity_imp'].describe()
X_train['guest_popularity_imp_dum'].describe()

# -> Add polynomial

for dataset in [X_train, X_test]:
    dataset['guest_popularity_imp_squ'] = dataset['guest_popularity_imp'] ** 2

# -> Add interaction term with host popularity

for dataset in [X_train, X_test]:
    dataset['host_guest_popularity'] = (
        dataset['host_popularity'] * dataset['guest_popularity_imp']
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

X_test['publication_day_num'] = X_test['publication_day'].map(weekday_mapping)

# -> Add weekend dummy feature for weekend

for dataset in [X_train, X_test]:
    dataset['publication_weekend'] = (
        dataset['publication_day_num'].isin([6, 7]).astype(int)
    )


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

X_test['publication_time_num'] = (
    X_test['publication_time'].map(daytime_mapping)
)

# -> Add interaction term between publication day and publication time

for dataset in [X_train, X_test]:
    dataset['publication_day_time'] = (
        dataset['publication_day'] + " " + dataset['publication_time']
    )


#                                                                           #
# Number of ads

X_train['number_ads'].describe()

# -> Calculate ads per time

for dataset in [X_train, X_test]:
    dataset['ads_per_time'] = np.where(
        dataset['episode_length_imp'] == 0, 0,
        dataset['number_ads'] / dataset['episode_length_imp']
    )


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

X_test['episode_sentiment_num'] = (
    X_test['episode_sentiment'].map(sentiment_mapping)
)

# -> Add dummy feature for extreme sentiment

for dataset in [X_train, X_test]:
    dataset['extreme_sentiment'] = (
        dataset['episode_sentiment_num'].isin([1, 3]).astype(int)
    )


##                                                                          ##
## Prepare data

X_test.info()


#                                                                           #
# Define approach

# -> Define type and inclusion of features via dictionary entry

features = {
    # feature name:             (type,          include)
    'id':                       ('continuous',  False),
    'podcast_name':             ('categorical', True),
    'podcast_topic':            ('categorical', False),
    'episode_title':            ('categorical', False),
    'episode_title_num':        ('continuous',  True),
    'episode_length':           ('continuous',  False),
    'episode_length_imp':       ('continuous',  True),
    'episode_length_imp_squ':   ('continuous',  True),
    'episode_length_imp_alt':   ('continuous',  False),
    'episode_length_imp_dum':   ('categorical', True),
    'genre':                    ('categorical', True),
    'host_popularity':          ('continuous',  True),
    'host_popularity_squ':      ('continuous',  True),
    'guest_popularity':         ('continuous',  False),
    'guest_popularity_imp':     ('continuous',  True),
    'guest_popularity_imp_squ': ('continuous',  True),
    'guest_popularity_imp_dum': ('categorical', True),
    'host_guest_popularity':    ('continuous',  False),
    'publication_day':          ('categorical', True),
    'publication_day_num':      ('continuous',  False),
    'publication_weekend':      ('categorical', False),
    'publication_time':         ('categorical', True),
    'publication_time_num':     ('continuous',  False),
    'publication_day_time':     ('categorical', False),
    'number_ads':               ('continuous',  True),
    'ads_per_time':             ('continuous',  True),
    'episode_sentiment':        ('categorical', True),
    'episode_sentiment_num':    ('continuous',  False),
    'extreme_sentiment':        ('categorical', False),
}

# -> Define whether continuous features should be standardized via variable

standardization = True


#                                                                           #
# Drop features

features_drop = [
    feat for feat, (feat_type, include) in features.items() if not include
]

for dataset in [X_train, X_test]:
    dataset.drop(features_drop, axis=1, inplace=True)


#                                                                           #
# Prepare features

# -> Standardize continuous features if chosen

if standardization:
    
    cont_feats = [
        feat for feat, (feat_type, include) in features.items() if
        feat_type == 'continuous' and include
    ]   

    scaler = StandardScaler().fit(X_train[cont_feats])

    X_train[cont_feats] = scaler.transform(X_train[cont_feats])

    X_test[cont_feats] = scaler.transform(X_test[cont_feats])

# -> Encode string variables

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)


#                                                                           #
# Order features

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


##                                                                          ##
## Fit XGBRegressor



xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

submission_04 = pd.DataFrame({
     'id': submission,
     'Listening_Time_minutes': xgb_pred
})

submission_04.to_csv('output/submission_04.csv', index=False)
