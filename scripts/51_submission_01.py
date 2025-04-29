##############################################################################
##                                                                          ##
##  Project: Podcast Listening Time Prediction                              ##
##                                                                          ##
##  Script 51: Submission 01                                                ##
##                                                                          ##
##                                                                          ##
##  Author: Vitus Puttmann                                                  ##
##                                                                          ##
##  Version: 1.0                                                            ##
##                                                                          ##
##  Date: 24.04.2025                                                        ##
##                                                                          ##
##############################################################################


##                                                                          ##
## Import libraries

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor


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

# -> Transform episode title to numeric feature

for dataset in [X_test, X_train]:
    dataset['episode_title_num'] = (
        dataset['episode_title'].str.replace('Episode ', '')
    )
    dataset['episode_title_num'] = dataset['episode_title_num'].astype(int)


##                                                                          ##
## Prepare data

X_test.info()


#                                                                           #
# Define approach

# -> Define type and inclusion of features via dictionary entry

features = {
    # feature name:             (type,          include)
    'id':                       ('id',          False),
    'podcast_name':             ('categorical', True),
    'episode_title':            ('categorical', False),
    'episode_title_num':        ('continuous',  True),
    'episode_length':           ('continuous',  False),
    'episode_length_imp':       ('continuous',  True),
    'episode_length_imp_dum':   ('categorical', True),
    'genre':                    ('categorical', True),
    'host_popularity':          ('continuous',  True),
    'guest_popularity':         ('continuous',  False),
    'guest_popularity_imp':     ('continuous',  True),
    'guest_popularity_imp_dum': ('categorical', True),
    'publication_day':          ('categorical', True),
    'publication_time':         ('categorical', True),
    'number_ads':               ('continuous',  True),
    'episode_sentiment':        ('categorical', True)
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


##                                                                         ##
## Fit random forest

rand_forest = RandomForestRegressor(
    warm_start=True,
    n_jobs=-1,
    random_state=42
)

rand_forest.fit(X_train, y_train)

rand_forest_pred = rand_forest.predict(X_test)

submission_01 = pd.DataFrame({
     'id': submission,
     'Listening_Time_minutes': rand_forest_pred
})

submission_01.to_csv('output/submission_01.csv', index=False)
