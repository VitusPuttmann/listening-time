##############################################################################
##                                                                          ##
##  Project: Podcast Listening Time Prediction                              ##
##                                                                          ##
##  Script 02: Data exploration                                             ##
##                                                                          ##
##                                                                          ##
##  Author: Vitus Puttmann                                                  ##
##                                                                          ##
##  Version: 1.0                                                            ##
##                                                                          ##
##  Date: 12.04.2025                                                        ##
##                                                                          ##
##############################################################################


##                                                                          ##
## Import libraries

import itertools

import numpy as np
import pandas as pd

import seaborn as sns
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from scipy.stats import pearsonr
from scipy.stats import norm


##                                                                          ##
## Adapt formatting

plt.style.use('seaborn-v0_8-muted')


##                                                                          ##
## Read data

X_train = pd.read_csv('data/01_01_X_train_prep.csv')
y_train = pd.read_csv('data/51_01_y_train_prep.csv')


##                                                                          ##
## Explore features

#                                                                           #
# ID

len(X_train['id'].unique())
X_train['id'].describe()

# -> Unique identifier


#                                                                           #
# Podcast name

X_train['podcast_name'].unique()
len(X_train['podcast_name'].unique())

# -> 48 unique podcasts

for genre in X_train['genre'].unique():
    print(genre)
    print(len(X_train[X_train['genre'] == genre]['podcast_name'].unique()))
    print(X_train[X_train['genre'] == genre]['podcast_name'].unique())
for name in X_train['podcast_name'].unique():
    print(name)
    print(X_train[X_train['podcast_name'] == name]['genre'].unique())

# -> Different episodes of the same podcast assigned to different genres

X_train.groupby('podcast_name')[
    'episode_length'].agg('mean').sort_values(ascending=True)

X_train.groupby('podcast_name')[
    'host_popularity'].agg('mean').sort_values(ascending=True)

X_train.groupby('podcast_name')[
    'guest_popularity'].agg('mean').sort_values(ascending=True)

X_train.groupby('podcast_name')[
    'number_ads'].agg('mean').sort_values(ascending=True)

# -> Only limited variation in average episode length, host popularity, guest
#       popularity, and number of ads

for name in X_train['podcast_name'].unique():
    print(name)
    print(X_train[
        X_train['podcast_name'] == name
    ]['publication_day'].value_counts())

for name in X_train['podcast_name'].unique():
    print(name)
    print(X_train[
        X_train['podcast_name'] == name
    ]['publication_time'].value_counts())

for name in X_train['podcast_name'].unique():
    print(name)
    print(X_train[
        X_train['podcast_name'] == name
    ]['episode_sentiment'].value_counts())

# -> Rather similar distribution for weekdays, daytimes, and episode sentiment


#                                                                           #
# Episode title

X_train['episode_title'].unique()
sorted(X_train['episode_title'].unique())
len(X_train['episode_title'].unique())

X_train['episode_title'].value_counts()
X_train[
    X_train['podcast_name'] == 'Mystery Matters'
]['episode_title'].value_counts()
len(X_train[
    X_train['podcast_name'] == 'Mystery Matters'
]['episode_title'].value_counts())
X_train[
    X_train['podcast_name'] == 'Joke Junction'
]['episode_title'].value_counts()
len(X_train[
    X_train['podcast_name'] == 'Joke Junction'
]['episode_title'].value_counts())

# -> Same titles used by different podcasts
# -> Same titles used more than once by the same podcast


#                                                                           #
# Episode length

X_train['episode_length'].describe()
X_train['episode_length'].plot.density()
X_train['episode_length_imp'].plot.density()

# -> Not normally distributed

X_train.groupby('genre')['episode_length'].agg('mean')

X_train.groupby('publication_day')['episode_length'].agg('mean')

X_train.groupby('publication_time')['episode_length'].agg('mean')

# -> Hardly any variation for genres, publication days, and publication times

X_train.groupby('number_ads')['episode_length'].agg('mean')
X_train['episode_length'].corr(X_train['number_ads'])

# -> Weak negative relationship with number of ads


#                                                                           #
# Genre

X_train['genre'].unique()
len(X_train['genre'].unique())
X_train['genre'].value_counts()

# -> Episodes not equally distributed over genres

X_train.groupby('genre')['publication_day'].value_counts(normalize=True)

X_train.groupby('genre')['publication_time'].value_counts(normalize=True)

X_train.groupby('genre')['episode_sentiment'].value_counts(normalize=True)

# -> Similar distribution of publication day, publication time and episode
#       sentiments


#                                                                           #
# Host popularity

X_train['host_popularity'].describe()

X_train['host_popularity'].corr(X_train['guest_popularity'])

# -> No obvious association with guest popularity

X_train.groupby('podcast_name')['host_popularity'].mean()

# -> Differences between podcasts small

X_train.groupby('genre')['host_popularity'].mean()

X_train.groupby('publication_day')['host_popularity'].mean()

X_train.groupby('publication_time')['host_popularity'].mean()

# -> No obvious associations with genre, publication day, and publication time


#                                                                           #
# Guest popularity (percentage)

X_train['guest_popularity'].describe()

X_train.groupby('podcast_name')['guest_popularity'].mean()

X_train.groupby('genre')['guest_popularity'].mean()

X_train.groupby('publication_day')['guest_popularity'].mean()

X_train.groupby('publication_time')['guest_popularity'].mean()

# -> No obvious associations with podcast, genre, publication day, and
#       publication time


#                                                                           #
# Publication timing

X_train['publication_day'].unique()
len(X_train['publication_day'].unique())

X_train['publication_time'].unique()
len(X_train['publication_time'].unique())

X_train.groupby('publication_day')[
    'publication_time'
].value_counts(normalize=True)

# -> No obvious systematic association between publication day and time


#                                                                           #
# Number of ads

X_train['number_ads'].describe()

X_train.groupby('podcast_name')['number_ads'].agg('mean')

X_train.groupby('genre')['number_ads'].agg('mean')

X_train.groupby('publication_day')['number_ads'].agg('mean')

X_train.groupby('publication_time')['number_ads'].agg('mean')

X_train.groupby('episode_sentiment')['number_ads'].agg('mean')

# -> No obvious systematic association with podcasts, genre, publication day.
#       publication time, and episode sentiment


#                                                                           #
# Episode sentiment

X_train['episode_sentiment'].unique()
len(X_train['episode_sentiment'].unique())

X_train.groupby(
    'podcast_name'
)['episode_sentiment'].value_counts(normalize=True)

X_train.groupby(
    'genre'
)['episode_sentiment'].value_counts(normalize=True)

X_train.groupby(
    'publication_day'
)['episode_sentiment'].value_counts(normalize=True)

X_train.groupby(
    'publication_time'
)['episode_sentiment'].value_counts(normalize=True)

# -> No obvious systematic association with podcast, genre, publication day,
#       and publication time


##                                                                          ##
## Provide overview on associations

#                                                                           #
# Categorical features

cat_feats = [
    'podcast_name',
    'episode_title',
    'episode_length_imp_dum',
    'genre',
    'publication_day',
    'publication_time',
    'guest_popularity_imp_dum',
    'episode_sentiment'
]

results = []
for feat_1, feat_2 in itertools.combinations(cat_feats, 2):
    ct = pd.crosstab(X_train[feat_1], X_train[feat_2])
    _, p, _, _ = chi2_contingency(ct)
    results.append({
        'p_value': round(p, 3),
        'feat_1': feat_1,
        'feat_2': feat_2,
    })
cat_results_df = pd.DataFrame(results)

print(cat_results_df)


#                                                                           #
# Continuous features

cont_feats = [
    'id',
    'episode_length',
    'episode_length_imp',
    'host_popularity',
    'guest_popularity',
    'guest_popularity_imp',
    'number_ads'
]

results = []
for feat_1, feat_2 in itertools.combinations(cont_feats, 2):
    data_clean = X_train[[feat_1, feat_2]].dropna()
    r, p = pearsonr(data_clean[feat_1], data_clean[feat_2])
    results.append({
        'pearson_r': round(r, 3),
        'p_value': round(p, 3),
        'feat_1': feat_1,
        'feat_2': feat_2
    })
cont_results_df = pd.DataFrame(results)

print(cont_results_df)


##                                                                          ##
## Explore outcome and association with features

#                                                                           #
# Combine datasets

train = pd.concat([X_train, y_train], axis=1)


#                                                                           #
# Listening time

train['listening_time'].describe()

train['listening_time'].plot.density()
plt.show()

train['listening_time'].plot.box()
plt.show()


#                                                                           #
# ID

train['listening_time'].corr(train['id'])

plt.scatter(train['listening_time'], train['id'], s=1, alpha=0.3)
plt.show()

# -> No association


#                                                                           #
# Podcast name

train.groupby('podcast_name')['listening_time'].agg('mean')
sorted(train.groupby('podcast_name')['listening_time'].agg('mean'))

time_podcast = train.groupby('podcast_name')['listening_time'].agg('mean')
time_podcast.plot.density()

# -> No obvious association


#                                                                           #
# Episode title

train.groupby('episode_title')['listening_time'].agg('mean')
sorted(train.groupby('episode_title')['listening_time'].agg('mean'))

time_episode = train.groupby('episode_title')['listening_time'].agg('mean')
time_episode.plot.density()

# -> No obvious association


#                                                                           #
# Episode length

train['listening_time'].corr(train['episode_length'])
train['listening_time'].corr(train['episode_length_imp'])

sns.regplot(
    x='episode_length',
    y='listening_time',
    data=train,
    scatter_kws={'alpha': 0.6, 's': 5},
    line_kws={'color': 'black'}
)

# -> Strong positive correlation

train.groupby('episode_length_imp_dum')['listening_time'].agg('mean')

train[train['episode_length_imp_dum'] == 0]['listening_time'].plot.density()
train[train['episode_length_imp_dum'] == 1]['listening_time'].plot.density()

# -> Rather similar distributions


#                                                                           #
# Genre

train.groupby('genre')['listening_time'].describe()

# -> No obvious association


#                                                                           #
# Host popularity

train['host_popularity'].corr(train['listening_time'])

sns.regplot(
    x='host_popularity',
    y='listening_time',
    data=train,
    scatter_kws={'alpha': 0.6, 's': 5},
    line_kws={'color': 'black'}
)

# -> No obvious association


#                                                                           #
# Publication day

weekday_map = {
    'Monday': '1 Monday',
    'Tuesday': '2 Tuesday',
    'Wednesday': '3 Wednesday',
    'Thursday': '4 Thursday',
    'Friday': '5 Friday',
    'Saturday': '6 Saturday',
    'Sunday': '7 Sunday'
}
train['publication_day_ord'] = train['publication_day'].map(weekday_map)

train.groupby('publication_day_ord')['listening_time'].agg('mean')

train.boxplot(column='listening_time', by='publication_day_ord')

# -> No obvious association


#                                                                           #
# Publication time

time_map = {
    'Morning': '1 Morning',
    'Afternoon': '2 Afternoon',
    'Evening': '3 Evening',
    'Night': '4 Night'
}
train['publication_time_ord'] = train['publication_time'].map(time_map)

train.groupby('publication_time_ord')['listening_time'].agg('mean')

train.boxplot(column='listening_time', by='publication_time_ord')

# -> No obvious association


#                                                                           #
# Guest popularity

train['listening_time'].corr(train['guest_popularity'])
train['listening_time'].corr(train['guest_popularity_imp'])

sns.regplot(
    x='guest_popularity',
    y='listening_time',
    data=train,
    scatter_kws={'alpha': 0.6, 's': 5},
    line_kws={'color': 'black'}
)

# -> No obvious association

train.groupby('guest_popularity_imp_dum')['listening_time'].agg('mean')

train[train['guest_popularity_imp_dum'] == 0]['listening_time'].plot.density()
train[train['guest_popularity_imp_dum'] == 1]['listening_time'].plot.density()

# -> Rather similar distributions


#                                                                           #
# Number of ads

train['listening_time'].corr(train['number_ads'])

train.groupby('number_ads')['listening_time'].agg('mean')

train.boxplot(column='listening_time', by='number_ads')

smf.ols(
    'listening_time ~ number_ads + episode_length', data=train
).fit().summary()

# -> Negative association (also when controlling for episode length)


#                                                                           #
# Episode sentiment

sentiment_map = {
    'Negative': '1 Negative',
    'Neutral': '2 Neutral',
    'Positive': '3 Positive'
}
train['episode_sentiment_ord'] = train['episode_sentiment'].map(sentiment_map)

train.groupby('episode_sentiment_ord')['listening_time'].agg('mean')

train.boxplot(column='listening_time', by='episode_sentiment_ord')

# -> Weak positive association
