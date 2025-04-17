##############################################################################
##                                                                          ##
##  Project: Podcast Listening Time Prediction                              ##
##                                                                          ##
##  Script 01: Data preparation                                             ##
##                                                                          ##
##                                                                          ##
##  Author: Vitus Puttmann                                                  ##
##                                                                          ##
##  Version: 1.1                                                            ##
##                                                                          ##
##  Date: 16.04.2025                                                        ##
##                                                                          ##
##############################################################################


##                                                                          ##
## Import libraries

import pandas as pd

from sklearn.model_selection import train_test_split


##                                                                          ##
## Define functions

def shift_col(dataset: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    """ Take a dataframe and shift the column 'right' next to the column
        'left'. """

    cols = dataset.columns.tolist()
    cols.remove(right)
    cols.insert(cols.index(left) + 1, right)
    dataset = dataset[cols]
    return dataset


##                                                                          ##
## Import and split data

podcast = pd.read_csv('original/train.csv')

podcast.info()

X = podcast.drop('Listening_Time_minutes', axis=1)
y = podcast['Listening_Time_minutes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


##                                                                          ##
## Prepare data

X_train.info()
X_test.info()

# ID                                                                        #

X_train['id'].unique()
len(X_train['id'].unique())
X_train['id'].describe()
X_train['id'].dtype
X_train['id'].isna().sum()

# -> No issues


# Podcast name                                                              #

X_train['Podcast_Name'].unique()
len(X_train['Podcast_Name'].unique())
X_train['Podcast_Name'].dtype
X_train['Podcast_Name'].isna().sum()
X_train['Podcast_Name'].value_counts()

# -> No issues

for dataset in [X_train, X_test]:
    dataset.rename(
        columns={'Podcast_Name': 'podcast_name'},
        inplace=True
    )


# Episode title                                                             #

X_train['Episode_Title'].unique()
len(X_train['Episode_Title'].unique())
sorted(X_train['Episode_Title'].unique())
X_train['Episode_Title'].dtype
X_train['Episode_Title'].isna().sum()
X_train['Episode_Title'].value_counts()

# -> No issues

for dataset in [X_train, X_test]:
    dataset.rename(
        columns={'Episode_Title': 'episode_title'},
        inplace=True
    )


# Episode length (minutes)                                                  #

X_train['Episode_Length_minutes'].unique()
X_train['Episode_Length_minutes'].dtype
X_train['Episode_Length_minutes'].describe()
X_train['Episode_Length_minutes'].isna().sum()

# -> High number of missing values
#   -> Deal with below

for dataset in [X_train, X_test]:
    dataset.rename(
        columns={'Episode_Length_minutes': 'episode_length'},
        inplace=True
    )


# Genre                                                                     #

X_train['Genre'].unique()
len(X_train['Genre'].unique())
X_train['Genre'].dtype
X_train['Genre'].isna().sum()
X_train['Genre'].value_counts()

# -> No issues

for dataset in [X_train, X_test]:
    dataset.rename(
        columns={'Genre': 'genre'},
        inplace=True
    )


# Host popularity (percentage)                                              #

X_train['Host_Popularity_percentage'].unique()
X_train['Host_Popularity_percentage'].dtype
X_train['Host_Popularity_percentage'].describe()
X_train['Host_Popularity_percentage'].isna().sum()

# -> No issues

for dataset in [X_train, X_test]:
    dataset.rename(
        columns={'Host_Popularity_percentage': 'host_popularity'},
        inplace=True
    )


# Publication day                                                           #

X_train['Publication_Day'].unique()
len(X_train['Publication_Day'].unique())
X_train['Publication_Day'].dtype
X_train['Publication_Day'].isna().sum()

# -> No issues

for dataset in [X_train, X_test]:
    dataset.rename(
        columns={'Publication_Day': 'publication_day'},
        inplace=True
    )


# Publication time                                                          #

X_train['Publication_Time'].unique()
len(X_train['Publication_Time'].unique())
X_train['Publication_Time'].dtype
X_train['Publication_Time'].isna().sum()

# -> No issues

for dataset in [X_train, X_test]:
    dataset.rename(
        columns={'Publication_Time': 'publication_time'},
        inplace=True
    )


# Guest popularity (percentage)                                             #

X_train['Guest_Popularity_percentage'].unique()
len(X_train['Guest_Popularity_percentage'].unique())
X_train['Guest_Popularity_percentage'].dtype
X_train['Guest_Popularity_percentage'].describe()
X_train['Guest_Popularity_percentage'].isna().sum()

# -> Large number of missing values

for dataset in [X_train, X_test]:
    dataset.rename(
        columns={'Guest_Popularity_percentage': 'guest_popularity'},
        inplace=True
    )


# Number of ads                                                             #

X_train['Number_of_Ads'].unique()
len(X_train['Number_of_Ads'].unique())
X_train['Number_of_Ads'].describe()
X_train['Number_of_Ads'].value_counts()
X_train['Number_of_Ads'].isna().sum()

# -> One missing value
# -> A few extreme values

for dataset in [X_train, X_test]:
    dataset.rename(
        columns={'Number_of_Ads': 'number_ads'},
        inplace=True
    )


# Episode sentiment                                                         #

X_train['Episode_Sentiment'].unique()
len(X_train['Episode_Sentiment'].unique())
X_train['Episode_Sentiment'].dtype
X_train['Episode_Sentiment'].isna().sum()

# -> No issues

for dataset in [X_train, X_test]:
    dataset.rename(
        columns={'Episode_Sentiment': 'episode_sentiment'},
        inplace=True
    )


# Listening time (minutes)                                                  #

y_train.unique()
len(y_train.unique())
y_train.dtype
y_train.describe()
y_train.isna().sum()

# -> No issues

y_test.unique()
len(y_test.unique())
y_test.dtype
y_test.describe()
y_test.isna().sum()

# -> No issues


##                                                                          ##
## Clean data

X_train.info()


# Episode length

# -> Handle missing values

X_train['episode_length'].isna().sum()

X_train[X_train['episode_length'].isna()]['id'].describe()
X_train[X_train['episode_length'].isna()]['podcast_name'].value_counts()
X_train[X_train['episode_length'].isna()]['episode_title'].value_counts()
X_train[X_train['episode_length'].isna()]['genre'].value_counts()
X_train[X_train['episode_length'].isna()]['host_popularity'].describe()
X_train[X_train['episode_length'].isna()]['publication_day'].value_counts()
X_train[X_train['episode_length'].isna()]['publication_time'].value_counts()
X_train[X_train['episode_length'].isna()]['number_ads'].describe()
X_train[X_train['episode_length'].isna()]['episode_sentiment'].value_counts()

# -> No obvious systematic associations between missings and other features
# -> Issue that episode length directly influences listening time
# -> Dropping instances with missings not optimal due to missings in the
#       testing datasetdata
#   -> Impute missing values with 0 and add imputation dummy

for dataset in [X_train, X_test]:
    dataset['episode_length_imp'] = dataset['episode_length']
    dataset.loc[
        dataset['episode_length'].isna(), 'episode_length_imp'
    ] = 0
    dataset['episode_length_imp_dum'] = 0
    dataset.loc[dataset['episode_length'].isna(), 'episode_length_imp_dum'] = 1
X_train = shift_col(X_train, 'episode_length', 'episode_length_imp')    
X_train = shift_col(X_train, 'episode_length_imp', 'episode_length_imp_dum')    
X_test = shift_col(X_test, 'episode_length', 'episode_length_imp')    
X_test = shift_col(X_test, 'episode_length_imp', 'episode_length_imp_dum')    

X_train['episode_length_imp'].isna().sum()
X_train['episode_length_imp_dum'].describe()

X_test['episode_length_imp'].isna().sum()
X_test['episode_length_imp_dum'].describe()


# Guest popularity

# -> Handle missing values

X_train['guest_popularity'].isna().sum()

X_train[X_train['guest_popularity'].isna()]['id'].describe()
X_train[X_train['guest_popularity'].isna()]['podcast_name'].value_counts()
X_train[X_train['guest_popularity'].isna()]['episode_title'].value_counts()
X_train[X_train['guest_popularity'].isna()]['episode_length'].describe()
X_train[X_train['guest_popularity'].isna()]['genre'].value_counts()
X_train[X_train['guest_popularity'].isna()]['host_popularity'].describe()
X_train[X_train['guest_popularity'].isna()]['publication_day'].value_counts()
X_train[X_train['guest_popularity'].isna()]['publication_time'].value_counts()
X_train[X_train['guest_popularity'].isna()]['number_ads'].describe()
X_train[X_train['guest_popularity'].isna()]['episode_sentiment'].value_counts()

# -> No obvious systematic associations between missings and other features
# -> Presumably no guest in episode
#   -> Impute missing values with 0 and add imputation dummy

for dataset in [X_train, X_test]:
        dataset['guest_popularity_imp'] = dataset['guest_popularity']
        dataset.loc[
            dataset['guest_popularity'].isna(),
            'guest_popularity_imp'
        ] = 0
        dataset['guest_popularity_imp_dum'] = 0
        dataset.loc[
            dataset['guest_popularity'].isna(), 'guest_popularity_imp_dum'
        ] = 1

X_train = shift_col(X_train, 'guest_popularity', 'guest_popularity_imp')
X_train = shift_col(X_train, 'guest_popularity_imp', 'guest_popularity_imp_dum')
X_test = shift_col(X_test, 'guest_popularity', 'guest_popularity_imp')
X_test = shift_col(X_test, 'guest_popularity_imp', 'guest_popularity_imp_dum')

X_train['guest_popularity_imp'].isna().sum()
X_train['guest_popularity_imp_dum'].describe()
X_test['guest_popularity_imp'].isna().sum()
X_test['guest_popularity_imp_dum'].describe()


# Number ads

# -> Handle missing values

X_train['number_ads'].isna().sum()
X_train[X_train['number_ads'].isna()].head(1)

# -> Only one missing value
#   -> Impute with median

ads_median = X_train['number_ads'].median()

for dataset in [X_train, X_test]:
    dataset.loc[dataset['number_ads'].isna(), 'number_ads'] = ads_median

X_train['number_ads'].isna().sum()
X_test['number_ads'].isna().sum()

# -> Handle extreme values

X_train['number_ads'].value_counts()

X_train[X_train['number_ads'] > 3]['id'].describe()
X_train[X_train['number_ads'] > 3]['podcast_name'].value_counts()
X_train[X_train['number_ads'] > 3]['episode_title'].value_counts()
X_train[X_train['number_ads'] > 3]['episode_length'].describe()
X_train[X_train['number_ads'] > 3]['genre'].value_counts()
X_train[X_train['number_ads'] > 3]['host_popularity'].describe()
X_train[X_train['number_ads'] > 3]['publication_day'].value_counts()
X_train[X_train['number_ads'] > 3]['publication_time'].value_counts()
X_train[X_train['number_ads'] > 3]['guest_popularity'].describe()
X_train[X_train['number_ads'] > 3]['episode_sentiment'].value_counts()

# -> Only very few (4) extreme values
# -> No obvious systematic associations with other features
#   -> Set values to maximum of observations with non-extreme values (3)

for dataset in [X_train, X_test]:
    dataset.loc[dataset['number_ads'] > 3, 'number_ads'] = 3

X_train['number_ads'].describe()
X_test['number_ads'].describe()


##                                                                          ##
## Save dataset

X_train.to_csv('data/01_01_X_train_prep.csv', index=False)
y_train.to_csv('data/99_01_y_train_prep.csv', index=False)

X_test.to_csv('data/01_02_X_test_prep.csv', index=False)
y_test.to_csv('data/99_02_y_test_prep.csv', index=False)
