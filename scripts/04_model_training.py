##############################################################################
##                                                                          ##
##  Project: Podcast Listening Time Prediction                              ##
##                                                                          ##
##  Script 04: Model training                                               ##
##                                                                          ##
##                                                                          ##
##  Author: Vitus Puttmann                                                  ##
##                                                                          ##
##  Version: 2.0                                                            ##
##                                                                          ##
##  Date: 23.04.2025                                                        ##
##                                                                          ##
##############################################################################


##                                                                          ##
## Import libraries

import time

import numpy as np
import pandas as pd

from sklearn.metrics import root_mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

import lightgbm as lgb


##                                                                          ##
## Define functions

def calc_runtime(func):
    """ Measure runtime of function. """

    def wrapper(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)    
        time_end = time.time()
        runtime = round((time_end - time_start), 3)
        return result, runtime
    return wrapper


def standardize_train(feat: str) -> pd.DataFrame:
    """ Standardize feature of the training dataset. """

    X_train_output = X_train.copy()
    X_train_output[feat] = (
        X_train_output[feat] - X_train_output[feat].mean()
    ) / X_train_output[feat].std()
    return X_train_output


def standardize_test(feat: str) -> pd.DataFrame:
    """ Standardize feature of the testing dataset. """

    X_test_output = X_test.copy()
    X_test_output[feat] = (
        X_test_output[feat] - X_train[feat].mean()
    ) / X_train[feat].std()
    return X_test_output


##                                                                          ##
## Load data

X_train = pd.read_csv('data/02_01_X_train_eng.csv')
y_train = pd.read_csv('data/51_01_y_train_prep.csv')

X_test = pd.read_csv('data/02_02_X_test_eng.csv')
y_test = pd.read_csv('data/51_02_y_test_prep.csv')


##                                                                          ##
## Prepare data

# -> Define type and inclusion of features via dictionary entry

features = {
    # feature name:             (type,          include)
    'id':                       ('continuous',  False),
    'podcast_name':             ('categorical', False),
    'episode_title':            ('categorical', False),
    'episode_title_num':        ('continuous',  False),
    'episode_length':           ('continuous',  False),
    'episode_length_imp':       ('continuous',  True),
    'episode_length_imp_dum':   ('categorical', True),
    'genre':                    ('categorical', False),
    'host_popularity':          ('continuous',  False),
    'guest_popularity':         ('continuous',  False),
    'guest_popularity_imp':     ('continuous',  False),
    'guest_popularity_imp_dum': ('categorical', False),
    'publication_day':          ('categorical', False),
    'publication_day_num':      ('continuous',  False),
    'publication_time':         ('categorical', False),
    'publication_time_num':     ('continuous',  False),
    'number_ads':               ('continuous',  True),
    'episode_sentiment':        ('categorical', True),
    'episode_sentiment_num':    ('continuous',  False)
}


#                                                                           #
# Drop features

features_drop = [
    feat for feat, (type, include) in features.items() if not include
]

for dataset in [X_train, X_test]:
    dataset.drop(features_drop, axis=1, inplace=True)


#                                                                           #
# Prepare features

# -> Standardize continuous features
# -> Standardize test dataset first to have correct reference values in the
#   train dataset

cont_feats = [
    feat for feat, (type, include) in features.items() if
        type == 'continuous' and include
]
 
for feat in cont_feats:
    X_test = standardize_test(feat)

for feat in cont_feats:
    X_train = standardize_train(feat)

# -> Encode string variables

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)


#                                                                           #
# Order features

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


#                                                                           #
# Adapt outcome format

y_train = y_train.to_numpy().ravel()


##                                                                          ##
## Fit linear regression

lin_reg = LinearRegression()
lin_reg.fit = calc_runtime(lin_reg.fit)

result, runtime = lin_reg.fit(X_train, y_train)
lin_reg_pred = lin_reg.predict(X_test)
rmse_lin_reg = root_mean_squared_error(y_test, lin_reg_pred)

print(f"RMSE: {round(rmse_lin_reg, 5)}\n"
      f"Runtime fitting: {round(runtime / 60, 1)} min.")


##                                                                          ##
## Fit decision tree

"""
parameters = {
    'criterion': ['squared_error'],
    'max_depth': [10],
    'min_samples_split': [2],
    'min_samples_leaf': [10],
    'max_leaf_nodes': [500],
    'min_impurity_decrease': [0.01],
    'ccp_alpha': [0.001]
}

grid = GridSearchCV(DecisionTreeRegressor(), parameters, cv=2)
grid.fit(X_train, y_train)
grid.best_params_
"""

dec_tree = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=10,
    min_samples_split=2,
    #min_samples_leaf=10,
    max_leaf_nodes=500,
    #min_impurity_decrease=0.01,
    ccp_alpha = 0.001,
    random_state=42)
dec_tree.fit = calc_runtime(dec_tree.fit)

result, runtime = dec_tree.fit(X_train, y_train)
dec_tree_pred = dec_tree.predict(X_test)
rmse_tree_pred = root_mean_squared_error(y_test, dec_tree_pred)

print(f"RMSE: {round(rmse_tree_pred, 5)}\n"
      f"Runtime fitting: {round(runtime / 60, 1)} min.")


##                                                                          ##
## Fit random forest

rand_forest = RandomForestRegressor(
    warm_start=True,
    n_jobs=-1,
    random_state=42
)
rand_forest.fit = calc_runtime(rand_forest.fit)

result, runtime = rand_forest.fit(X_train, y_train)
rand_forest_pred = rand_forest.predict(X_test)
rmse_rand_forest = root_mean_squared_error(y_test, rand_forest_pred)

print(f"RMSE: {round(rmse_rand_forest, 5)}\n"
      f"Runtime fitting: {round(runtime / 60, 1)} min.")
