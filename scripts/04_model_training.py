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
##  Date: 25.04.2025                                                        ##
##                                                                          ##
##############################################################################


##                                                                          ##
## Import libraries

import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

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

from xgboost import XGBRegressor

from sklearn.ensemble import StackingRegressor

import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.inspection import permutation_importance

from sklearn.metrics import root_mean_squared_error


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


##                                                                          ##
## Load data

X_train = pd.read_csv('data/02_01_X_train_eng.csv')
y_train = pd.read_csv('data/51_01_y_train_prep.csv')

X_test = pd.read_csv('data/02_02_X_test_eng.csv')
y_test = pd.read_csv('data/51_02_y_test_prep.csv')


##                                                                          ##
## Prepare data

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


#                                                                           #
# Adapt outcome format

y_train = y_train.to_numpy().ravel()


##                                                                          ##
## Fit linear regression

"""

lin_reg = LinearRegression()
lin_reg.fit = calc_runtime(lin_reg.fit)

result, runtime = lin_reg.fit(X_train, y_train)
lin_reg_pred = lin_reg.predict(X_test)
rmse_lin_reg = root_mean_squared_error(y_test, lin_reg_pred)

print(
    f"Linear regression\n"
    f"RMSE: {round(rmse_lin_reg, 5)}\n"
    f"Runtime fitting: {round(runtime / 60, 1)} min."
)

"""


##                                                                          ##
## Fit Ridge regression

"""

parameters = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['auto', 'lsqr', 'sag', 'lbfgs'],
    'tol': [0.01, 0.001]
}

grid = GridSearchCV(Ridge(), parameters, cv=2)
grid.fit(X_train, y_train)
grid.best_params_

"""

"""

ridg_reg = Ridge(
    alpha=0.01,
    positive=False,
    solver='sag',
    tol=0.001
)
ridg_reg.fit = calc_runtime(ridg_reg.fit)

result, runtime = ridg_reg.fit(X_train, y_train)
ridg_reg_pred = ridg_reg.predict(X_test)
rmse_ridg_reg = root_mean_squared_error(y_test, ridg_reg_pred)

print(
    f"Ridge regression\n"
    f"RMSE: {round(rmse_ridg_reg, 5)}\n"
    f"Runtime fitting: {round(runtime / 60, 1)} min.")

"""


##                                                                          ##
## Fit Lasso regression

"""

parameters = {
    'alpha': [0.001, 0.01, 0.1, 1, 10],
    'tol': [0.01, 0.001],
    'selection': ['cyclic', 'random']
}

grid = GridSearchCV(Lasso(), parameters, cv=2)
grid.fit(X_train, y_train)
grid.best_params_

"""

"""

lass_reg = Lasso(
    alpha=0.001,
    selection='cyclic',
    tol=0.001)
lass_reg.fit = calc_runtime(lass_reg.fit)

result, runtime = lass_reg.fit(X_train, y_train)
lass_reg_pred = lass_reg.predict(X_test)
rmse_lass_reg = root_mean_squared_error(y_test, lass_reg_pred)

print(
    f"Lasso regression\n"
    f"RMSE: {round(rmse_lass_reg, 5)}\n"
    f"Runtime fitting: {round(runtime / 60, 1)} min."
)

"""


##                                                                          ##
## Fit Elastic Net

"""

elnet_reg = ElasticNet()
elnet_reg.fit = calc_runtime(elnet_reg.fit)

result, runtime = elnet_reg.fit(X_train, y_train)
elnet_reg_pred = elnet_reg.predict(X_test)
rmse_elnet_reg = root_mean_squared_error(y_test, elnet_reg_pred)

print(
    f"Elastic net\n"
    f"RMSE: {round(rmse_elnet_reg, 5)}\n"
    f"Runtime fitting: {round(runtime / 60, 1)} min."
)

"""


##                                                                          ##
## Fit Bayesian Ridge regression

"""

bayrid_reg = BayesianRidge()
bayrid_reg.fit = calc_runtime(bayrid_reg.fit)

result, runtime = bayrid_reg.fit(X_train, y_train)
bayrid_reg_pred = bayrid_reg.predict(X_test)
rmse_bayrid_reg = root_mean_squared_error(y_test, bayrid_reg_pred)

print(
    f"Bayesian Ridge regression\n"
    f"RMSE: {round(rmse_bayrid_reg, 5)}\n"
    f"Runtime fitting: {round(runtime / 60, 1)} min."
)

"""


##                                                                          ##
## Fit decision tree

"""

parameters = {
    'max_depth': [3, 10, 20],
    'min_samples_split': [2, 10, 50],
    'min_samples_leaf': [1, 5, 20],
    'ccp_alpha': [0.1, 0.01]
}

grid = GridSearchCV(DecisionTreeRegressor(), parameters, cv=2)
grid.fit(X_train, y_train)
grid.best_params_

"""

"""

dec_tree = DecisionTreeRegressor(
    criterion='squared_error',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=20,
    ccp_alpha = 0.01,
    random_state=42
)
dec_tree.fit = calc_runtime(dec_tree.fit)

result, runtime = dec_tree.fit(X_train, y_train)
dec_tree_pred = dec_tree.predict(X_test)
rmse_tree_pred = root_mean_squared_error(y_test, dec_tree_pred)

print(
    f"Decision tree\n"
    f"RMSE: {round(rmse_tree_pred, 5)}\n"
    f"Runtime fitting: {round(runtime / 60, 1)} min."
)

"""

##                                                                          ##
## Fit random forest

"""

rand_forest = RandomForestRegressor(
    n_estimators=200,
    max_depth=40,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.3,
    warm_start=True,
    n_jobs=1,
    random_state=42
)
rand_forest.fit = calc_runtime(rand_forest.fit)

result, runtime = rand_forest.fit(X_train, y_train)
rand_forest_pred = rand_forest.predict(X_test)
rmse_rand_forest = root_mean_squared_error(y_test, rand_forest_pred)

print(
    "Random forest\n"
    f"RMSE: {round(rmse_rand_forest, 5)}\n"
    f"Runtime fitting: {round(runtime / 60, 1)} min."
)

"""


##                                                                          ##
## Fit gradient boosting machine

"""

gb_mach = GradientBoostingRegressor(
    n_estimators=200,       # 100 # 500
    learning_rate=0.2,
    max_depth=9,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=1.0,
    max_features=0.5
)
gb_mach.fit = calc_runtime(gb_mach.fit)

result, runtime = gb_mach.fit(X_train, y_train)
gb_mach_pred = gb_mach.predict(X_test)
rmse_gb_mach = root_mean_squared_error(y_test, gb_mach_pred)

print(
    f"Gradient boosting machine\n"
    f"RMSE: {round(rmse_gb_mach, 5)}\n"
    f"Runtime fitting: {round(runtime / 60, 1)} min."
)

"""


##                                                                          ##
## Fit support vector regression

"""

svr = LinearSVR(
    C=1,
    epsilon=0.1,
    tol=1e-2
)
svr.fit = calc_runtime(svr.fit)

result, runtime = svr.fit(X_train, y_train)
svr_pred = svr.predict(X_test)
rmse_svr = root_mean_squared_error(y_test, svr_pred)

print(
    f"Support vector regression\n"
    f"RMSE: {round(rmse_svr, 5)}\n"
    f"Runtime fitting: {round(runtime / 60, 1)} min."
)

"""


##                                                                          ##
## Fit KNN regression

"""

knn_reg = KNeighborsRegressor(
    n_neighbors=47,
    weights='distance',
    algorithm='auto'
)
knn_reg.fit(X_train, y_train)

knn_reg.predict = calc_runtime(knn_reg.predict)
knn_reg_pred, runtime = knn_reg.predict(X_test)
rmse_knn_reg = root_mean_squared_error(y_test, knn_reg_pred)

print(
    f"KNN regression\n"
    f"RMSE: {round(rmse_knn_reg, 5)}\n"
    f"Runtime fitting: {round(runtime / 60, 1)} min."
)

"""


##                                                                          ##
## Fit multi-layer perceptron regression

"""

mlp_reg = MLPRegressor(
    max_iter=200, # # 500, 1000
    hidden_layer_sizes=(100, 50, 30), # # (200, 100), (100, 100, 50)
    alpha=0.0001, # # 1e-5, 1e-2
    learning_rate='constant', # # 'adaptive'
    solver='adam',
    batch_size=64, # # 'auto'
    early_stopping=True,
    tol=1e-3 # # 1e-4
)
mlp_reg.fit = calc_runtime(mlp_reg.fit)

result, runtime = mlp_reg.fit(X_train, y_train)
mlp_reg_pred = mlp_reg.predict(X_test)
rmse_mlp_reg = root_mean_squared_error(y_test, mlp_reg_pred)

print(
    f"Multi-layer perceptron regression\n"
    f"RMSE: {round(rmse_mlp_reg, 5)}\n"
    f"Runtime fitting: {round(runtime / 60, 1)} min."
)

"""


##                                                                          ##
## Fit LightGMB model

"""

data_train = lgb.Dataset(X_train, label=y_train)
data_test = lgb.Dataset(X_test, label=y_test, reference=data_train)

parameters = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 127,
    'learning_rate': 0.2,
    'max_depth': 20,
    'n_estimators': 200,
    'n_jobs': -1,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.6,
    'min_data_in_leaf': 60,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'seed': 42,
    'verbose': -1
}
lgb_reg = lgb.train(
    parameters,
    data_train,
    num_boost_round=100,
    valid_sets=[data_test],
)

lgb_reg_pred = lgb_reg.predict(X_test, num_iteration=lgb_reg.best_iteration)
rmse_lgb_reg = root_mean_squared_error(y_test, lgb_reg_pred)

print(
    f"LightBMG model\n"
    f"RMSE: {round(rmse_lgb_reg, 5)}\n"
)

"""


##                                                                          ##
## Fit XGB Regressor

"""

xgb = XGBRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=17,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.6,
    gamma=0,
    reg_lambda=1,
    reg_alpha=0,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42
)
xgb.fit = calc_runtime(xgb.fit)

result, runtime = xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
rmse_xgb = root_mean_squared_error(y_test, xgb_pred)

print(
    f"XGB regressor\n"
    f"RMSE: {round(rmse_xgb, 5)}\n"
    f"Runtime fitting: {round(runtime / 60, 1)} min."
)

"""


##                                                                          ##
## Fit stacking regressor

"""

base_learners = [
    ("ridge", Ridge(
        alpha=0.01,
        positive=False,
        solver='sag',
        tol=0.001
    )),
    ("tree",  DecisionTreeRegressor(
        criterion='squared_error',
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=20,
        ccp_alpha = 0.01,
        random_state=42
    )),
    ("rf", RandomForestRegressor(
        n_estimators=200,
        max_depth=40,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=0.3,
        warm_start=True,
        n_jobs=1,
        random_state=42
    )),
]

stack = StackingRegressor(
    estimators=base_learners,
    final_estimator=Ridge(alpha=0.1),
    passthrough=False, # set True to add raw features to meta-learner
    n_jobs=1
)

stack.fit(X_train, y_train)

stack_pred = stack.predict(X_test)
rmse_stack = root_mean_squared_error(y_test, stack_pred)

print(
    f"Stacking regressor\n"
    f"RMSE: {round(rmse_stack, 5)}\n"
)

"""
