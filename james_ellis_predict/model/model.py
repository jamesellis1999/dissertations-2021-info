import pandas as pd 
import numpy as np
import xgboost as xgb

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold

# Import data
data = pd.read_csv('model_data_full.csv')

# Define independent and dependent variables
y = data.successful.astype(int)
X = data.drop(columns=['uuid', 'successful'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y)

# Loading data into DMatrices

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# params
params = {
    'max_depth': 6,
    'min_child_weight': 1,
    'eta': 0.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective':'binary:logistic'
}

params['eval_metric'] = 'auc'

num_boost_round = 999

# model
# model = xgb.train(
#     params,
#     dtrain,
#     num_boost_round=num_boost_round,
#     evals=[(dtest, "Test")],
#     early_stopping_rounds=10
# )

# print("Best AUC: {:.2f} with {} rounds".format(model.best_score, model.best_iteration+1))

# Using cross validation
# cv_results = xgb.cv(
#     params,
#     dtrain,
#     num_boost_round=num_boost_round,
#     seed=42,
#     nfold=5,
#     metrics={'auc', 'F1'},
#     early_stopping_rounds=10,
#     stratified=True
# )

# Optimising max_depth and min_child_weight parameters

# gridsearch_params = [
#     (max_depth, min_child_weight)
#     for max_depth in range(2,6)
#     for min_child_weight in range(2,20)
# ]

# max_auc = -float("Inf")
# best_params = None
# for max_depth, min_child_weight in gridsearch_params:
#     print("CV with max_depth = {}, min_child_weight={}".format(
#         max_depth,
#         min_child_weight
#     ))

#     params['max_depth'] = max_depth
#     params['min_child_weight'] = min_child_weight

#     cv_results = xgb.cv(
#         params,
#         dtrain,
#         num_boost_round=num_boost_round,
#         seed=42,
#         nfold=5,
#         stratified=True,
#         metrics={'auc'},
#         early_stopping_rounds=10
#     )

#     mean_auc = cv_results['test-auc-mean'].max()
#     boost_rounds = cv_results['test-auc-mean'].argmax()
#     print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
#     if mean_auc > max_auc:
#         max_auc = mean_auc
#         best_params = (max_depth, min_child_weight)

# print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))

# Best params are max depth of 4  and 18 min_child_weight

params = {
    'max_depth': 4,
    'min_child_weight': 18,
    'eta': 0.3,
    'subsample': 1,
    'colsample_bytree': 1,
    'objective':'binary:logistic'
}

# Optimising subsample and colsample_bytree

# gridsearch_params = [
#     (subsample, colsample)
#     for subsample in [i/10 for i in range(7,11)]
#     for colsample in [i/10 for i in range(7,11)]
# ]

# max_auc = -float("Inf")
# best_params = None
# for subsample, colsample in reversed(gridsearch_params):
#     print("CV with subsample = {}, colsample={}".format(
#         subsample,
#         colsample
#     ))

#     params['subsample'] = subsample
#     params['colsample_bytree'] = colsample

#     cv_results = xgb.cv(
#         params,
#         dtrain,
#         num_boost_round=num_boost_round,
#         seed=42,
#         nfold=5,
#         stratified=True,
#         metrics={'auc'},
#         early_stopping_rounds=10
#     )

#     mean_auc = cv_results['test-auc-mean'].max()
#     boost_rounds = cv_results['test-auc-mean'].argmax()
#     print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
#     if mean_auc > max_auc:
#         max_auc = mean_auc
#         best_params = (subsample, colsample)

# print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))


# params = {
#     'max_depth': 4,
#     'min_child_weight': 18,
#     'eta': 0.3,
#     'subsample': 0.9,
#     'colsample_bytree': 0.9,
#     'objective':'binary:logistic'
# }

# max_auc = -float("Inf")
# best_params = None
# for eta in [0.3, 0.2, 0.1, 0.05, 0.01, 0.005]:
#     print("CV with eta={}".format(eta))

#     params['eta'] = eta

#     cv_results = xgb.cv(
#         params,
#         dtrain,
#         num_boost_round=num_boost_round,
#         seed=42,
#         nfold=5,
#         stratified=True,
#         metrics={'auc'},
#         early_stopping_rounds=10
#     )

#     mean_auc = cv_results['test-auc-mean'].max()
#     boost_rounds = cv_results['test-auc-mean'].argmax()
#     print("\tAUC {} for {} rounds".format(mean_auc, boost_rounds))
#     if mean_auc > max_auc:
#         max_auc = mean_auc
#         best_params = eta

# print("Best params: {}, AUC: {}".format(best_params, max_auc))

params = {
    'max_depth': 4,
    'min_child_weight': 18,
    'eta': 0.05,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'eval_metric': 'auc',
    'objective':'binary:logistic',
    'scale_pos_weight': 5
}

model = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    early_stopping_rounds=10,
    evals=[(dtrain, "Train")]
    
)

y_pred = model.predict(dtest)

cm = confusion_matrix(y_test, (y_pred>0.5))
print(cm)

print("Accuracy: {}".format( (cm[0,0] + cm[1,1]) / (cm[0,0] + cm[1,1] + cm[1,0] + cm[0,1]) ))
print("Recall: {}".format( (cm[0,0]) / (cm[0,0] + cm[1,0]) ))
