import pandas as pd 
import time 

# import tabloo
import xgboost as xgb

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from utils import dict_product

# Same as AutoML search https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
paramGrid = {
    'booster': ['gbtree', 'dart'],
    'colsample_bylevel': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'max_depth': [5, 10, 15, 20],
    'min_child_weight': [0.01, 0.1, 1.0, 3.0, 5.0, 10.0, 15.0, 20.0],
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
    'lambda': [0.001, 0.01, 0.1, 0.5, 1],
    'subsample': [0.6, 0.8, 1.0],
    'eta': [0.3]
}

# Import data
data = pd.read_csv('model_data_full.csv')

# Define independent and dependent variables
y = data.successful.astype(int)
X = data.drop(columns=['uuid', 'successful'])

# Using same random state ensures no data leakage into the test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=42)

# Converting to dmatrix for xgboost module
dtrain = xgb.DMatrix(X_train, label=y_train)

df = pd.DataFrame(columns=['AUC', 'Boosting Rounds', 'params'])

# Random parameter search for an hour
t1 = time.time()
while time.time() - t1 < 3600:

    for i, params in enumerate(dict_product(paramGrid)):
        
        params['objective'] = 'binary:logistic'
        params['tree_method'] = 'gpu_hist'
        cv_result = xgb.cv(
            params,
            dtrain,
            num_boost_round=10000,
            early_stopping_rounds=10,
            nfold=5,
            stratified=True,
            metrics=['auc'],
            as_pandas=True
        )
        
        df.loc[i, 'Boosting Rounds'] = cv_result.shape[0]
        df.loc[i, 'AUC'] = cv_result.iloc[-1]['test-auc-mean']
        df.loc[i, 'params'] = str(params)
        
        if i == 1:
            break
    
    break

tabloo.show(df)



