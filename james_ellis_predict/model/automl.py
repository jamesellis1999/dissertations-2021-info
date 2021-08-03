import h2o
import pandas as pd
import tabloo
from h2o.automl import H2OAutoML
from sklearn.model_selection import train_test_split

h2o.init()

# Import data
data_path = 'model_data_full.csv'
data = h2o.import_file(path=data_path)

# Splitting data into train (which will be split during cross validation) and test
train, test = data.split_frame(ratios=[0.75])

# Identify predictors and response
x = train.columns
y = 'successful'
x.remove(y)


# Columns to change to factor
binary_cols = ['USA', 'CHN', 'IND', 'GBR', 'SGP', 'SWE', 'DEU', 'CAN', 'KOR', 'other_country', 
'has_email', 'has_phone', 'has_facebook_url', 'has_twitter_url', 'has_linkedin_url', 'Internet Services', 
'Media and Entertainment', 'Software', 'Commerce and Shopping', 'Financial Services', 'Lending and Investments', 
'Payments', 'Hardware', 'Travel and Tourism', 'Video', 'Artificial Intelligence', 'Data and Analytics', 'Science and Engineering', 
'Mobile', 'Clothing and Apparel', 'Design', 'Information Technology', 'Manufacturing', 'Transportation', 'Music and Audio', 
'Professional Services', 'Sales and Marketing', 'Other', 'Community and Lifestyle', 'Events', 'Messaging and Telecommunications', 
'Privacy and Security', 'Content and Publishing', 'Education', 'Consumer Electronics', 'Food and Beverage', 'Real Estate', 'Apps', 
'Advertising', 'Health Care', 'Administrative Services', 'Government and Military', 'Consumer Goods', 'Platforms', 'Biotechnology',
'Gaming', 'Agriculture and Farming', 'Sports', 'Navigation and Mapping', 'Sustainability', 'Energy', 'Natural Resources', 
'last_round_type_angel', 'last_round_type_seed', 'last_round_type_grant', 'last_round_type_product_crowdfunding', 'last_round_type_pre_seed',
'has_bachelors', 'has_masters', 'has_phd', 'successful']

# Binary classification
train[binary_cols] = train[binary_cols].asfactor()
test[binary_cols] = test[binary_cols].asfactor()

# AutoML
aml = H2OAutoML(nfolds=5, include_algos=['XGBoost'], sort_metric='auc')
aml.train(x=x, y=y, training_frame=train)

lb = aml.leaderboard
lb.head(rows=lb.nrows)

