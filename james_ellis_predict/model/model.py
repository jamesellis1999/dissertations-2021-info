import pandas as pd 
import numpy as np
import xgboost as xgb
import json
import shap
import time
import matplotlib
import matplotlib.pyplot as plt
from tabloo import show

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

from funding_info import funding_info

def run_model(wws_centrality=None, syn_centrality=None, n_repeats=3, plot=False, custom_data=None, custom_json=None):

    accs = []
    recalls = []
    precisions = []
    df_list = []

    # Logic to point to correct data
    if wws_centrality == None and syn_centrality == None:
        JSON_PATH = 'optimisation_results/baseline'
        DATA_PATH = 'model_data/baseline/data.csv'
        MATCH_STRING = None
    if wws_centrality:
        JSON_PATH = 'optimisation_results/baseline+wws_{}'.format(wws_centrality)
        DATA_PATH = 'model_data/baseline+wws+syn/data_{}.csv'.format(wws_centrality)
        MATCH_STRING = wws_centrality
    if syn_centrality:
        JSON_PATH = 'optimisation_results/baseline+wws_{}'.format(syn_centrality)
        DATA_PATH = 'model_data/baseline+wws+syn/data_{}.csv'.format(syn_centrality)
        MATCH_STRING = syn_centrality
    if wws_centrality and syn_centrality:
        JSON_PATH = 'optimisation_results/baseline+wws_{}+syn_{}'.format(wws_centrality, syn_centrality)
        DATA_PATH = 'model_data/baseline+wws+syn/data_{}_{}.csv'.format(wws_centrality, syn_centrality)
        MATCH_STRING = '{}|{}'.format(wws_centrality, syn_centrality)

    if custom_data and custom_json:
        JSON_PATH = custom_json
        DATA_PATH = custom_data

    for i in range(n_repeats):
     
        SEED = i + 1
        EXP_JSON_PATH = JSON_PATH + '-{}-hp.json'.format(SEED)

        data = pd.read_csv(DATA_PATH)

        y = data.successful.astype(int)
        X = data.drop(columns=['uuid', 'successful'])
        
        # Split data according to seed provided
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=SEED)
    
        # Loading data into DMatrices
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Load optimised parameters
        f = open(EXP_JSON_PATH)
        params = json.load(f)

        num_boost_rounds = params['n_estimators']
        del params['n_estimators']

        # Some extra parameters for all models
        params['objective'] = 'binary:logistic'
        params['scale_pos_weight'] = 3.7
        params['eval_metric'] = 'auc'
        params['booster'] = 'gbtree'
        params['tree_method'] = 'gpu_hist'

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_rounds
        )
    
        y_pred = model.predict(dtest)
        predictions = [round(val) for val in y_pred]

        acc = accuracy_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        precision = precision_score(y_test, predictions)

        accs.append(acc)
        recalls.append(recall)
        precisions.append(precision)

        if MATCH_STRING is not None:
            explainer = shap.Explainer(model)
            shap_values = explainer(X_train)
            cohorts= {"": shap_values}
            cohort_labels = list(cohorts.keys())
            cohort_exps = list(cohorts.values())

            feature_vals = cohort_exps[0].values
            feature_names = cohort_exps[0].feature_names

            mean_abs_shap = np.average(np.absolute(feature_vals), axis=0)
        
            df = pd.DataFrame({'feature':feature_names, 'mean_abs_shap':mean_abs_shap})
            df = df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=False)
            df['rank'] = df.index.get_level_values(0).values + 1

            # Only want centrality feature(s)
            # df = df[ df['feature'].str.contains(MATCH_STRING)]
            df_list.append(df)
        
        if plot:
            plt.style.use('plot_style.txt')
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5), tight_layout=True)
            
            if plot == 'linkedin':
                explainer = shap.Explainer(model)
                shap_values = explainer(X_train)
            
                wws_feature = 'know_how_closeness_centrality'
                syn_feature = 'last_round_max_inv_eigenvector_centrality'
                
                
                idx = np.where(X_train['has_linkedin_url']==1)
        
                lin = shap.Explanation(
                    values = shap_values[:,wws_feature].values[idx], 
                    base_values= shap_values[:,wws_feature].base_values[idx],
                    data = shap_values[:,wws_feature].data[idx])

                shap.plots.scatter(shap_values[:,wws_feature], dot_size=6, color='#000000', x_jitter=0.004, ax=ax1, show=False)
                shap.plots.scatter(lin, dot_size=6, color='#000000', x_jitter=0.004, ax=ax2, show=False)
        
                ax1.set_ylabel('SHAP Value for \n WWS Closeness Centrality')
                ax1.set_xlabel('WWS Closeness Centrality')

                ax2.set_ylabel('SHAP Value for \n WWS Closeness Centrality \n with LinkedIn Presence')
                ax2.set_xlabel('WWS Closness Centrality with LinkedIn Presence')

                ax1.set_ylim(-1,1.75)
                ax2.set_ylim(-1,1.75)
                
                plt.show()

            if plot == 'completeness':
                explainer = shap.Explainer(model)
                shap_values = explainer(X_train)

                X_train['completeness'] = 1 - (X_train.isna().sum(axis=1)/X_train.shape[1])
                c = 'last_round_max_inv_eigenvector_centrality'
                data = X_train[[c, 'completeness']]
                
                bin_size = 0.02
                mean = []
                err = []
                x = np.arange(0.0, 0.13, 0.02) + 0.01
                for bin in np.arange(0.0, 0.13, 0.02):
                    _data = data[(data[c]>=bin) & (data[c] < bin+bin_size)]
                
                    mean.append(_data['completeness'].mean())
                    err.append(_data['completeness'].sem())

                
                shap.plots.scatter(shap_values[:,c], dot_size=6, color='#000000', x_jitter=0.004, ax=ax1, show=False)
                ax2.errorbar(x, mean, yerr=err, color='black')

                ax1.set_ylabel('SHAP Value for \n Syndicate Eigenvector Centrality')
                ax1.set_xlabel('Syndicate Eigenvector Centrality')

                ax2.set_ylabel('Data Completeness')
                ax2.set_xlabel('Syndicate Eigenvector Centrality')
                plt.show()

            if plot == 'bias_control':

                fig, ax = plt.subplots(figsize=(6,5), tight_layout=True)

                explainer = shap.Explainer(model)
                shap_values = explainer(X_train)
                c = 'last_round_max_inv_eigenvector_centrality'

                shap.plots.scatter(shap_values[:,c], dot_size=6, color='#000000', ax=ax, x_jitter=0.004, show=False)

                ax.set_ylabel('SHAP Value for \n Controlled Syndicate Eigenvector Centrality')
                ax.set_xlabel('Syndicate Eigenvector Centrality')
                plt.show()
            
    average_acc = np.average(accs)
    average_recall = np.average(recalls)
    average_precision = np.average(precisions)

    err_acc = np.std(accs)/np.sqrt(n_repeats)
    err_recall = np.std(recalls)/np.sqrt(n_repeats)
    err_precision = np.std(precisions)/np.sqrt(n_repeats)

    print('Results for wws: {}, syn: {}'.format(wws_centrality, syn_centrality))
    print('\tAverage Test Accuracy: {:.4f}+-{:.4f}'.format(average_acc, err_acc))
    print('\tAverage Test Recall: {:.4f}+-{:.4f}'.format(average_recall, err_recall))
    print('\tAverage Test Precision: {:.4f}+-{:.4f}'.format(average_precision, err_precision))

    # i.e. there was a centrality measure in the experiment
    # if MATCH_STRING is not None: 

    #     df_concat = pd.concat(df_list)
    #     df_concat = df_concat.set_index('feature', drop=False)
    #     by_row_index = df_concat.groupby(df_concat.index)

    #     df_concat['mean_shap'] = by_row_index['mean_abs_shap'].mean()
    #     df_concat['mean_shap_err'] = by_row_index['mean_abs_shap'].std()/np.sqrt(n_repeats)
    #     df_concat['median_rank'] = by_row_index['rank'].median()

    #     df_concat = df_concat[['index','feature','mean_shap', 'mean_shap_err', 'median_rank']].drop_duplicates()
    
    #     order = df_concat['index'].to_numpy()

    #     plt.style.use('plot_style.txt')

    #     fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5), tight_layout=True)
        
    #     shapvals = df_concat['mean_shap'].to_list()[:15]
    #     shap_err = df_concat['mean_shap_err'].to_list()[:15]
    #     x_pos = np.flip(np.linspace(-0.2,16.3,15))

    #     ax1.barh(x_pos, shapvals, height=0.5, yerr=shap_err, color='grey')
    #     ax1.set_ylim(-2.5, 17.5)
    #     ax1.set_yticks(x_pos)
    #     ax1.set_xlabel('Mean(|SHAP Value|)')
    #     shap.plots.beeswarm(shap_values, max_display=16, order=order, show=False)
    #     ax2.axes.yaxis.set_visible(False)
    #     plt.show()

    #     show(df_concat)

        
        # for i in range(df_concat.shape[0]):
        #     row = df_concat.iloc[i]
        
        #     print('\tSHAP info on feature: {}'.format(row['feature']))
        #     print('\t\tAverage of the mean mod shap value: {:.4f}+-{:.4f} '.format(row['mean_shap'], row['mean_shap_err']))
        #     print('\t\tMedian SHAP rank: {}'.format(row['median_rank']))

if __name__=="__main__":
    # run_model('baseline')

    # centralities = ['closeness', 'betweenness', 'degree', 'eigenvector']
    
    # run_model(wws_centrality='closeness', syn_centrality='eigenvector', plot='completeness', n_repeats=1)


    
    # run_model(wws_centrality='closeness', syn_centrality='degree')
   
    # run_model(wws_centrality='closeness', syn_centrality='eigenvector', plot=True)

    run_model(custom_data='model_data/baseline+syn/data_eigenvector_control.csv', custom_json='optimisation_results/baseline+syn_eigenvector_control', n_repeats=1, plot='bias_control')