import pandas as pd
import matplotlib.pyplot as plt
from tabloo import show
from datetime import datetime
import numpy as np

def funding_info(tc, ts, syn_centrality=None, cb_rank=False, investor_type=False, stats=False):
    '''
    Creates the same company information features used by Arroyo et al. for use
    in the warmup period where tc <= warmup period < ts

    Parameters:
    tc (str): Datestring (yyyy-mm-dd) indicating start of the warmup window
    ts (str): Datestring (yyyy-mm-dd) indicating start of the simulation window
    syn_centrality (str): Centrality measure to use for investor syndicate network. 
                        If None, centrality not included. Must be one of 'betweenness', 
                        'closeness', 'degree', or 'eigenvector'
    cb_rank (bool): Use the reciprocal of the CB rank for the investor instead of centrality measure
    
    Returns:
    Pandas dataframe of company information features
    '''    
    
    # CSV imports
    fr_cols = ['uuid', 'investment_type', 'announced_on', 'raised_amount_usd', 'investor_count','org_uuid']
    fr = pd.read_csv('../../../data/raw/funding_rounds.csv', usecols=fr_cols)
    
    inv_cols = ['funding_round_uuid', 'investor_uuid']
    inv = pd.read_csv('../../../data/raw/investments.csv', usecols=inv_cols)

    # Only interested in funding rounds that occured in the warmup window
    fr = fr[
        (fr['announced_on'] >= tc) &
        (fr['announced_on'] < ts)
        ]
    
    # Less than series A condition 
    rounds = ['angel', 'seed', 'pre_seed']
    fr = fr[fr['investment_type'].isin(rounds)]

    if stats:
        # Shows significant spikes at the start of each year i.e trust code 4
        fr['announced_on'].value_counts(normalize=True).sort_index().plot()
        plt.show()

    # Round count in the warmup period 
    fr['round_count'] = fr.groupby('org_uuid')['uuid'].transform('size')

    # Total raised amount in all rounds in the warmup period, but put 0 to NaN
    fr['total_raised_amount_usd'] = fr.groupby('org_uuid')['raised_amount_usd'].transform('sum').replace(0.0, np.nan)

    # Last round investment type
    fr['last_round_investment_type'] = fr.sort_values('announced_on').groupby('org_uuid')['investment_type'].transform('last')

    # Last round amount raised in usd
    fr['last_round_raised_amount_usd'] = fr.sort_values('announced_on').groupby('org_uuid')['raised_amount_usd'].transform('last')

    # NOTE need to check having 0 months is valid
    # Last round timelapse in months
    def age_months(x):
        fmt = "%Y-%m-%d"
        tdelta = datetime.strptime(ts, fmt) - datetime.strptime(x.tail(1).item(), fmt)
        return round(tdelta.days/365.2425*12)

    fr['last_round_timelapse_months'] = fr.sort_values('announced_on').groupby('org_uuid')['announced_on'].transform(age_months)

    # Create intersection between funding rounds and the investors
    fr = pd.merge(fr, inv, left_on='uuid', right_on='funding_round_uuid', how='left')
    fr = fr.drop(columns = 'funding_round_uuid')

    # Number of investors column doesn't include non-renowned investors 
    fr['investor_count'] = fr['investor_count'].fillna(0)
    fr['investor_count'] += fr.groupby('uuid')['investor_uuid'].transform(lambda x: x.isna().sum())

    # Number of unique investors in the warmup period
    fr['total_investor_count'] = fr.groupby('org_uuid')['investor_uuid'].transform(lambda x: x.nunique(dropna=True) + x.isna().sum())

    # Syndicate network centralities
    if syn_centrality: 
        cs = ['betweenness', 'closeness', 'degree', 'eigenvector']
        if syn_centrality not in cs:
            raise ValueError('Invalid centrality measure. Must be one of "betweenness", "closeness", "degree", or "eigenvector"')
        else:
            syn_c_vals = pd.read_csv('network_centralities/syndicate_network/syndicate_{}_centralities.csv'.format(syn_centrality))
            fr = pd.merge(fr, syn_c_vals, on='investor_uuid', how='left').drop(columns='Unnamed: 0')

            fr['last_round_max_inv_{}_centrality'.format(syn_centrality)] = \
            fr.sort_values(['announced_on', '{}_centrality'.format(syn_centrality)]).groupby('org_uuid')['{}_centrality'.format(syn_centrality)].transform('last')

            if investor_type:
                investor_cols = ['uuid', 'investor_types']
                investors = pd.read_csv('../../../data/raw/investors.csv', usecols=investor_cols).rename(columns={'uuid':'investor_uuid'})
                fr = pd.merge(fr, investors, on='investor_uuid', how='left')
                fr['last_investor_type'] = fr.sort_values(['announced_on', '{}_centrality'.format(syn_centrality)]).groupby('org_uuid')['investor_types'].transform('last')
                fr = fr.drop(columns = 'investor_types')

            fr = fr.drop(columns= '{}_centrality'.format(syn_centrality))
        
    if cb_rank:
        if syn_centrality:
            raise ValueError('Please only pick one of centrality measure, or CB rank.')
        else:
            investor_cols = ['uuid', 'rank']
            investors = pd.read_csv('../../../data/raw/investors.csv', usecols=investor_cols).rename(columns={'uuid':'investor_uuid'})
    
            fr = pd.merge(fr, investors, on='investor_uuid', how='left')
            fr['last_round_best_inv_cbrank'] = \
            fr.sort_values(['announced_on', 'rank'], ascending=[True, False]).groupby('org_uuid')['rank'].transform('last')

            fr = fr.drop(columns='rank')
    
    

    # Number of unique renowned investors in the warmup period
    fr['known_investor_count'] = fr.groupby('org_uuid')['investor_uuid'].transform(lambda x: x.nunique(dropna=True))
    
    # Number of investors in the last funding round in the warmup window
    fr['last_round_investor_count'] = fr.sort_values('announced_on').groupby('org_uuid')['investor_count'].transform('last')

    # Number of renowned investors in the last funding round in the warmup window 
    fr['fund_round_known_investor_count'] = fr.groupby(['org_uuid', 'uuid'])['investor_uuid'].transform(lambda x: x.nunique(dropna=True))
    fr['last_round_known_investor_count'] = fr.sort_values('announced_on').groupby('org_uuid')['fund_round_known_investor_count'].transform('last')

    # Removing any unnecessary columns
    fr = fr.drop(columns = ['uuid','raised_amount_usd', 'investment_type', 'announced_on', 'investor_count', 
                            'investor_uuid', 'fund_round_known_investor_count'])

    # Dropping duplicates 
    fr = fr.drop_duplicates(subset=['org_uuid'])
  
    for round_type in rounds:
        fr['last_round_type_{}'.format(round_type)] = np.where(fr['last_round_investment_type'] == round_type, 1, 0)
    fr = fr.drop(columns='last_round_investment_type')

    return fr

def angel_seriesa(tc, ts):
    fr_cols = ['uuid', 'investment_type', 'announced_on', 'raised_amount_usd', 'investor_count','org_uuid']
    fr = pd.read_csv('../../../data/raw/funding_rounds.csv', usecols=fr_cols)
    
    inv_cols = ['funding_round_uuid', 'investor_uuid']
    inv = pd.read_csv('../../../data/raw/investments.csv', usecols=inv_cols)

    syn_c_vals = pd.read_csv('network_centralities/syndicate_network/syndicate_eigenvector_centralities.csv')

    pre_ts = fr[
        (fr['announced_on'] >= tc) &
        (fr['announced_on'] < ts)
        ]
    
    post_ts = fr[fr['announced_on'] >= ts]

    # Last funding round angel 
    pre_ts = pre_ts.sort_values('announced_on').drop_duplicates(subset=['org_uuid'], keep='last')
    
    rounds = ['angel', 'seed', 'pre_seed']
    pre_ts = pre_ts[pre_ts['investment_type'].isin(rounds)]

    pre_ts = pd.merge(pre_ts, inv, left_on='uuid', right_on='funding_round_uuid', how='left').drop(columns='funding_round_uuid')   
    pre_ts = pd.merge(pre_ts, syn_c_vals, on='investor_uuid', how='left').drop(columns='Unnamed: 0')

    pre_ts = pre_ts.sort_values('eigenvector_centrality').drop_duplicates(subset=['org_uuid'], keep='last')
    pre_ts = pre_ts.rename(columns = {'eigenvector_centrality': 'angel_centrality'})

    # Series A as next 
    post_ts = post_ts.sort_values('announced_on').drop_duplicates(subset=['org_uuid'], keep='first')
    beyond = ['series_c', 'series_d', 'series_e', 'series_f', 'series_g']
    post_ts = post_ts[post_ts['investment_type'] == 'angel']
    # post_ts = post_ts[post_ts['investment_type'].isin(beyond)]
    
    post_ts = pd.merge(post_ts, inv, left_on='uuid', right_on='funding_round_uuid', how='left').drop(columns='funding_round_uuid')  
    post_ts = pd.merge(post_ts, syn_c_vals, on='investor_uuid', how='left').drop(columns='Unnamed: 0')

    post_ts = post_ts.sort_values('eigenvector_centrality').drop_duplicates(subset=['org_uuid'], keep='last')
    post_ts = post_ts.rename(columns = {'eigenvector_centrality': 'series_a_centrality'})

    # Merge
    angel = pre_ts[['org_uuid', 'angel_centrality']]
    seriesa = post_ts[['org_uuid', 'series_a_centrality']]
    df = pd.merge(angel, seriesa, on='org_uuid', how='left').dropna()
    show(df)
    # df = df[df['angel_centrality']>0.02]
    
    df['beat'] = np.where(df['series_a_centrality'] >= df['angel_centrality'], 1,0)
    df['residual'] = df['series_a_centrality'] - df['angel_centrality']

    bin_size = 0.02
    mean_res = []
    err = []
    x = np.arange(0.0, 0.13, 0.02) + 0.01
    for bin in np.arange(0.0, 0.13, 0.02):
        _df = df[(df['angel_centrality']>=bin) & (df['angel_centrality'] < bin+bin_size)]
        # print(_df['beat'].value_counts(normalize=True)[1])
        mean_res.append(_df['residual'].mean())
        err.append(_df['residual'].sem())

    plt.errorbar(x, mean_res, yerr=err)
    plt.show()
    

    # print(avg)
    # print(err)
    # plt.scatter(df['angel_centrality'], df['series_a_centrality'])
    # plt.show()

if __name__=="__main__":
    show(funding_info('2013-12-15','2017-12-15', syn_centrality='eigenvector',  investor_type=True))
    
