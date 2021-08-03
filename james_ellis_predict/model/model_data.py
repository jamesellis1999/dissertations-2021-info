import pandas as pd
import numpy as np
from tabloo import show

from company_info import company_info
from funding_info import funding_info
from founders_info import founders_info
from target_variable import add_target_variable

def model_data(tc, ts, tf):

    comp = company_info(tc, ts)
    fund = funding_info(tc, ts)
    founder = founders_info(tc, ts)

    df = pd.merge(comp, fund, left_on='uuid', right_on='org_uuid', how='left').drop(columns='org_uuid')
    df = pd.merge(df, founder, left_on='uuid', right_on='featured_job_organization_uuid', how='left').drop(columns='featured_job_organization_uuid')

    df = add_target_variable(df, tc, ts, tf)
    
    # Removing those that were successful in the warmup period
    df = df[df['successful'] != -1]

    df.to_csv('model_data_full.csv', index=False)
   
if __name__=="__main__":
    model_data('2013-12-01','2017-12-01', '2020-12-01')

