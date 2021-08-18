import pandas as pd
import numpy as np
from tabloo import show

def control_model_data(save_path=None):
    _df = pd.read_csv('model_data/baseline+syn/data_eigenvector.csv')
    df = _df.dropna(axis='columns')
    cname = 'last_round_max_inv_eigenvector_centrality'
    df[cname] = _df.loc[:,cname]

    if save_path:
        df.to_csv(save_path, index=False)
    
    return df

if __name__=="__main__":
    control_model_data(save_path='model_data/baseline+syn/data_eigenvector_control.csv')

    
