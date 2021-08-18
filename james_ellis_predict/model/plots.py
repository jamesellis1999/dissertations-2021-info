import pandas as pd
from tabloo import show
import matplotlib.pyplot as plt
import numpy as np

def missing_data():
    df = pd.read_csv('model_data/baseline+wws+syn/data_closeness_eigenvector.csv')

    df['completeness'] = 1 - (df.isna().sum(axis=1)/df.shape[1])
    c = 'last_round_max_inv_eigenvector_centrality'
    data = df[[c, 'completeness']]

    bin_size = 0.02
    mean = []
    err = []
    x = np.arange(0.0, 0.13, 0.02) + 0.01
    for bin in np.arange(0.0, 0.13, 0.02):
        _data = data[(data[c]>=bin) & (data[c] < bin+bin_size)]
    
        mean.append(_data['completeness'].mean())
        err.append(_data['completeness'].sem())

    plt.errorbar(x, mean, yerr=err)
    plt.show()

if __name__=="__main__":
    missing_data()