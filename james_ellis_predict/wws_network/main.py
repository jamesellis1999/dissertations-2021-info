import tabloo
import pandas as pd
import time
from job_transfer import job_transfer
from network import centrality

def wws_centrality(ts, metric='closeness'):
    data = job_transfer(ts)
    results = centrality(data, metric=metric)

    results.to_csv('wws_{}_centralities.csv'.format(metric))
    
if __name__=="__main__":
    metrics = ['closeness', 'betweenness', 'degree', 'eigenvector']
    for metric in metrics:
        t1 = time.time()
        wws_centrality('2017-12-15', metric=metric)
        print('Time for {} centrality: {:.2f}'.format(metric, time.time()-t1))