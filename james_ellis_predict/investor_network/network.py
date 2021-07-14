import networkx as nx 
import pandas as pd
from investments import investments
from networkx.algorithms import bipartite
from networkx.algorithms.centrality import closeness_centrality, degree_centrality, betweenness_centrality
import tabloo

def centrality_t(datestring, metric = 'closeness'):
    '''
    Creates a network at a given time 
    '''

    metrics = {
        'closeness': closeness_centrality,
        'degree': degree_centrality,
        'betweenness': betweenness_centrality
    }

    if metric not in metrics:
        raise Exception("Invalid centrality metric. Pick one of 'closeness', 'degree', 'betweenness'")

    data = investments(datestring)

    invest_type = data[['lead_investor_uuids', 'most_common_invest_type']].drop_duplicates(subset='lead_investor_uuids')

    G = nx.from_pandas_edgelist(data, 'org_uuid', 'lead_investor_uuids')

    P = bipartite.projected_graph(G, data['lead_investor_uuids'])

    # Calculating centrality values for a given metric
    c_vals = metrics[metric](P)
    
    df = pd.DataFrame(c_vals.items(), columns = ['uuid', '{}_centrality'.format(metric)])
    investors = pd.read_csv('../../../data/raw/investors.csv', usecols=['uuid', 'name', 'investor_types', 'investment_count', 'total_funding_usd'])
    
    merge = pd.merge(df, investors, on='uuid', how='left')
    merge = pd.merge(merge, invest_type, left_on='uuid', right_on='lead_investor_uuids', how='left')
    tabloo.show(merge)

    
if __name__=="__main__":
    centrality_t('2020-01-01')

