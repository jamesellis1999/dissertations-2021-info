import networkx as nx 
import pandas as pd
import tabloo
import numpy as np
import matplotlib.pyplot as plt

from networkx.algorithms.bipartite.projection import projected_graph


data = {
    'from': [1,2,3,4,5,6,7,8,9],
    'to': ['A','A','A','A', 'B', 'B', 'B', 'C', 'C'],
    'lead': [True, False, False, False, True, False, False, False, False],
    'num_lead': [1,1,1,1,1,1,1,0,0]
}

df = pd.DataFrame(data, columns=[*data])

G = nx.DiGraph()

# Add lead
lead = df[ (df['lead'] == True)  & (df['num_lead'] == 1)]
G.add_edges_from(zip(lead['from'], lead['to']))

# Add non lead
non_lead = df[ (df['lead'] == False) & (df['num_lead'] >=1) ]
G.add_edges_from(zip(non_lead['to'], non_lead['from']))

# No leader in funding round
non_lead_nolead = df[ (df['lead'] == False) & (df['num_lead'] == 0) ]
G.add_edges_from(zip(non_lead_nolead['to'], non_lead_nolead['from']))
G.add_edges_from(zip(non_lead_nolead['from'], non_lead_nolead['to']))



P = projected_graph(G, [1,2,3,4,5,6,7,8,9])
nx.draw(P, with_labels=True)
plt.show()
