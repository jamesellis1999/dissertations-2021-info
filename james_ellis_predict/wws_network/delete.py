import pandas as pd
import tabloo

cpu = pd.read_csv('wws_betweeness_centralities_cpu.csv')
gpu = pd.read_csv('wws_betweeness_centralities.csv')

print(cpu.shape[0])
print(gpu.shape[0])

not_common = cpu[~cpu.isin(gpu)].dropna()

print(not_common.shape[0])

tabloo.show(not_common)