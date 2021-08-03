import pandas as pd 
import numpy as np
from itertools import product

def splitDataFrameList(df,target_column,separator):
    ''' df = dataframe to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 
    The values in the other columns are duplicated across the newly divided rows.
    '''
    def splitListToRows(row,row_accumulator,target_column,separator):
        split_row = row[target_column].split(separator)
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)
    new_rows = []
    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,separator))
    new_df = pd.DataFrame(new_rows)
    return new_df


def dict_product(d, random_seed=42):
    keys = d.keys()
    dict_list = []
    for element in product(*d.values()):
        dict_list.append(dict(zip(keys, element)))
    
    np.random.seed(random_seed)
    np.random.shuffle(dict_list)
    
    return dict_list