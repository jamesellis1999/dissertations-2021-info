import pandas as pd
import tabloo

data = {
    'investor' : [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2],
    'tp' : ['seed', 'seed', 'seed','seed','seed','seed','seed','seed','seed','series A', 'series B', 'series c', 'series B', 'series B','series d', 'series B','series B','series B' ]
}

df = pd.DataFrame(data)

def most_common(x):
    print(len(x.mode()))
    return str(x.mode()[0])

df['most_common_invest_type'] = df.groupby(['investor'])['tp'].transform(most_common)

tabloo.show(df)