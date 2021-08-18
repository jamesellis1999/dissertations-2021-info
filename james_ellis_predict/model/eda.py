import pandas as pd
import matplotlib.pyplot as plt
from tabloo import show



def countries():
    plt.style.use('plot_style.txt')
    df = pd.read_csv('model_data/baseline+wws+syn/data_closeness_eigenvector.csv')
    country_codes = ['other_country', 'USA', 'IND', 'GBR', 'CHN', 'CAN','DEU','SWE', 'KOR', 'SGP']
    counts = []
    success_frac = []
    for c in country_codes:
        _df = df[df[c] == 1]
        count = _df.shape[0]
        success = (_df['successful']==1).sum()/count

        counts.append(count)
        success_frac.append(success)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5), tight_layout=True)
    countries= ['Other', 'USA', 'India', 'UK', 'China', 'Canada', 'Germany', 'Sweden', 'South Korea', 'Singapore']
   
    ax1.bar(countries, counts, color='grey')
    ax2.bar(countries, success_frac, color='grey')

    ax1.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='x', rotation=45)

    ax1.set_ylabel('Startup Count')
    ax2.set_ylabel('Startup Success Rate')
    

    avg_success = (df['successful']==1).sum()/df.shape[0]

    ax2.axhline(y=avg_success,linewidth=1, color='k', linestyle='--', label='Average Success Rate')
    ax2.legend()

    plt.show()

def sectors():
    df = pd.read_csv('model_data/baseline+wws+syn/data_closeness_eigenvector.csv')
    sectors = ['Internet Services', 'Media and Entertainment', 'Software',
       'Commerce and Shopping', 'Financial Services',
       'Lending and Investments', 'Payments', 'Hardware', 'Travel and Tourism',
       'Video', 'Artificial Intelligence', 'Data and Analytics',
       'Science and Engineering', 'Mobile', 'Clothing and Apparel', 'Design',
       'Information Technology', 'Manufacturing', 'Transportation',
       'Music and Audio', 'Professional Services', 'Sales and Marketing',
       'Other', 'Community and Lifestyle', 'Events',
       'Messaging and Telecommunications', 'Privacy and Security',
       'Content and Publishing', 'Education', 'Consumer Electronics',
       'Food and Beverage', 'Real Estate', 'Apps', 'Advertising',
       'Health Care', 'Administrative Services', 'Government and Military',
       'Consumer Goods', 'Platforms', 'Biotechnology', 'Gaming',
       'Agriculture and Farming', 'Sports', 'Navigation and Mapping',
       'Sustainability', 'Energy', 'Natural Resources']
    
  
    counts = []
    success_frac = []
    for s in sectors:
        _df = df[df[s] == 1]
        count = _df.shape[0]
        success = (_df['successful']==1).sum()/count

        counts.append(count)
        success_frac.append(success)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5), tight_layout=True)
   
    ax1.bar(sectors, counts, color='grey')
    ax2.bar(sectors, success_frac, color='grey')

    ax1.tick_params(axis='x', rotation=90)
    ax2.tick_params(axis='x', rotation=90)


    avg_success = (df['successful']==1).sum()/df.shape[0]

    ax2.axhline(y=avg_success,linewidth=1, color='k', linestyle='--', label='Average Success Rate')
    ax2.legend()

    plt.show()


if __name__=="__main__":
    sectors()
