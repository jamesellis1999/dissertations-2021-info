import pandas as pd
import re
import selenium
import requests
from bs4 import BeautifulSoup

links = pd.read_csv("techcrunch_links.csv", index_col=False)
columns = ['title', 'author', 'date', 'body']
data = pd.DataFrame(columns=columns)

for i in range(0, len(links)):
    j = 0
    # try every article 10 times before moving onto the next
    while j<10:
        j += 1
        print(i, end=" ")
        try:
            url = links.iloc[i][0]
            page = requests.get(url)
            soup = BeautifulSoup(page.content, 'html.parser')

            title = soup.find("h1", class_="article__title").text
            author = soup.find("div", class_="article__byline")
            author = author.find("a").text.strip()
            date = re.findall(r'[0-9]{4}/[0-9]{2}/[0-9]{2}', url)[0]
            body = soup.find("div", class_="article-content")

            # remove all links to other articles which are part of body
            other_link = body.find("div", class_="embed breakout")
            while other_link != None:
                body.find("div", class_="embed breakout").decompose()
                other_link = body.find("div", class_="embed breakout")
            body = body.text

            # append to dataframe
            article = pd.DataFrame([[title, author, date, body]], columns=columns)
            data = pd.concat([data, article], ignore_index=True)
            # every 50 articles, store locally
            if i%50==0:
                data.to_csv('techcrunch_articles.csv', encoding='utf-8-sig')
            break
        except:
            pass
    print("")
    # save any remaining articles
    data.to_csv('techcrunch_articles.csv', encoding='utf-8-sig')