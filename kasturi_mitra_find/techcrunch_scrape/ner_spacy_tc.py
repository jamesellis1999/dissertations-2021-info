# only tag entities and extract those of interest; doesn't validate against database
# use SpaCy
import pandas as pd
import spacy
import time
from itertools import *

data = pd.read_csv("techcrunch_articles.csv", index_col=0)
columns = ['article_id', 'people', 'orgs', 'potential_startups']
article_entities = pd.DataFrame(columns=columns)

for i in range(0, len(data)):
    j = 0
    while j<10:
        try:
            print(i, end=" ")
            j += 1

            body = data.iloc[i]['body']
            title = data.iloc[i]['title']
            # print(title)

            # NER with SpaCy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(body)
            # for entity in doc.ents:
            #     print(entity.body, entity.label_)
            entities = {key: list(set(map(lambda x: str(x), g))) for key, g in groupby(sorted(doc.ents, key=lambda x: x.label_), lambda x: x.label_)}
            orgs = None
            try:
                orgs = entities['ORG']
            except: # no key error
                orgs = []
            # print(orgs)

            # remove possesive apostrophes 's and ’s (two distinct unicode characters)
            title = title.replace("'s", "")
            title = title.replace("’s", "")
            # for words ending with 's'
            title = title.replace("s'", "s")
            title = title.replace("s’", "s")
            for org in range(len(orgs)):  # replace in place
                orgs[org] = orgs[org].replace("'s", "")
                orgs[org] = orgs[org].replace("’s", "")
                orgs[org] = orgs[org].replace("s'", "s")
                orgs[org] = orgs[org].replace("s’", "s")

            potential_startups = []
            # if an ORG in body is present in the title, it is a potential startup
            for org in orgs:
                if title.find(org) != -1:
                    potential_startups.append(org)
            if len(potential_startups) == 0: # no matching ORG in body and title
                potential_startups = orgs

            people = None
            try:
                people = entities['PERSON']
            except:  # no key error
                people = []
            potential_founders = people

            temp = pd.DataFrame([[i, potential_founders, orgs, potential_startups]], columns=columns)
            article_entities = pd.concat([article_entities, temp], ignore_index=True)

            print("")
            break
        except KeyboardInterrupt:
            pass
            break
        except e:
            print(e)
            pass

# save locally
while True:
    try:
        time.sleep(5)
        article_entities.to_csv("articles_entities_spacy.csv", encoding='utf-8-sig')
        print("CSV stored!")
        break
    except KeyboardInterrupt:
        print("Trying again in 20s...")
    except:
        print("Error storing file...")
        pass