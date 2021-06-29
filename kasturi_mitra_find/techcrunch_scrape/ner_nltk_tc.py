# only tag entities and extract those of interest; doesn't validate against database
# use NLTK built in
import pandas as pd
import spacy
import time
from itertools import *
import nltk
from nltk.tokenize import word_tokenize
from nltk.chunk import tree2conlltags
from nltk import pos_tag
import os
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('all')

data = pd.read_csv("techcrunch_articles.csv", index_col=0)
columns = ['article_id', 'people', 'orgs', 'potential_startups', 'matched_title']
article_entities = pd.DataFrame(columns=columns)



for i in range(0, len(data)):
    j = 0
    while j < 10:
        try:
            print(i, end=" ")
            j += 1

            title = data.iloc[i]['title']
            body = data.iloc[i]['body']
            tokens = word_tokenize(body)
            tag = pos_tag(tokens)
            # tags as a tree
            ne_tree = nltk.ne_chunk(tag)
            # convert to list of lists
            classified_body = tree2conlltags(ne_tree)
            # print(classified_body)

            n = len(classified_body)
            orgs = []

            # group organizations
            j = 0
            while j<n:
                org = ""
                k = 1
                if classified_body[j][2].find('ORGANIZATION') != -1:
                    org += classified_body[j][0] + " "
                    while j+k<n and classified_body[j+k][2].find('ORGANIZATION') != -1:
                        org += classified_body[j+k][0] + " "
                        k += 1
                    orgs.append(org.strip())
                j += k

            founders = []
            # group people
            j = 0
            while j < n:
                founder = ""
                k = 1
                if classified_body[j][2].find('PERSON') != -1:
                    founder += classified_body[j][0] + " "
                    while j+k<n and classified_body[j + k][2].find('PERSON') != -1:
                        founder += classified_body[j + k][0] + " "
                        k += 1
                    founders.append(founder.strip())
                j += k
            # print(founders)
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
                orgs[org] = orgs[org].replace("s '", "s")
                orgs[org] = orgs[org].replace("s ’", "s")

            # print(title)
            # print(orgs)
            potential_startups = []
            matched_title = 1
            # if an ORG in body is present in the title, it is a potential startup
            for org in orgs:
                if title.find(org) != -1:
                    potential_startups.append(org)
            if len(potential_startups) == 0:  # no matching ORG in body and title
                potential_startups = orgs
                matched_title = 0

            temp = pd.DataFrame([[i, founders, orgs, potential_startups, matched_title]], columns=columns)
            article_entities = pd.concat([article_entities, temp], ignore_index=True)
            print("")
            break
        except KeyboardInterrupt:
            pass
            break
        except Exception as e:
            print(e)
            pass

while True:
    try:
        time.sleep(5)
        article_entities.to_csv("articles_entities_nltk.csv", encoding='utf-8-sig')
        print("CSV stored!")
        break
    except KeyboardInterrupt:
        print("Trying again in 20s...")
    except:
        print("Error storing file...")
        pass