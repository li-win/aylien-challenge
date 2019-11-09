# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 12:51:38 2019
Python 3.7
@author: Lisa Winkler

Read json file with news and save relevant data (title, body) in dataframe
"""

import pandas as pd
import json
from pandas.io.json import json_normalize
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import spacy
import en_core_web_sm
import string
from sklearn.model_selection import train_test_split
import numpy as np

nlp = en_core_web_sm.load()

#creating keyword lists that are relevant to find aviation incidents in news
aviation_terms = ['air', 'air cargo', 'airport', 'arrival area', 'baggage', 
                  'baggage area', 'concessions', 'concourse', 'departure area',
                  'destination', 'flight', 'gates', 'hangar', 'hub', 'landing', 
                  'runway', 'take-off', 'terminal', 'tower', 'aircraft', 'attack',
                  'biplane', 'fighter', 'fixed-wing aircraft', 'jet', 'propeller', 
                  'helicopter', 'pilot', 'triplane', 'bomber', 'steward', 'stewardess',
                  'passenger']
incident_terms = ['weather', 'pothole', 'damage', 'damaged', 'president', 'strike', 
                  'protest', 'mechanical', 'failure', 'error']

with open('news.json', encoding='utf-8') as data_file:
    data = json.load(data_file)
    
df = json_normalize(data, 'stories')
df.to_excel('news.xlsx', index=False, header=True)
body = df.iloc[:, 1].values
title = df.iloc[:, 19].values

nrows = len(df)

#created new dataframe for better overview when checking results
df2 = pd.DataFrame()
df2['title'] = df['title']
df2['body'] = df['body']

nlp2 = en_core_web_sm.load()
punctuations = string.punctuation
nlp = English()
stopwords = spacy.lang.en.stop_words.STOP_WORDS


#Text preparation (can be done with regular expressions, too. string replacement method was faster)
#delete unneccessary signs and "'s"
df2['prep1'] = df2['body'].str.replace("\n", " ")
df2['prep1'] = df2['prep1'].str.replace("\r", " ")
df2['prep1'] = df2['prep1'].str.replace('"', " ")
df2['prep1'] = df2['prep1'].str.replace("'s", "")
#lowercase all words
df2['prep2'] = df2['prep1'].str.lower()
#delete punctuation
df2['prep3'] = df2['prep2']
for punct in punctuations:
    df2['prep3'] = df2['prep3'].str.replace(punct, '')

#Lemmatize text and delete stopwords
lem_text_list = []
for row in range(0,nrows):
    lem_list = []
    text = df2.loc[row]['prep3']
    #process text to spacy format
    words = nlp(text)
    for word in words:
        if word.is_stop==False:
            lem_list.append(word.lemma_)

    lem_text = ", ".join(lem_list)
    lem_text_list.append(lem_text)

df2['prep_final'] = lem_text_list


#avoid duplicate words in keyword lists    
def unique(data):
    return list(dict.fromkeys(data))
#check text for relevant keywords
test_lem_list = []
test_lem_list2 = []
nrows = len(df)
for row in range(0,nrows):
    test_label = []
    test_label2 = []
    text = df.loc[row]['body']
    words = nlp(text)
    for word in words:
        if word.text in aviation_terms:
            test_label.append(word.lemma_)
        if word.text in incident_terms:
            test_label2.append(word.lemma_)
    test_lem = " ".join(unique(test_label))
    test_lem_list.append(test_lem)
    test_lem2 = " ".join(unique(test_label2))
    test_lem_list2.append(test_lem2)
df2['test label'] = test_lem_list
df2['test label 2'] = test_lem_list2

#if a text body contains keywords from aviation terms and incident terms 'Label' is set to 1
#doesn't word
df2['Label'] = np.where((df2['test label'] != None) & (df2['test label 2'] != None), 1, np.nan)

"""
var3 = 0
label_list_all = []
for row in range(0,nrows):
    label_list = []
    var1 = df2.loc[row]['test label']
    var2 = df2.loc[row]['test label 2']
    if (var1 == None) and (var2 == None):
        var3 = 0
        label_list_all.append(var3)
    else:
        var3 = 1
        label_list_all.append(var3)    
df2['Label'] = label_list_all
"""
#save relevant data in dataframe for later usage        
sentiment = df.loc[row]['sentiment']
keywords = df.loc[row]['keywords']

#Save data in excel file
df2.to_excel('test_labeling.xlsx', index=False, header=True)

"""
#prepare test data for classifier
#split data in test and train set
x_train, x_test, y_train, y_test = train_test_split(df2['prep_final'], df2['Label'], test_size=0.40, random_state=8)

#vectorize text data
tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words=None, lowercase=False, max_df=1., min_df=10, max_features=100, sublinear_tf=True, encoding='utf-8')

features_test = tfidf.fit_transform(x_test).toarray()
labels_test = y_test
"""