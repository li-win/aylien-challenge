# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 21:52:57 2019
Python 3.7
@author: Lisa Winkler
"""

import pandas as pd
import numpy as np
from timeit import default_timer as timer

start = timer()

#read csv file that contains articles (headline, text, label)
df = pd.read_csv('aviation-test.csv')
#print(df)

import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import en_core_web_sm

nlp2 = en_core_web_sm.load()
punctuations = string.punctuation
nlp = English()
stopwords = spacy.lang.en.stop_words.STOP_WORDS

#Text preparation (can be done with regular expressions, too but string replacement was faster)
#delete unneccessary text parts
df['prep1'] = df['Body'].str.replace("\n", " ")
df['prep1'] = df['prep1'].str.replace("\r", " ")
df['prep1'] = df['prep1'].str.replace('"', " ")
df['prep1'] = df['prep1'].str.replace("'s", "")
#lowercase all text data
df['prep2'] = df['prep1'].str.lower()
#delete punctuation
df['prep3'] = df['prep2']
for punct in punctuations:
    df['prep3'] = df['prep3'].str.replace(punct, '')

#Lemmatize text and delete stop words
nrows = len(df)
lem_text_list = []
for row in range(0,nrows):
    lem_list = []
    text = df.loc[row]['prep3']
    #process text to spacy format
    words = nlp(text)
    for word in words:
        if word.is_stop==False:
            lem_list.append(word.lemma_)

    lem_text = ", ".join(lem_list)
    lem_text_list.append(lem_text)
df['prep_final'] = lem_text_list

#label news articles with -1 (incident) and 0 (no incident)
label_new = {'Incident': -1, 'Not Incident': 0}
df['Label New'] = df['Label']
df = df.replace({'Label New':label_new})

#Save data in excel file
df.to_excel('test.xlsx', index=False, header=True)

"""
#import test data that got labeled and cleaned before
test_df = pd.read_csv('test_labeling.csv')
x_test2 = test_df['prep_final']
y_test2 = test_df['Label New']
"""

#Train and Test with Multinomial Naive Bayes:
#split data in test and train set
x_train, x_test, y_train, y_test = train_test_split(df['prep_final'], df['Label New'], test_size=0.40, random_state=8)

#vectorize text data
tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words=None, lowercase=False, max_df=1., min_df=10, max_features=100, sublinear_tf=True, encoding='utf-8')

#train data
features_train = tfidf.fit_transform(x_train).toarray()
labels_train = y_train
print(features_train.shape)
#test data
features_test = tfidf.fit_transform(x_test).toarray()
labels_test = y_test
print(features_test.shape)
#prediction with Multinomial Naive Bayes Model
nb = MultinomialNB()
nb.fit(features_train, labels_train)
nb_pred = nb.predict(features_test)

print("Training accuracy: ", accuracy_score(labels_train,nb.predict(features_train)))
print("Test accuracy: ", accuracy_score(labels_test, nb_pred))
print("\n Classification report \n", classification_report(labels_test, nb_pred))
print("\n Confusion matrix: \n", confusion_matrix(labels_test, nb_pred))

#Show label predictions
#print(x_test)
#print(nb_pred)
data = pd.DataFrame()
data['x_test'] = x_test
data['nb_pred'] = nb_pred
label_new2 = {-1: 'Incident', 0: 'Not Incident'}
data['Label'] = data['nb_pred']
data = data.replace({'Label':label_new2})
#sort dataframe by text ID
data.sort_index(inplace=True)
data.to_excel('data1.xlsx')
#iterate over x_test data with NER to find places - save places in new column

end = timer()
print("\n elapsed time in seconds: ", end - start)