#!/usr/bin/env python3

"""
Baseline model.
Run as ./baseline.py datadir,
e.g. ./baseline ../data/english/
if running from code/
"""

import os, sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD


from nltk.corpus import stopwords

from getdata import *
from utils import *
import codecs
from output_scores import output_scores


if __name__ == "__main__":
    test_datafilename = sys.argv[1]
    train_datafilename = sys.argv[2]
    train_datafile = codecs.open(train_datafilename,encoding='utf-8', errors = 'ignore')
    test_datafile = codecs.open(test_datafilename,encoding='utf-8', errors = 'ignore')

    train_data = train_datafile.readlines()
    test_data = test_datafile.readlines()
    tweets_train = []
    labels_train = []
    #parse train data
    for line in train_data:
        seperated = line.split('\t')
        label = seperated[-1].strip()
        tweet = seperated[-2].strip()
        tweets_train.append(tweet)
        labels_train.append(label)


    #parse test data
    tweets_test = []
    labels_test = []
    #parse train data
    for line in test_data:
        seperated = line.split('\t')
        label = seperated[-1].strip()
        tweet = seperated[-2].strip()
        tweets_test.append(tweet)
        labels_test.append(label)
   
   # tweets_test, labels_test = get_alldata(test_datadir)
   # gender_test_labels = [l[1].lower() for l in labels_test]
    #flatten
   # all_tweets_test = []
   # all_gender_test = []
   # for index, (author, g)  in enumerate(zip(tweets_test, gender_test_labels)):
   #     gvalue = g
    #    clean_tweets = []
     #   for index_2, tweet in enumerate(author):
     #       t = tweet.strip()
     #       all_gender_test.append(gvalue)
     #       all_tweets_test.append(t)

  #  print(len(all_gender_test))
    vec_gen = FeatureUnion([('word_ngrams', TfidfVectorizer(ngram_range = (1, 5),
                                                            analyzer = 'word',
                                                            min_df = 3,
                                                            tokenizer = identity)),
                            ('char_ngrams', CountVectorizer(ngram_range = (1, 8),
                                                            analyzer = 'char',
                                                            min_df = 3,
                                                            tokenizer = identity))])
 
print("fitting")
cl_gen = Pipeline([('vec', vec_gen),  ('cls', SVC())])
cl_gen.fit(tweets_train, labels_train)
pred = cl_gen.predict(tweets_test)
print(pred)
output_scores(["m", "f"], labels_test, pred)
#acc = accuracy_score(gender_test_labels, pred)
#print(acc)
