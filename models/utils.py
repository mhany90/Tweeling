
"""
This files contains some frequently
used methods for data preprocessing 
and classification that we kept copying.
You can also set the `n' for n-fold
cross-validation here (the NSPL constant).
"""

import sys
sys.path.append('../tools')
import random
random.seed(42)

from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from twokenize import *
import nltk

## n for n-fold cross-validation
NSPL = 5

## A dummy function that just returns its input
def identity(x):
    return x

#Tokenizer function to extract the first letters(s) from a word
def first_letter(x):
    new_x = ' '.join([word[:1] for word in x])
    return new_x

#POS tokenizer function from LFD slides
def pos_token(tokens):
    tokens2 = tokenizeRawTweetText(tokens)
    #print([token +" " + tag for token, tag in nltk.pos_tag(tokens2)])
    return [tag for token, tag in nltk.pos_tag(tokens2)]


def preproc(x):
    """
    x is one user's data = list of tweets.
    We create a single document from these and
    apply preprocessing steps as described in
    Overview of the 5th Author Profiling Task at PAN 2017
    """
    return ' '.join([t.strip() for t in x])

def classify(classifier, data, labeltype):
    return cross_val_predict(classifier, data, labeltype, cv = NSPL)

def get_metrics(true, predicted):
    return({'acc' : metrics.accuracy_score(true, predicted),
            'fsc' : metrics.f1_score(true, predicted, average = 'weighted')})

## A function to print confusion matrix in a nice way
def pretty_print_cm(cm, class_labels):
    row_format = "{:>5}" * (len(class_labels) + 1)
    print(row_format.format("", *class_labels))
    for l1, row in zip(class_labels, cm):
        print(row_format.format(l1, *row))
