#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 17:58:29 2022

@author: nayzaw
"""

import numpy as np
import pandas as pd
import datetime
from textblob import TextBlob
import nltk
from nltk import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
#from sentence_transformers import SentenceTransformer, util
#from transformers import AutoTokenizer
from addictionStudy import getTopNGrams
import operator
import re


def getNGrams(df, col, n): # n is the 'n' in 'n-grams'
    ndiction = {}
    textbody = df[col]
    for text in textbody: # take each text/entry in df[col] and tokenize it
        #text = re.sub(r"\S*https?:\S*", "", text)
        text = re.sub(r'http\S+', '', text)
        candidates = word_tokenize(text)
        tokens = []
        stop_words = stopwords.words('english')
        stop_words.extend(['www', 'nlm', 'ncbi', 'x200b', 'pubmed', 'amp',
                           'pjpg', 'com', 'r', 'stopsmoking', 'azvz9tuefvotx9tr2zmhawmpgduxus',
                           'yqjqftiisfbn91zmhwjcxvcfncgylnlkcwbvd'])
        for tk in candidates:
            if tk not in stop_words and tk.isalnum():
                tokens.append(tk)
        n_grams = ngrams(tokens, n)
        for grams in n_grams:
            joined_grams = '-'.join(grams)
            if joined_grams in ndiction:
                ndiction[joined_grams] += 1
            else:
                ndiction[joined_grams] = 1
    sorted_grams_lst = sorted(ndiction.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_grams_lst
    

df_C = pd.read_csv('stopsmoking_allcomments_LIWC.csv')
df_S = pd.read_csv('stopsmoking_allsubmissions_LIWC.csv')

# splitting comments and submissions again into before Covid and after Covid
beforedf_C =df_C.loc[df_C['created_utc'] < datetime.datetime(2020, 3, 10, 0, 0).timestamp()]
afterdf_C =df_C.loc[df_C['created_utc'] > datetime.datetime(2020, 3, 11, 0, 0).timestamp()]

beforedf_S =df_S.loc[df_S['created_utc'] < datetime.datetime(2020, 3, 10, 0, 0).timestamp()]
afterdf_S =df_S.loc[df_S['created_utc'] > datetime.datetime(2020, 3, 11, 0, 0).timestamp()]


# FIRST use submissions file to get top 10 4-grams (or 5-grams)

# dropping the duplicates on submissions file
dfS = df_S.drop_duplicates(subset=['selftext'])

# calling the getNGrams function to get the NGrams for the submissions file
five_gramS = getNGrams(dfS, 'selftext', 5) # 5-grams
four_gramS = getNGrams(dfS, 'selftext', 4) # 4-grams

# getting top 10 of both 5-grams and 4-grams
top10_fiveGramS = getTopNGrams(five_gramS, 10)
top10_fourGramS = getTopNGrams(four_gramS, 10)




